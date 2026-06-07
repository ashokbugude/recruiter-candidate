#!/usr/bin/env python3
"""Offline preprocessing orchestrator — Phase 2."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifacts_manifest import is_locked, write_manifest  # noqa: E402
from app.bm25_index import build_bm25_index, save_bm25  # noqa: E402
from app.candidates import load_candidates, load_candidates_list  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embeddings import encode_candidates, save_embeddings  # noqa: E402
from app.feature_store import (  # noqa: E402
    build_features_frame,
    load_silver_tiers,
    validate_features_frame,
    write_features_parquet,
)
from app.features import FEATURE_NAMES  # noqa: E402
from app.jd_requirements import load_or_build_jd_requirements  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

STEPS = ("jd", "features", "embeddings", "bm25", "labels", "all")


def _skip_llm_calls(settings, artifacts_dir: Path, *, force_llm: bool = False) -> bool:
    if force_llm:
        return False
    if settings.skip_gemini:
        return True
    if is_locked(artifacts_dir):
        return True
    return False


def run_jd(settings, artifacts_dir: Path, *, force: bool = False, force_llm: bool = False) -> None:
    if _skip_llm_calls(settings, artifacts_dir, force_llm=force_llm) and settings.jd_requirements_path.exists():
        logger.info("Skipping Gemini JD parse (locked/skip-llm) — using %s", settings.jd_requirements_path)
        return
    load_or_build_jd_requirements(
        settings.job_description_path,
        settings.jd_requirements_path,
        settings=settings,
        force=force or force_llm,
    )


def run_features(settings, candidates_path: Path, *, limit: int | None = None, force: bool = False) -> None:
    out_path = settings.artifact_path("candidate_features.parquet")
    if out_path.exists() and not force:
        logger.info("Features exist at %s — skipping (use --force)", out_path)
        return

    jd = load_or_build_jd_requirements(
        settings.job_description_path,
        settings.jd_requirements_path,
        settings=settings,
    )
    silver = load_silver_tiers(settings.artifact_path("labels_silver.parquet"))
    gemini_path = settings.artifact_path("gemini_tiers.parquet")
    gemini: dict[str, int] = {}
    if gemini_path.exists():
        import polars as pl

        gdf = pl.read_parquet(gemini_path)
        gemini = dict(zip(gdf["candidate_id"].to_list(), gdf["gemini_tier"].to_list(), strict=False))

    candidates = load_candidates(candidates_path, limit=limit)
    frame = build_features_frame(candidates, jd, silver_tiers=silver, gemini_tiers=gemini)
    validate_features_frame(frame)
    write_features_parquet(frame, out_path)
    logger.info("Feature columns: %d features + metadata", len(FEATURE_NAMES))


def run_embeddings(
    settings,
    candidates_path: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    device: str | None = None,
    batch_size: int | None = None,
) -> None:
    npy_path = settings.artifact_path("bge_embeddings.npy")
    index_path = settings.artifact_path("faiss.index")
    if npy_path.exists() and index_path.exists() and not force:
        logger.info("Embeddings exist — skipping (use --force)")
        return

    from app.embeddings import resolve_embedding_device

    candidates = load_candidates_list(candidates_path, limit=limit)
    resolved = resolve_embedding_device(device)
    logger.info("Encoding %d candidates with BGE-small on %s...", len(candidates), resolved)
    matrix, ids = encode_candidates(
        candidates,
        show_progress=len(candidates) > 100,
        device=resolved,
        batch_size=batch_size,
    )
    save_embeddings(matrix, ids, npy_path=npy_path, index_path=index_path)


def run_bm25(settings, candidates_path: Path, *, limit: int | None = None, force: bool = False) -> None:
    bm25_path = settings.artifact_path("bm25.pkl")
    if bm25_path.exists() and not force:
        logger.info("BM25 index exists — skipping (use --force)")
        return

    candidates = load_candidates_list(candidates_path, limit=limit)
    artifacts = build_bm25_index(candidates)
    save_bm25(artifacts, bm25_path)


def run_labels(
    settings,
    candidates_path: Path,
    artifacts_dir: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    force_llm: bool = False,
) -> None:
    from scripts.label_archetypes import build_gemini_tiers

    out_path = settings.artifact_path("gemini_tiers.parquet")
    if out_path.exists() and not force and not force_llm:
        logger.info("Gemini tiers exist at %s — skipping (use --force-llm to re-label)", out_path)
        return
    if _skip_llm_calls(settings, artifacts_dir, force_llm=force_llm) and out_path.exists():
        logger.info("Skipping Gemini labels (locked/skip-llm) — reusing %s", out_path)
        return

    jd = load_or_build_jd_requirements(
        settings.job_description_path,
        settings.jd_requirements_path,
        settings=settings,
    )
    build_gemini_tiers(
        candidates_path,
        out_path,
        settings=settings,
        jd=jd,
        limit=limit,
        force=force or force_llm,
        skip_api=_skip_llm_calls(settings, artifacts_dir, force_llm=force_llm),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline preprocessing pipeline (Phase 2).")
    parser.add_argument(
        "--step",
        choices=STEPS,
        default="all",
        help="Pipeline step to run (default: all).",
    )
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit candidates for quick runs.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if artifacts exist.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Never call Gemini (reuse cached JD/labels; no API spend).",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Re-run Gemini JD parse + archetype labels even when artifacts are locked.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Embedding device override (default: auto-detect CUDA).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size override (default: 256 on CUDA, 64 on CPU).",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.skip_llm:
        settings = settings.model_copy(update={"skip_gemini": True})
    configure_logging(settings.log_level)
    settings.ensure_artifacts_dir()
    artifacts_dir = settings.artifacts_dir.resolve()

    candidates_path = (args.candidates or settings.candidates_path).resolve()
    step = args.step

    if step in ("jd", "all"):
        logger.info("=== Step: JD parse ===")
        run_jd(settings, artifacts_dir, force=args.force, force_llm=args.force_llm)
    if step in ("labels", "all"):
        logger.info("=== Step: Gemini archetype labels ===")
        run_labels(
            settings,
            candidates_path,
            artifacts_dir,
            limit=args.limit,
            force=args.force,
            force_llm=args.force_llm,
        )
    if step in ("features", "all"):
        logger.info("=== Step: Feature extraction ===")
        run_features(settings, candidates_path, limit=args.limit, force=args.force)
    if step in ("embeddings", "all"):
        logger.info("=== Step: BGE embeddings + FAISS ===")
        run_embeddings(
            settings,
            candidates_path,
            limit=args.limit,
            force=args.force,
            device=args.device,
            batch_size=args.batch_size,
        )
    if step in ("bm25", "all"):
        logger.info("=== Step: BM25 index ===")
        run_bm25(settings, candidates_path, limit=args.limit, force=args.force)

    if step == "all" and not args.force_llm:
        write_manifest(artifacts_dir, locked=is_locked(artifacts_dir))

    logger.info("Preprocessing step '%s' complete.", step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
