#!/usr/bin/env python3
"""Offline preprocessing orchestrator — Phase 2."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifacts_manifest import is_locked, write_manifest  # noqa: E402
from app.bm25_index import build_bm25_index, save_bm25  # noqa: E402
from app.candidates import count_candidates, load_candidates, load_candidates_list  # noqa: E402
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
from app.gemini_client import gemini_jd_enabled, gemini_labels_enabled  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.progress import format_duration  # noqa: E402

logger = logging.getLogger(__name__)

STEPS = ("jd", "features", "embeddings", "bm25", "labels", "all")


def _candidate_total(candidates_path: Path, limit: int | None) -> int:
    if limit is not None:
        return limit
    return count_candidates(candidates_path)


def _features_stale(settings) -> bool:
    """True when JD was rebuilt after the feature parquet (skill/title features depend on JD)."""
    feat_path = settings.artifact_path("candidate_features.parquet")
    jd_path = settings.jd_requirements_path
    if not feat_path.exists() or not jd_path.exists():
        return False
    return jd_path.stat().st_mtime_ns > feat_path.stat().st_mtime_ns


def run_jd(settings, artifacts_dir: Path, *, force: bool = False, force_llm: bool = False) -> None:
    jd_path = settings.jd_requirements_path
    if jd_path.exists() and not force:
        if gemini_jd_enabled(settings):
            try:
                cached = json.loads(jd_path.read_text(encoding="utf-8"))
                if cached.get("source") != "gemini_pro":
                    logger.info(
                        "Cached JD source=%s but Gemini JD enabled — rebuilding with Gemini Pro",
                        cached.get("source", "?"),
                    )
                    force = True
                else:
                    logger.info("Using cached gemini_pro JD requirements at %s", jd_path)
                    return
            except (OSError, json.JSONDecodeError):
                force = True
        else:
            logger.info("Using cached JD requirements at %s", jd_path)
            return
    if not gemini_jd_enabled(settings):
        logger.info("Gemini JD parse disabled — building heuristic JD requirements")
    load_or_build_jd_requirements(
        settings.job_description_path,
        settings.jd_requirements_path,
        settings=settings,
        force=force,
    )


def run_features(settings, candidates_path: Path, *, limit: int | None = None, force: bool = False) -> None:
    out_path = settings.artifact_path("candidate_features.parquet")
    if out_path.exists() and not force:
        if _features_stale(settings):
            logger.info("JD requirements newer than features — rebuilding feature parquet")
        else:
            logger.info("Features exist at %s — skipping (use --force)", out_path)
            return

    total = _candidate_total(candidates_path, limit)
    logger.info("Feature extraction: %d candidates", total)

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
    frame = build_features_frame(
        candidates,
        jd,
        silver_tiers=silver,
        gemini_tiers=gemini,
        total_candidates=total,
    )
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

    from app.embeddings import default_batch_size, resolve_embedding_device

    candidates = load_candidates_list(candidates_path, limit=limit)
    resolved = resolve_embedding_device(device)
    effective_batch = batch_size or default_batch_size(resolved)
    total = len(candidates)
    batches = (total + effective_batch - 1) // effective_batch if total else 0
    logger.info(
        "Embeddings: encoding %d candidates on %s (batch_size=%d, ~%d batches, tqdm bar below)",
        total,
        resolved,
        effective_batch,
        batches,
    )
    t0 = time.perf_counter()
    matrix, ids = encode_candidates(
        candidates,
        show_progress=total > 100,
        device=resolved,
        batch_size=batch_size,
    )
    logger.info("Embeddings: encode finished in %s", format_duration(time.perf_counter() - t0))
    t1 = time.perf_counter()
    save_embeddings(matrix, ids, npy_path=npy_path, index_path=index_path)
    logger.info("Embeddings: saved FAISS + npy in %s", format_duration(time.perf_counter() - t1))


def run_bm25(settings, candidates_path: Path, *, limit: int | None = None, force: bool = False) -> None:
    bm25_path = settings.artifact_path("bm25.pkl")
    if bm25_path.exists() and not force:
        logger.info("BM25 index exists — skipping (use --force)")
        return

    total = _candidate_total(candidates_path, limit)
    logger.info("BM25: building index for %d candidates", total)
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

    total = _candidate_total(candidates_path, limit)
    if gemini_labels_enabled(settings):
        logger.info("Gemini archetype labels: %d candidates (API enabled)", total)
    else:
        logger.info("Silver-tier labels only: %d candidates (Gemini labels disabled)", total)

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
        skip_api=not gemini_labels_enabled(settings) or (is_locked(artifacts_dir) and not force_llm),
    )


def _run_named_step(
    name: str,
    fn: Callable[[], None],
    *,
    index: int | None = None,
    total_steps: int | None = None,
) -> None:
    if index is not None and total_steps is not None:
        logger.info("=== Pipeline step %d/%d: %s ===", index, total_steps, name)
    else:
        logger.info("=== Step: %s ===", name)
    t0 = time.perf_counter()
    fn()
    logger.info("=== %s complete (elapsed %s) ===", name, format_duration(time.perf_counter() - t0))


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
        help="Disable all Gemini calls (same as --no-gemini-jd --no-gemini-labels).",
    )
    parser.add_argument(
        "--gemini-jd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Gemini Pro JD parse (default: from REDROB_GEMINI_JD_PARSE).",
    )
    parser.add_argument(
        "--gemini-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Gemini Flash 100K labels (default: from REDROB_GEMINI_LABELS).",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Re-run Gemini archetype labels even when gemini_tiers.parquet exists.",
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
    settings_updates: dict = {}
    if args.skip_llm:
        settings_updates["skip_gemini"] = True
    if args.gemini_jd is not None:
        settings_updates["gemini_jd_parse"] = args.gemini_jd
    if args.gemini_labels is not None:
        settings_updates["gemini_labels"] = args.gemini_labels
    if settings_updates:
        settings = settings.model_copy(update=settings_updates)
    configure_logging(settings.log_level)
    settings.ensure_artifacts_dir()
    artifacts_dir = settings.artifacts_dir.resolve()

    candidates_path = (args.candidates or settings.candidates_path).resolve()
    step = args.step
    pipeline_t0 = time.perf_counter()

    if step == "all":
        total_candidates = _candidate_total(candidates_path, args.limit)
        logger.info(
            "Preprocess pipeline: %d candidates | gemini_jd=%s | gemini_labels=%s | force=%s | force_llm=%s",
            total_candidates,
            gemini_jd_enabled(settings),
            gemini_labels_enabled(settings),
            args.force,
            args.force_llm,
        )
        planned: list[tuple[str, Callable[[], None]]] = [
            ("JD parse", lambda: run_jd(settings, artifacts_dir, force=args.force, force_llm=args.force_llm)),
            (
                "Gemini archetype labels",
                lambda: run_labels(
                    settings,
                    candidates_path,
                    artifacts_dir,
                    limit=args.limit,
                    force=args.force,
                    force_llm=args.force_llm,
                ),
            ),
            (
                "Feature extraction",
                lambda: run_features(settings, candidates_path, limit=args.limit, force=args.force),
            ),
            (
                "BGE embeddings + FAISS",
                lambda: run_embeddings(
                    settings,
                    candidates_path,
                    limit=args.limit,
                    force=args.force,
                    device=args.device,
                    batch_size=args.batch_size,
                ),
            ),
            ("BM25 index", lambda: run_bm25(settings, candidates_path, limit=args.limit, force=args.force)),
        ]
        for idx, (name, fn) in enumerate(planned, start=1):
            _run_named_step(name, fn, index=idx, total_steps=len(planned))
        if not args.force_llm:
            write_manifest(artifacts_dir, locked=is_locked(artifacts_dir))
        logger.info(
            "Preprocess pipeline complete — total elapsed %s",
            format_duration(time.perf_counter() - pipeline_t0),
        )
        return 0

    if step == "jd":
        _run_named_step("JD parse", lambda: run_jd(settings, artifacts_dir, force=args.force, force_llm=args.force_llm))
    elif step == "labels":
        _run_named_step(
            "Gemini archetype labels",
            lambda: run_labels(
                settings,
                candidates_path,
                artifacts_dir,
                limit=args.limit,
                force=args.force,
                force_llm=args.force_llm,
            ),
        )
    elif step == "features":
        _run_named_step(
            "Feature extraction",
            lambda: run_features(settings, candidates_path, limit=args.limit, force=args.force),
        )
    elif step == "embeddings":
        _run_named_step(
            "BGE embeddings + FAISS",
            lambda: run_embeddings(
                settings,
                candidates_path,
                limit=args.limit,
                force=args.force,
                device=args.device,
                batch_size=args.batch_size,
            ),
        )
    elif step == "bm25":
        _run_named_step(
            "BM25 index",
            lambda: run_bm25(settings, candidates_path, limit=args.limit, force=args.force),
        )

    logger.info("Preprocessing step '%s' complete (elapsed %s).", step, format_duration(time.perf_counter() - pipeline_t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
