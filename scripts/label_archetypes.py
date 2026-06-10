#!/usr/bin/env python3
"""Batch Gemini archetype labeling (tier 0-5) — offline only."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.candidates import load_candidates, load_candidates_list  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.feature_store import load_silver_tiers  # noqa: E402
from app.gemini_client import (  # noqa: E402
    FLASH_MODEL_FALLBACKS,
    gemini_labels_enabled,
    generate_json,
    has_gemini_auth,
    load_prompt_template,
    resolve_flash_model,
)
from app.jd_requirements import JDRequirements  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.progress import ProgressTracker, format_duration  # noqa: E402

logger = logging.getLogger(__name__)

def _batch_size(settings) -> int:
    return int(getattr(settings, "gemini_label_batch_size", 50))


def _sleep_seconds(settings) -> float:
    return float(getattr(settings, "gemini_label_sleep_seconds", 0.2))


def _compact_candidate(candidate: dict) -> dict:
    profile = candidate.get("profile") or {}
    return {
        "candidate_id": candidate["candidate_id"],
        "title": profile.get("current_title"),
        "headline": profile.get("headline"),
        "summary": (profile.get("summary") or "")[:400],
        "yoe": profile.get("years_of_experience"),
        "country": profile.get("country"),
        "skills": [s.get("name") for s in (candidate.get("skills") or [])[:12]],
        "career_titles": [r.get("title") for r in (candidate.get("career_history") or [])[:4]],
    }


def label_batch_with_gemini(
    batch: list[dict],
    *,
    job_summary: str,
    settings,
) -> dict[str, int]:
    template = load_prompt_template("label_archetype.txt")
    prompt = (
        template.replace("{{JOB_SUMMARY}}", job_summary)
        .replace("{{CANDIDATES_JSON}}", json.dumps([_compact_candidate(c) for c in batch], indent=0))
    )
    payload = generate_json(
        prompt,
        settings=settings,
        model=resolve_flash_model(settings),
        temperature=0.0,
        model_fallbacks=FLASH_MODEL_FALLBACKS,
    )
    tiers: dict[str, int] = {}
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "candidate_id" in item:
                tiers[str(item["candidate_id"])] = int(item.get("tier", 2))
    return tiers


def _rows_from_batch(
    batch: list[dict],
    gemini_tiers: dict[str, int],
    silver_tiers: dict[str, int],
) -> list[dict]:
    rows: list[dict] = []
    for c in batch:
        c_id = str(c["candidate_id"])
        rows.append(
            {
                "candidate_id": c_id,
                "gemini_tier": gemini_tiers.get(c_id, silver_tiers.get(c_id, 2)),
                "label_source": "gemini_flash" if c_id in gemini_tiers else "silver_fallback",
            }
        )
    return rows


def _silver_fallback_rows(batch: list[dict], silver_tiers: dict[str, int]) -> list[dict]:
    return [
        {
            "candidate_id": str(c["candidate_id"]),
            "gemini_tier": silver_tiers.get(str(c["candidate_id"]), 2),
            "label_source": "silver_fallback",
        }
        for c in batch
    ]


def build_gemini_tiers(
    candidates_path: Path,
    output_path: Path,
    *,
    settings,
    jd: JDRequirements,
    limit: int | None = None,
    force: bool = False,
    skip_api: bool = False,
) -> pl.DataFrame:
    if output_path.exists() and not force:
        logger.info("Loading cached Gemini tiers from %s", output_path)
        return pl.read_parquet(output_path)

    use_gemini = not skip_api and gemini_labels_enabled(settings) and has_gemini_auth(settings)
    if skip_api:
        logger.info("skip_api=True — writing silver-tier labels only (no Gemini spend)")
    elif not use_gemini:
        logger.warning("No Gemini credentials — writing silver-tier fallback labels")
    silver_path = settings.artifact_path("labels_silver.parquet")
    silver_tiers = load_silver_tiers(silver_path)

    job_summary = (
        f"Role: {jd.role_title}. Must-have: {', '.join(jd.must_have_skills[:8])}. "
        f"YOE: {jd.yoe_min}-{jd.yoe_max}. Country: {jd.preferred_country}."
    )

    rows: list[dict] = []
    batch: list[dict] = []
    candidates = load_candidates_list(candidates_path, limit=limit) if limit else list(load_candidates(candidates_path))
    total_candidates = len(candidates)
    batch_size = _batch_size(settings)
    sleep_seconds = _sleep_seconds(settings)
    total_batches = (total_candidates + batch_size - 1) // batch_size if use_gemini else 0

    if use_gemini:
        model = resolve_flash_model(settings)
        logger.info(
            "Gemini labeling: %d candidates in %d batches (batch_size=%d, model=%s)",
            total_candidates,
            total_batches,
            batch_size,
            model,
        )
        batch_progress = ProgressTracker(
            logger, label="Gemini labels", total=total_batches, log_every=1, unit="batches"
        )

        def flush_batch() -> None:
            nonlocal batch
            if not batch:
                return
            t0 = time.perf_counter()
            try:
                gemini_tiers = label_batch_with_gemini(batch, job_summary=job_summary, settings=settings)
                rows.extend(_rows_from_batch(batch, gemini_tiers, silver_tiers))
            except Exception as exc:
                logger.warning("Gemini batch failed (%s); using silver fallback", exc)
                rows.extend(_silver_fallback_rows(batch, silver_tiers))
            batch_progress.tick()
            logger.debug("Batch API call took %s", format_duration(time.perf_counter() - t0))
            batch = []
            time.sleep(sleep_seconds)

        for candidate in candidates:
            batch.append(candidate)
            if len(batch) >= batch_size:
                flush_batch()

        flush_batch()
        batch_progress.finish(message="labeling complete")
    else:
        candidate_progress = ProgressTracker(
            logger,
            label="Silver tier labels",
            total=total_candidates,
            log_every=max(total_candidates // 20, 1),
            unit="candidates",
        )
        for candidate in candidates:
            cid = str(candidate["candidate_id"])
            rows.append(
                {
                    "candidate_id": cid,
                    "gemini_tier": silver_tiers.get(cid, 2),
                    "label_source": "silver_fallback",
                }
            )
            candidate_progress.tick()
        candidate_progress.finish()

    frame = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    gemini_count = frame.filter(pl.col("label_source") == "gemini_flash").height
    logger.info(
        "Wrote Gemini tiers: %d rows (%d gemini_flash, %d silver_fallback) → %s",
        frame.height,
        gemini_count,
        frame.height - gemini_count,
        output_path,
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch label candidate archetypes with Gemini Flash.")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit candidates (testing).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_artifacts_dir()

    from app.jd_requirements import load_or_build_jd_requirements

    jd = load_or_build_jd_requirements(
        settings.job_description_path,
        settings.jd_requirements_path,
        settings=settings,
    )
    candidates_path = (args.candidates or settings.candidates_path).resolve()
    out_path = (args.out or settings.artifact_path("gemini_tiers.parquet")).resolve()

    if not has_gemini_auth(settings):
        logger.warning("No Gemini credentials — writing silver-tier fallback labels")

    build_gemini_tiers(
        candidates_path,
        out_path,
        settings=settings,
        jd=jd,
        limit=args.limit,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
