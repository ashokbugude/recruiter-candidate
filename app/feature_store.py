"""Build and persist candidate feature parquet."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl

from app.constants import REFERENCE_DATE
from app.features import FEATURE_NAMES, extract_features
from app.jd_requirements import JDRequirements
from app.labels.honeypots import detect_honeypot
from app.progress import ProgressTracker

logger = logging.getLogger(__name__)


def load_silver_tiers(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    frame = pl.read_parquet(path)
    if "candidate_id" not in frame.columns or "tier" not in frame.columns:
        return {}
    return dict(zip(frame["candidate_id"].to_list(), frame["tier"].to_list(), strict=False))


def build_features_frame(
    candidates: Iterator[dict[str, Any]],
    jd: JDRequirements,
    *,
    silver_tiers: dict[str, int] | None = None,
    gemini_tiers: dict[str, int] | None = None,
    reference_date=None,
    total_candidates: int | None = None,
) -> pl.DataFrame:
    """Build feature DataFrame for all candidates."""
    ref = reference_date or REFERENCE_DATE
    silver_tiers = silver_tiers or {}
    gemini_tiers = gemini_tiers or {}

    rows: list[dict[str, Any]] = []
    progress = None
    if total_candidates:
        log_every = max(total_candidates // 20, 1)
        progress = ProgressTracker(
            logger,
            label="Feature extraction",
            total=total_candidates,
            log_every=log_every,
            unit="candidates",
        )

    for candidate in candidates:
        cid = str(candidate["candidate_id"])
        feats = extract_features(candidate, jd, reference_date=ref)
        hp = detect_honeypot(candidate, reference_date=ref)
        row: dict[str, Any] = {"candidate_id": cid, **feats}
        row["silver_tier"] = int(silver_tiers.get(cid, -1))
        row["gemini_tier"] = int(gemini_tiers.get(cid, -1))
        row["is_honeypot"] = hp.is_honeypot
        row["is_trap"] = hp.is_trap
        row["should_exclude"] = hp.should_exclude
        rows.append(row)
        if progress:
            progress.tick()

    if progress:
        progress.finish()

    return pl.DataFrame(rows)


def write_features_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    logger.info("Wrote %d feature rows (%d cols) to %s", frame.height, len(frame.columns), path)


def validate_features_frame(frame: pl.DataFrame) -> None:
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if frame["candidate_id"].n_unique() != frame.height:
        raise ValueError("Duplicate candidate_id values in feature frame")
