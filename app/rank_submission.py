"""Shared ranking entry point for rank.py CLI and sandbox API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.fusion import settings_with_fusion_params
from app.pipeline import RankingPipeline


def rank_candidates_file(
    candidates_path: Path,
    *,
    artifacts_dir: Path | None = None,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Run the same ranking pipeline that produces team_sarva_automata.csv."""
    settings = settings or get_settings()
    artifacts_dir = (artifacts_dir or settings.artifacts_dir).resolve()
    settings = settings_with_fusion_params(settings, artifacts_dir)
    pipeline = RankingPipeline(settings, artifacts_dir)
    return pipeline.rank(candidates_path.resolve(), top_k=top_k)


def run_portal_submission_rank(
    *,
    artifacts_dir: Path | None = None,
    settings: Settings | None = None,
    candidates_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Identical to: python rank.py --candidates ./challenge/candidates.jsonl --out ./team_sarva_automata.csv"""
    settings = settings or get_settings()
    path = (candidates_path or settings.candidates_path).resolve()
    return rank_candidates_file(path, artifacts_dir=artifacts_dir, settings=settings, top_k=None)
