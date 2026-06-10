"""Precomputed career-text recall scores (plain-language tier-5 profiles)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.artifact_names import CAREER_SCORES as CAREER_SCORES_NAME


@lru_cache(maxsize=8)
def load_career_scores(artifacts_dir: str) -> dict[str, float]:
    path = Path(artifacts_dir) / CAREER_SCORES_NAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in payload.get("scores", {}).items()}


def career_recall_ranking(artifacts_dir: Path, *, top_k: int = 4000) -> list[str]:
    """Return candidate IDs ranked by precomputed career IR relevance."""
    scores = load_career_scores(str(artifacts_dir.resolve()))
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [cid for cid, _ in ranked[:top_k]]


def career_score(candidate_id: str, artifacts_dir: Path) -> float:
    return load_career_scores(str(artifacts_dir.resolve())).get(str(candidate_id), 0.0)
