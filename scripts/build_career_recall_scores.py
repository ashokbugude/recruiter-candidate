#!/usr/bin/env python3
"""Build career_recall_scores.json from candidate career history vs JD IR keywords."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifact_names import CAREER_SCORES  # noqa: E402
from app.candidates import load_candidates  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.constants import ML_PRODUCTION_KEYWORDS, SEARCH_RETRIEVAL_KEYWORDS  # noqa: E402
from app.jd_requirements import load_or_build_jd_requirements  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

TOP_K = 4000
MIN_SCORE = 0.35


def career_text(candidate: dict) -> str:
    parts: list[str] = []
    for role in candidate.get("career_history") or []:
        parts.append(str(role.get("title") or ""))
        parts.append(str(role.get("company") or ""))
        parts.append(str(role.get("description") or ""))
    return " ".join(parts).lower()


def score_career(text: str) -> float:
    if not text.strip():
        return 0.0
    ir_hits = sum(1 for kw in SEARCH_RETRIEVAL_KEYWORDS if kw in text)
    prod_hits = sum(1 for kw in ML_PRODUCTION_KEYWORDS if kw in text)
    raw = ir_hits * 0.12 + prod_hits * 0.04
    return min(1.0, raw)


def build_scores(candidates_path: Path, *, settings) -> dict[str, float]:
    jd = load_or_build_jd_requirements(
        settings.job_description_path,
        settings.jd_requirements_path,
        settings=settings,
    )
    _ = jd  # JD loaded for consistency; scoring uses IR keyword overlap on career text
    scored: list[tuple[str, float]] = []
    for candidate in load_candidates(candidates_path):
        cid = str(candidate["candidate_id"])
        score = score_career(career_text(candidate))
        if score >= MIN_SCORE:
            scored.append((cid, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    top = scored[:TOP_K]
    return dict(top)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build career recall score artifact.")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    candidates_path = (args.candidates or settings.candidates_path).resolve()
    out_path = (args.out or settings.artifacts_dir / CAREER_SCORES).resolve()

    scores = build_scores(candidates_path, settings=settings)
    payload = {"scores": scores, "top_k": TOP_K, "min_score": MIN_SCORE}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %d career scores to %s", len(scores), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
