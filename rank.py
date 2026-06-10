#!/usr/bin/env python3
"""
Redrob candidate ranker — CLI entry point.

Reproduce command (submission spec §10.3):
    python rank.py --candidates ./challenge/candidates.jsonl --out ./team_sarva_automata.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from app.artifacts_check import ArtifactValidationError, validate_artifacts
from app.config import PROJECT_ROOT, get_settings
from app.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rank.py",
        description="Rank candidates for the Redrob Senior AI Engineer JD.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl (one JSON object per line).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "team_sarva_automata.csv",
        help="Portal submission CSV (use your registered team_xxx.csv name).",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifacts directory (default: ./artifacts).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level override.",
    )
    return parser


def validate_candidates_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Candidates file is empty: {path}")


def write_submission(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    artifacts_dir = (args.artifacts or settings.artifacts_dir).resolve()

    configure_logging(args.log_level or settings.log_level)
    logger.info("Redrob ranker starting")

    try:
        validate_candidates_file(args.candidates.resolve())
        warnings = validate_artifacts(artifacts_dir)
        for msg in warnings:
            logger.warning("%s", msg)
    except (FileNotFoundError, ValueError, ArtifactValidationError) as exc:
        logger.error("%s", exc)
        return 1

    from app.fusion import settings_with_fusion_params
    from app.pipeline import RankingPipeline

    settings = settings_with_fusion_params(settings, artifacts_dir)
    pipeline = RankingPipeline(settings, artifacts_dir)
    candidates_path = args.candidates.resolve()
    results = pipeline.rank(candidates_path)
    out_path = args.out.resolve()
    write_submission(results, out_path)
    logger.info("Wrote submission CSV: %s (%d rows)", out_path, len(results))

    from app.fusion import update_live_metrics

    ranked_ids = [row["candidate_id"] for row in results]
    update_live_metrics(
        artifacts_dir,
        ranked_ids,
        candidates_path,
        reference_date=settings.reference_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
