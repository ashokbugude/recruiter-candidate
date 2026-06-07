#!/usr/bin/env python3
"""
Redrob candidate ranker — CLI entry point.

Reproduce command (submission spec §10.3):
    python rank.py --candidates ./challenge/candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from app.config import get_settings
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
        required=True,
        help="Output submission CSV path.",
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


def validate_artifacts(artifacts_dir: Path) -> None:
    required = [
        "job_description.txt",
        "jd_requirements.json",
        "candidate_features.parquet",
        "bm25.pkl",
        "faiss.index",
        "bge_embeddings.npy",
        "candidate_id_index.json",
        "ltr_model.lgb",
    ]
    missing = [name for name in required if not (artifacts_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts: "
            + ", ".join(missing)
            + ". Run: python scripts/preprocess.py && python scripts/train_ltr.py"
        )


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
        validate_artifacts(artifacts_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    from app.pipeline import RankingPipeline

    pipeline = RankingPipeline(settings, artifacts_dir)
    results = pipeline.rank(args.candidates.resolve())
    write_submission(results, args.out.resolve())
    logger.info("Wrote submission CSV: %s (%d rows)", args.out, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
