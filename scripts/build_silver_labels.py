#!/usr/bin/env python3
"""Build silver labels and candidate list artifacts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.candidates import load_candidates  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labels.silver import build_silver_labels, write_silver_labels  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build silver labels for LTR training.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Candidates file (default: settings.candidates_path).",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help="Output parquet path (default: artifacts/labels_silver.parquet).",
    )
    parser.add_argument(
        "--lists-out",
        type=Path,
        default=None,
        help="Output JSON path (default: artifacts/candidate_lists.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N candidates (for testing).",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_artifacts_dir()

    candidates_path = (args.candidates or settings.candidates_path).resolve()
    labels_path = (args.labels_out or settings.artifact_path("labels_silver.parquet")).resolve()
    lists_path = (args.lists_out or settings.artifact_path("candidate_lists.json")).resolve()

    logger.info("Building silver labels from %s", candidates_path)
    rows, lists = build_silver_labels(load_candidates(candidates_path, limit=args.limit))
    write_silver_labels(rows, lists, parquet_path=labels_path, lists_path=lists_path)

    logger.info(
        "Silver labels: %d rows | tier5=%d tier0=%d subtle_honeypots=%d should_exclude=%d",
        len(rows),
        sum(1 for r in rows if r.tier == 5),
        sum(1 for r in rows if r.tier == 0),
        sum(1 for r in rows if r.is_honeypot),
        sum(1 for r in rows if r.should_exclude),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
