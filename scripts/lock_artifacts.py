#!/usr/bin/env python3
"""Validate and lock preprocessed artifacts — prevents accidental Gemini re-runs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifacts_manifest import assert_ready_for_preprocess_lock, write_manifest  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lock artifacts/ after a successful full preprocess (no Gemini re-spend)."
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--unlock", action="store_true", help="Write manifest with locked=false.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    artifacts_dir = (args.artifacts or settings.artifacts_dir).resolve()

    try:
        if not args.unlock:
            assert_ready_for_preprocess_lock(artifacts_dir)
        manifest = write_manifest(artifacts_dir, locked=not args.unlock)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if manifest.get("locked"):
        logger.info(
            "Artifacts locked. Commit artifacts/ to git (use Git LFS for *.npy, faiss.index). "
            "Future preprocess runs skip Gemini unless you use --force-llm."
        )
    else:
        logger.info("Artifacts unlocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
