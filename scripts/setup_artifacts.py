#!/usr/bin/env python3
"""Bootstrap artifacts directory with challenge reference documents."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

REFERENCE_DOCS: tuple[str, ...] = (
    "job_description.txt",
    "submission_spec.txt",
    "redrob_signals_doc.txt",
    "README_challenge.txt",
)


def ensure_reference_docs(settings, *, force: bool = False) -> list[Path]:
    """Ensure challenge reference documents exist under artifacts/."""
    settings.ensure_artifacts_dir()
    present: list[Path] = []

    for name in REFERENCE_DOCS:
        dest = settings.artifacts_dir / name
        if dest.exists():
            logger.info("Present: %s", dest.name)
            present.append(dest)
            continue
        if force:
            logger.warning("Missing (not auto-copied): %s", dest)
        else:
            logger.warning("Missing: %s — copy from challenge bundle or run preprocess", dest)

    return present


def write_gitkeep(artifacts_dir: Path) -> None:
    """Ensure artifacts/.gitkeep exists for empty-dir tracking."""
    gitkeep = artifacts_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap artifacts directory.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Report missing reference docs (no automatic copy).",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    write_gitkeep(settings.artifacts_dir)
    present = ensure_reference_docs(settings, force=args.force)

    if len(present) < len(REFERENCE_DOCS):
        logger.error(
            "Artifacts incomplete (%d/%d reference docs). See artifacts/ARTIFACTS.md.",
            len(present),
            len(REFERENCE_DOCS),
        )
        return 1

    logger.info("Artifacts ready at %s", settings.artifacts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
