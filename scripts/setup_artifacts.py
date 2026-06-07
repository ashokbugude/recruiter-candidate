#!/usr/bin/env python3
"""Bootstrap artifacts directory with challenge reference documents."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Source files inside challenge/ → artifacts/
DOC_MAPPINGS: dict[str, str] = {
    "job_description.txt": "challenge/_extracted/job_description.txt",
    "submission_spec.txt": "challenge/_extracted/submission_spec.txt",
    "redrob_signals_doc.txt": "challenge/_extracted/redrob_signals_doc.txt",
    "README_challenge.txt": "challenge/_extracted/README.txt",
}


def copy_reference_docs(settings, force: bool = False) -> list[Path]:
    """Copy challenge reference documents into artifacts/."""
    settings.ensure_artifacts_dir()
    copied: list[Path] = []

    for dest_name, src_rel in DOC_MAPPINGS.items():
        src = PROJECT_ROOT / src_rel
        dest = settings.artifacts_dir / dest_name

        if not src.exists():
            logger.warning("Source not found, skipping: %s", src)
            continue

        if dest.exists() and not force:
            logger.info("Already exists, skipping: %s", dest.name)
            copied.append(dest)
            continue

        shutil.copy2(src, dest)
        logger.info("Copied %s -> %s", src.name, dest)
        copied.append(dest)

    return copied


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
        help="Overwrite existing artifact documents.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    write_gitkeep(settings.artifacts_dir)
    copied = copy_reference_docs(settings, force=args.force)

    if not copied:
        logger.error("No documents copied. Run from project root after extracting challenge docs.")
        return 1

    logger.info("Artifacts ready at %s (%d files)", settings.artifacts_dir, len(copied))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
