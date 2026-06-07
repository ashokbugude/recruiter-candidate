#!/usr/bin/env python3
"""Parse job description into structured requirements JSON."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.jd_requirements import load_or_build_jd_requirements  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse JD into jd_requirements.json")
    parser.add_argument("--force", action="store_true", help="Overwrite cached output.")
    parser.add_argument("--jd", type=Path, default=None, help="Job description text file.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_artifacts_dir()

    jd_path = (args.jd or settings.job_description_path).resolve()
    out_path = (args.out or settings.jd_requirements_path).resolve()

    requirements = load_or_build_jd_requirements(
        jd_path, out_path, settings=settings, force=args.force
    )
    logger.info(
        "JD requirements ready: %d must-have skills, source=%s",
        len(requirements.must_have_skills),
        requirements.source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
