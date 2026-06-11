#!/usr/bin/env python3
"""Pre-rank bundled sample candidates for instant sandbox CSV (submission_spec §10.5)."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifacts_check import validate_artifacts  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.rank_submission import rank_candidates_file  # noqa: E402
from app.submission_csv import write_submission  # noqa: E402

logger = logging.getLogger(__name__)

SAMPLE_CSV_NAME = "sample_ranked.csv"


def bake_sample_ranking(*, artifacts_dir: Path | None = None, out_name: str = SAMPLE_CSV_NAME) -> Path:
    settings = get_settings()
    configure_logging(settings.log_level)
    artifacts_dir = (artifacts_dir or settings.artifacts_dir).resolve()
    sample_path = settings.sample_candidates_path.resolve()

    validate_artifacts(artifacts_dir)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample candidates: {sample_path}")

    records = json.loads(sample_path.read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for row in records:
            tmp.write(json.dumps(row) + "\n")
        jsonl_path = Path(tmp.name)

    try:
        results = rank_candidates_file(
            jsonl_path,
            artifacts_dir=artifacts_dir,
            top_k=min(100, len(records)),
        )
    finally:
        jsonl_path.unlink(missing_ok=True)

    out_path = artifacts_dir / out_name
    write_submission(results, out_path)
    logger.info("Wrote pre-baked sandbox CSV: %s (%d rows)", out_path, len(results))
    return out_path


def main() -> int:
    try:
        bake_sample_ranking()
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
