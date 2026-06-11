"""Submission CSV formatting (submission_spec §2)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

SUBMISSION_FIELDS = ("candidate_id", "rank", "score", "reasoning")


def format_submission_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(SUBMISSION_FIELDS))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_submission(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_submission_csv(rows), encoding="utf-8")
