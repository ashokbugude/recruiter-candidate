"""Candidate loading utilities."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CandidateRecord = dict[str, Any]


def load_candidates(path: Path, *, limit: int | None = None) -> Iterator[CandidateRecord]:
    """
    Stream candidates from JSONL (one object per line) or JSON array file.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if file format is unsupported or empty.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Candidates file is empty: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _load_jsonl(path, limit=limit)
    elif suffix == ".json":
        yield from _load_json_array(path, limit=limit)
    else:
        raise ValueError(f"Unsupported candidates format '{suffix}': {path}")


def load_candidates_list(path: Path, *, limit: int | None = None) -> list[CandidateRecord]:
    """Materialize all candidates into a list (use sparingly on large files)."""
    return list(load_candidates(path, limit=limit))


def count_candidates(path: Path) -> int:
    """Count candidates without fully parsing JSON objects."""
    path = path.resolve()
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    records = load_candidates_list(path)
    return len(records)


def _load_jsonl(path: Path, *, limit: int | None) -> Iterator[CandidateRecord]:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
            yield record
            count += 1
            if limit is not None and count >= limit:
                return


def _load_json_array(path: Path, *, limit: int | None) -> Iterator[CandidateRecord]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    if not payload:
        raise ValueError(f"Candidates array is empty: {path}")
    for idx, record in enumerate(payload):
        if limit is not None and idx >= limit:
            return
        yield record
