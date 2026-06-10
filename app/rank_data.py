"""Load candidate / feature lookups for ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def load_feature_lookup(features_path: Path) -> dict[str, dict[str, Any]]:
    frame = pl.read_parquet(features_path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in frame.iter_rows(named=True):
        lookup[str(row["candidate_id"])] = row
    return lookup


def load_candidates_lookup(candidates_path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            lookup[str(candidate["candidate_id"])] = candidate
    return lookup
