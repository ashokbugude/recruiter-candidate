#!/usr/bin/env python3
"""Compare two submission CSVs — metric deltas and rank changes in top 100."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from scripts.audit_submission import audit_submission  # noqa: E402


def load_ranks(path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {r["candidate_id"]: int(r["rank"]) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare old vs new submission CSV.")
    parser.add_argument("old_csv", type=Path)
    parser.add_argument("new_csv", type=Path)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    artifacts_dir = (args.artifacts or settings.artifacts_dir).resolve()
    candidates_path = (args.candidates or settings.candidates_path).resolve()

    old_audit = audit_submission(args.old_csv.resolve(), artifacts_dir=artifacts_dir, candidates_path=candidates_path)
    new_audit = audit_submission(args.new_csv.resolve(), artifacts_dir=artifacts_dir, candidates_path=candidates_path)

    old_ranks = load_ranks(args.old_csv)
    new_ranks = load_ranks(args.new_csv)

    changes: list[dict] = []
    for cid in set(old_ranks) | set(new_ranks):
        o = old_ranks.get(cid)
        n = new_ranks.get(cid)
        if o is None or n is None or o != n:
            if (o is not None and o <= 100) or (n is not None and n <= 100):
                changes.append({"candidate_id": cid, "old_rank": o, "new_rank": n, "delta": (o or 999) - (n or 999)})

    changes.sort(key=lambda item: min(item["old_rank"] or 999, item["new_rank"] or 999))

    report = {
        "old": str(args.old_csv),
        "new": str(args.new_csv),
        "metric_deltas": {
            key: round(new_audit.get(key, 0) - old_audit.get(key, 0), 4)
            for key in (
                "proxy_ndcg_at_10",
                "ndcg_at_30",
                "tier5_in_top_10",
                "tier5_in_top_30",
                "tier5_in_top_100",
                "template_clones_top_30",
                "honeypots_in_top_100",
            )
            if isinstance(old_audit.get(key), (int, float))
        },
        "old_audit": old_audit,
        "new_audit": new_audit,
        "rank_changes_top100": changes[:100],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
