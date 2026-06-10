#!/usr/bin/env python3
"""Audit submission CSV — honeypots, tier-5 recall, template clones, availability."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.fusion import ndcg_at_k  # noqa: E402
from app.ranking_modifiers import is_low_availability, is_senior_ai_summary_clone  # noqa: E402
from app.rank_data import load_candidates_lookup  # noqa: E402


def audit_submission(
    submission_path: Path,
    *,
    artifacts_dir: Path,
    candidates_path: Path,
) -> dict:
    rows = list(csv.DictReader(submission_path.open(encoding="utf-8")))
    ranked = sorted(rows, key=lambda r: int(r["rank"]))
    top_ids = [r["candidate_id"] for r in ranked[:100]]
    top10 = top_ids[:10]
    top30 = top_ids[:30]

    silver = pl.read_parquet(artifacts_dir / "labels_silver.parquet")
    tier_map = dict(zip(silver["candidate_id"].cast(pl.Utf8).to_list(), silver["tier"].to_list(), strict=False))
    exclude_map = dict(
        zip(silver["candidate_id"].cast(pl.Utf8).to_list(), silver["should_exclude"].to_list(), strict=False)
    )

    tier5_all = {cid for cid, t in tier_map.items() if t == 5}
    tier0_all = {cid for cid, t in tier_map.items() if t == 0}

    lookup = load_candidates_lookup(candidates_path)

    clones_top30 = sum(
        1 for cid in top30 if is_senior_ai_summary_clone(lookup.get(cid, {}))
    )
    low_avail_top30 = sum(1 for cid in top30 if is_low_availability(lookup.get(cid, {})))
    honeypots_top10 = sum(1 for cid in top10 if tier_map.get(cid) == 0 or exclude_map.get(cid))
    honeypots_top100 = sum(1 for cid in top_ids if tier_map.get(cid) == 0 or exclude_map.get(cid))

    research_top100 = sum(
        1
        for cid in top_ids
        if "research" in str((lookup.get(cid, {}).get("profile") or {}).get("current_title", "")).lower()
    )

    labels10 = [tier_map.get(cid, 2) for cid in top10]
    labels30 = [tier_map.get(cid, 2) for cid in top30]
    import numpy as np

    proxy_ndcg10 = ndcg_at_k(np.array(labels10, dtype=float), np.arange(10, 0, -1, dtype=float), k=10)
    proxy_ndcg30 = ndcg_at_k(np.array(labels30, dtype=float), np.arange(30, 0, -1, dtype=float), k=30)
    top10_fingerprint = {cid: int(r["rank"]) for cid, r in zip(top10, ranked[:10], strict=True)}

    lists_path = artifacts_dir / "candidate_lists.json"
    sample_t5_in_top10 = None
    if lists_path.exists():
        samples = json.loads(lists_path.read_text(encoding="utf-8")).get("samples", {})
        sample_t5 = set(samples.get("senior_ai_tier5", []))
        sample_t5_in_top10 = len(sample_t5 & set(top10))

    report = {
        "submission": str(submission_path),
        "rows": len(rows),
        "tier5_in_top_10": len(set(top10) & tier5_all),
        "tier5_in_top_30": len(set(top30) & tier5_all),
        "tier5_in_top_100": len(set(top_ids) & tier5_all),
        "tier5_pool_total": len(tier5_all),
        "honeypots_in_top_10": honeypots_top10,
        "honeypots_in_top_100": honeypots_top100,
        "honeypot_dq_risk": honeypots_top100 > 10,
        "template_clones_top_30": clones_top30,
        "low_availability_top_30": low_avail_top30,
        "research_titles_top_100": research_top100,
        "proxy_ndcg_at_10": round(proxy_ndcg10, 4),
        "ndcg_at_30": round(proxy_ndcg30, 4),
        "top10_fingerprint": top10_fingerprint,
        "sample_tier5_in_top_10": sample_t5_in_top10,
        "targets_met": {
            "tier5_top10_gte_6": len(set(top10) & tier5_all) >= 6,
            "tier5_in_top_30_target_gte_24": len(set(top30) & tier5_all) >= 24,
            "tier5_top100_gte_40": len(set(top_ids) & tier5_all) >= 40,
            "honeypots_top100_lte_10": honeypots_top100 <= 10,
            "low_avail_top30_lte_2": low_avail_top30 <= 2,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit submission quality.")
    parser.add_argument("submission", type=Path, nargs="?", default=PROJECT_ROOT / "team_sarva_automata.csv")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    artifacts_dir = (args.artifacts or settings.artifacts_dir).resolve()
    candidates_path = (args.candidates or settings.candidates_path).resolve()

    report = audit_submission(args.submission.resolve(), artifacts_dir=artifacts_dir, candidates_path=candidates_path)
    out_path = args.out or (artifacts_dir / "submission_audit.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["honeypot_dq_risk"]:
        print("\nWARNING: honeypot rate >10% in top 100 — Stage 3 DQ risk", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
