#!/usr/bin/env python3
"""Refresh live_* fields in fusion/modifier JSON from an existing submission CSV (no re-rank)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifact_names import FUSION_PARAMS, MODIFIER_PARAMS  # noqa: E402
from app.config import get_settings  # noqa: E402
from scripts.audit_submission import audit_submission  # noqa: E402


def patch_live_metrics(artifacts_dir: Path, report: dict) -> None:
    live = {
        "live_ndcg_at_10": report["proxy_ndcg_at_10"],
        "live_tier5_in_top_10": float(report["tier5_in_top_10"]),
        "live_tier5_in_top_30": float(report["tier5_in_top_30"]),
        "live_tier5_in_top_100": float(report["tier5_in_top_100"]),
        "live_template_clones_top_30": float(report["template_clones_top_30"]),
        "live_ndcg_at_30": report["ndcg_at_30"],
    }

    fusion_path = artifacts_dir / FUSION_PARAMS
    if fusion_path.exists():
        data = json.loads(fusion_path.read_text(encoding="utf-8"))
        data.update(live)
        fusion_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Updated {fusion_path}")

    mod_path = artifacts_dir / MODIFIER_PARAMS
    if mod_path.exists():
        data = json.loads(mod_path.read_text(encoding="utf-8"))
        data.update(live)
        mod_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Updated {mod_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync live_* artifact metrics from submission audit.")
    parser.add_argument("submission", type=Path, nargs="?", default=PROJECT_ROOT / "team_sarva_automata.csv")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    artifacts_dir = (args.artifacts or settings.artifacts_dir).resolve()
    candidates_path = (args.candidates or settings.candidates_path).resolve()

    report = audit_submission(args.submission.resolve(), artifacts_dir=artifacts_dir, candidates_path=candidates_path)
    patch_live_metrics(artifacts_dir, report)
    print(
        f"live_ndcg_at_10={report['proxy_ndcg_at_10']} "
        f"tier5@30={report['tier5_in_top_30']} tier5@100={report['tier5_in_top_100']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
