#!/usr/bin/env python3
"""Exploratory data analysis on the candidate dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.candidates import count_candidates, load_candidates  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.constants import KNOWN_TEMPLATE_HASHES, REFERENCE_DATE, career_description_hash  # noqa: E402
from app.labels.honeypots import detect_honeypot  # noqa: E402
from app.labels.tiers import assign_heuristic_tier  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def run_eda(candidates_path: Path, *, limit: int | None = None) -> dict:
    """Compute dataset statistics and trap/honeypot summaries."""
    total = count_candidates(candidates_path) if limit is None else min(
        count_candidates(candidates_path), limit
    )

    title_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    industry_counter: Counter[str] = Counter()
    yoe_buckets: Counter[str] = Counter()
    template_hash_counter: Counter[str] = Counter()
    honeypot_rule_counter: Counter[str] = Counter()
    tier_counter: Counter[int] = Counter()
    skill_count_buckets: Counter[str] = Counter()

    processed = 0
    honeypot_count = 0
    all_template_roles = 0

    for candidate in load_candidates(candidates_path, limit=limit):
        profile = candidate.get("profile") or {}
        title_counter[str(profile.get("current_title") or "unknown").lower()] += 1
        country_counter[str(profile.get("country") or "unknown")] += 1
        industry_counter[str(profile.get("current_industry") or "unknown").lower()] += 1

        yoe = float(profile.get("years_of_experience") or 0)
        if yoe < 3:
            yoe_buckets["0-3"] += 1
        elif yoe <= 5:
            yoe_buckets["3-5"] += 1
        elif yoe <= 9:
            yoe_buckets["5-9"] += 1
        elif yoe <= 15:
            yoe_buckets["9-15"] += 1
        else:
            yoe_buckets["15+"] += 1

        skills = candidate.get("skills") or []
        n_skills = len(skills)
        if n_skills <= 5:
            skill_count_buckets["0-5"] += 1
        elif n_skills <= 10:
            skill_count_buckets["6-10"] += 1
        elif n_skills <= 15:
            skill_count_buckets["11-15"] += 1
        else:
            skill_count_buckets["16+"] += 1

        career = candidate.get("career_history") or []
        role_template_hits = 0
        for role in career:
            desc = str(role.get("description") or "").strip()
            if len(desc) < 80:
                continue
            prefix = career_description_hash(desc)[:12]
            template_hash_counter[prefix] += 1
            if prefix in KNOWN_TEMPLATE_HASHES:
                role_template_hits += 1
        if career and role_template_hits == len(career):
            all_template_roles += 1

        hp = detect_honeypot(candidate, reference_date=REFERENCE_DATE)
        if hp.is_honeypot:
            honeypot_count += 1
        for rule in hp.rules_hit:
            honeypot_rule_counter[rule] += 1

        tier = assign_heuristic_tier(candidate, hp)
        tier_counter[tier.tier] += 1
        processed += 1

    known_template_role_total = sum(
        template_hash_counter[h[:12] if len(h) > 12 else h]
        for h in KNOWN_TEMPLATE_HASHES
        if h in template_hash_counter
    )

    report = {
        "candidates_path": str(candidates_path),
        "candidates_analyzed": processed,
        "candidates_total_in_file": total,
        "limit_applied": limit,
        "profile": {
            "top_titles": title_counter.most_common(25),
            "top_countries": country_counter.most_common(15),
            "top_industries": industry_counter.most_common(15),
            "yoe_distribution": dict(yoe_buckets),
            "skill_count_distribution": dict(skill_count_buckets),
        },
        "traps": {
            "known_template_hashes": sorted(KNOWN_TEMPLATE_HASHES),
            "top_recycled_description_hashes": template_hash_counter.most_common(12),
            "candidates_all_roles_template": all_template_roles,
            "known_template_role_occurrences": known_template_role_total,
        },
        "honeypots": {
            "detected_count": honeypot_count,
            "detection_rate": round(honeypot_count / max(1, processed), 4),
            "rule_hit_counts": dict(honeypot_rule_counter.most_common()),
        },
        "silver_tier_preview": {
            str(tier): count for tier, count in sorted(tier_counter.items())
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDA on candidate dataset.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Candidates file (default: settings.candidates_path).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON report (default: artifacts/eda_report.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N candidates (for quick runs).",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    candidates_path = (args.candidates or settings.candidates_path).resolve()
    out_path = (args.out or settings.artifact_path("eda_report.json")).resolve()

    logger.info("Running EDA on %s", candidates_path)
    report = run_eda(candidates_path, limit=args.limit)
    settings.ensure_artifacts_dir()
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "EDA complete: %d candidates, %d honeypots (%.1f%%)",
        report["candidates_analyzed"],
        report["honeypots"]["detected_count"],
        report["honeypots"]["detection_rate"] * 100,
    )
    logger.info("Report written to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
