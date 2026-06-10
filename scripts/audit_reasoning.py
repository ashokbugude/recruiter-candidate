#!/usr/bin/env python3
"""Verify reasoning text matches candidate profile fields (top 30 DQ check)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.rank_data import load_candidates_lookup  # noqa: E402


def _mentioned_skills(reasoning: str, skills: list[dict]) -> list[str]:
    """Return skill names from profile that appear in reasoning (case-insensitive)."""
    text = reasoning.lower()
    hits: list[str] = []
    for skill in skills[:15]:
        name = str(skill.get("name") or "").strip()
        if len(name) >= 3 and name.lower() in text:
            hits.append(name)
    return hits


def audit_reasoning_row(candidate: dict, reasoning: str) -> list[str]:
    issues: list[str] = []
    cid = candidate.get("candidate_id", "?")
    profile = candidate.get("profile") or {}
    location = str(profile.get("location") or "")
    yoe = profile.get("years_of_experience")
    skills = candidate.get("skills") or []

    loc_tokens = [t for t in re.split(r"[,/]", location) if t.strip()]
    if location and loc_tokens:
        if not any(token.strip().lower() in reasoning.lower() for token in loc_tokens):
            issues.append(f"{cid}: location '{location}' not reflected in reasoning")

    if yoe is not None:
        yoe_str = str(int(float(yoe))) if float(yoe) == int(float(yoe)) else str(yoe)
        if yoe_str not in reasoning and "year" not in reasoning.lower():
            issues.append(f"{cid}: YOE {yoe} not mentioned in reasoning")

    if skills and not _mentioned_skills(reasoning, skills):
        issues.append(f"{cid}: no profile skills echoed in reasoning")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reasoning vs profile consistency.")
    parser.add_argument("submission", type=Path, nargs="?", default=PROJECT_ROOT / "team_sarva_automata.csv")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    settings = get_settings()
    candidates_path = (args.candidates or settings.candidates_path).resolve()
    lookup = load_candidates_lookup(candidates_path)

    rows = sorted(csv.DictReader(args.submission.open(encoding="utf-8")), key=lambda r: int(r["rank"]))
    issues: list[str] = []
    for row in rows[: args.top]:
        cand = lookup.get(row["candidate_id"], {})
        issues.extend(audit_reasoning_row(cand, row.get("reasoning", "")))

    report = {"submission": str(args.submission), "rows_checked": min(args.top, len(rows)), "issues": issues}
    print(json.dumps(report, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
