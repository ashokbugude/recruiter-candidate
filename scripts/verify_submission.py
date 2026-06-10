#!/usr/bin/env python3
"""End-to-end pre-upload check: rank (optional), validate, audit, fingerprint."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Key IDs expected in a good submission (update after re-tune if rankings shift).
EXPECTED_TOP10_FINGERPRINT: dict[str, int] = {
    "CAND_0061257": 1,
    "CAND_0093193": 8,
    "CAND_0046525": 10,
}


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def load_top10(csv_path: Path) -> list[str]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    ranked = sorted(rows, key=lambda r: int(r["rank"]))
    return [r["candidate_id"] for r in ranked[:10]]


def check_fingerprint(csv_path: Path, expected: dict[str, int]) -> list[str]:
    errors: list[str] = []
    rows = {r["candidate_id"]: int(r["rank"]) for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
    for cid, rank in expected.items():
        actual = rows.get(cid)
        if actual is None:
            errors.append(f"Expected {cid} in submission, not found")
        elif actual != rank:
            errors.append(f"Expected {cid} at rank {rank}, got {actual}")
    return errors


def check_rerank_diff(csv_path: Path, baseline_path: Path) -> list[str]:
    errors: list[str] = []
    if not baseline_path.exists():
        return [f"Baseline CSV not found: {baseline_path}"]
    new_top = set(load_top10(csv_path))
    old_top = set(load_top10(baseline_path))
    if new_top != old_top:
        only_new = new_top - old_top
        only_old = old_top - new_top
        errors.append(f"Top-10 set changed: added={sorted(only_new)} removed={sorted(only_old)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify submission before portal upload.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "team_sarva_automata.csv",
        help="Submission CSV path.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-run rank.py before validate (uses ~5–8 min CPU).",
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Fail if key candidate IDs are not at expected ranks.",
    )
    parser.add_argument(
        "--rerank-diff",
        type=Path,
        default=None,
        metavar="BASELINE_CSV",
        help="Compare top-10 set to a committed baseline CSV.",
    )
    args = parser.parse_args()
    py = sys.executable
    csv_path = args.csv.resolve()

    if args.rerank:
        code = run(
            [
                py,
                "rank.py",
                "--candidates",
                str(PROJECT_ROOT / "challenge" / "candidates.jsonl"),
                "--out",
                str(csv_path),
            ]
        )
        if code != 0:
            return code

    for script in ("challenge/validate_submission.py", "scripts/audit_submission.py"):
        if run([py, script, str(csv_path)]) != 0:
            return 1

    if args.fingerprint:
        fp_errors = check_fingerprint(csv_path, EXPECTED_TOP10_FINGERPRINT)
        if fp_errors:
            for err in fp_errors:
                print("FINGERPRINT:", err, file=sys.stderr)
            return 1
        print("Fingerprint check passed.")

    if args.rerank_diff:
        diff_errors = check_rerank_diff(csv_path, args.rerank_diff.resolve())
        if diff_errors:
            for err in diff_errors:
                print("RERANK DIFF:", err, file=sys.stderr)
            return 1
        print("Top-10 set matches baseline.")

    print("\nAll checks passed. Upload:", csv_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
