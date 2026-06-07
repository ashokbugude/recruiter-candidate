"""Trap and honeypot penalties for ranking."""

from __future__ import annotations

from typing import Any


def trap_penalty(row: dict[str, Any]) -> float:
    """Subtractive penalty from precomputed feature / metadata row."""
    if row.get("should_exclude"):
        return 10.0
    penalty = 0.0
    honeypot = float(row.get("honeypot_score") or 0.0)
    if row.get("is_honeypot") or honeypot >= 0.6:
        penalty += 5.0
    if row.get("is_trap"):
        penalty += 2.5
    if float(row.get("trap_title_flag") or 0.0) >= 1.0:
        penalty += 1.5
    if float(row.get("all_templates_flag") or 0.0) >= 1.0:
        penalty += 2.0
    if float(row.get("stuffing_flag") or 0.0) >= 1.0:
        penalty += 1.0
    return penalty


def should_hard_exclude(row: dict[str, Any]) -> bool:
    """Exclude candidate from top-100 entirely."""
    if row.get("should_exclude"):
        return True
    if row.get("is_honeypot"):
        return True
    return float(row.get("honeypot_score") or 0.0) >= 0.6
