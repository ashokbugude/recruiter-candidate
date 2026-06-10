"""Behavioral signal multiplier (multiplicative ranking modifier)."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.constants import REFERENCE_DATE


def behavioral_multiplier(
    candidate: dict[str, Any],
    *,
    reference_date: date | None = None,
) -> float:
    """Compute multiplicative modifier in [0.55, 1.15] from Redrob signals."""
    ref = reference_date or REFERENCE_DATE
    signals = candidate.get("redrob_signals") or {}

    multiplier = 1.0
    multiplier *= 0.70 + 0.30 * float(signals.get("open_to_work_flag", False))
    rr = float(signals.get("recruiter_response_rate") or 0)
    if rr < 0.15:
        multiplier *= 0.40
    elif rr < 0.25:
        multiplier *= 0.60
    else:
        multiplier *= 0.55 + 0.45 * rr
    multiplier *= _recency_decay(signals.get("last_active_date"), ref)
    saved = min(1.0, int(signals.get("saved_by_recruiters_30d") or 0) / 8)
    multiplier *= 0.80 + 0.20 * saved

    gh_raw = signals.get("github_activity_score")
    if gh_raw == -1:
        github_norm = 0.0
    else:
        github_norm = min(1.0, max(0.0, float(gh_raw or 0) / 60))
    multiplier *= 0.75 + 0.25 * github_norm

    notice = int(signals.get("notice_period_days") or 90)
    if notice <= 30:
        multiplier *= 0.90
    elif notice <= 60:
        multiplier *= 0.80
    else:
        multiplier *= 0.65

    return max(0.35, min(1.15, multiplier))


def _recency_decay(value: Any, ref: date) -> float:
    if not value:
        return 0.7
    try:
        from datetime import datetime

        last = datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        days = (ref - last).days
    except ValueError:
        return 0.7
    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.85
    if days <= 180:
        return 0.7
    return 0.55
