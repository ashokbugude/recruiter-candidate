"""Fact-anchored reasoning strings (no LLM at runtime)."""

from __future__ import annotations

from typing import Any


def build_reasoning(candidate: dict[str, Any], *, rank: int) -> str:
    """Generate submission reasoning from whitelisted profile fields."""
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    title = str(profile.get("current_title") or "Professional")
    yoe = profile.get("years_of_experience")
    yoe_text = f"{float(yoe):.1f}" if yoe is not None else "?"

    skills = candidate.get("skills") or []
    matched = [
        str(s.get("name"))
        for s in skills
        if any(k in str(s.get("name") or "").lower() for k in ("retrieval", "ranking", "embedding", "search", "ml", "ai"))
    ][:4]
    skill_text = ", ".join(matched) if matched else "limited explicit AI/search skills listed"

    response = float(signals.get("recruiter_response_rate") or 0)
    parts = [f"{title} with {yoe_text} yrs; {skill_text}; response rate {response:.2f}."]

    career = candidate.get("career_history") or []
    for role in career[:2]:
        desc = str(role.get("description") or "").lower()
        if any(k in desc for k in ("retrieval", "ranking", "search", "recommendation")):
            company = role.get("company") or "prior role"
            parts.append(f"Career evidence: {company} work mentions search/ranking systems.")
            break

    notice = int(signals.get("notice_period_days") or 90)
    if rank > 20 and notice > 60:
        parts.append(f"Notice period {notice}d noted.")
    if signals.get("github_activity_score") == -1 and matched:
        parts.append("GitHub activity unavailable despite technical claims.")

    return " ".join(parts)[:500]
