"""Fact-anchored reasoning strings (no LLM at runtime)."""

from __future__ import annotations

from typing import Any

from app.constants import JD_HUB_DISPLAY
from app.jd_requirements import JDRequirements
from app.ranking_modifiers import (
    has_production_evidence,
    is_consulting_heavy,
    is_cv_speech_heavy_stack,
    is_excess_yoe,
    is_low_availability,
    is_outside_india,
    is_product_company_background,
    is_research_title,
    is_senior_ai_summary_clone,
    location_fit_score,
    recruiter_response_rate,
)


def _skill_phrase(candidate: dict[str, Any], max_skills: int = 4) -> str:
    skills = candidate.get("skills") or []
    names = [str(s.get("name") or "") for s in skills if s.get("name")]
    ir = [n for n in names if any(k in n.lower() for k in ("retrieval", "ranking", "search", "embedding", "faiss", "ltr"))]
    if ir:
        return ", ".join(ir[:max_skills])
    if names:
        return ", ".join(names[:max_skills])
    return "limited explicit stack listed"


def _eval_evidence(candidate: dict[str, Any]) -> str | None:
    text_parts: list[str] = []
    for role in candidate.get("career_history") or []:
        text_parts.append(str(role.get("description") or ""))
    blob = " ".join(text_parts).lower()
    for metric in ("ndcg", "map", "mrr", "offline eval", "online eval", "a/b test"):
        if metric in blob:
            return f"Mentions {metric.upper() if metric in ('ndcg', 'map', 'mrr') else metric} in career history."
    return None


def _location_phrase(candidate: dict[str, Any], jd: JDRequirements) -> str:
    profile = candidate.get("profile") or {}
    location = str(profile.get("location") or "India")
    fit = location_fit_score(candidate, jd)
    hubs = JD_HUB_DISPLAY
    if fit >= 0.95:
        return f"Based in {location} — strong fit for JD hubs ({hubs})."
    if fit >= 0.55:
        return f"India-based ({location}); relocation may be needed vs preferred {hubs}."
    if is_outside_india(candidate, jd):
        return f"Location {location} is outside India — JD targets Pune/Noida/Bangalore/Delhi corridor."
    return f"Location {location} is outside preferred India hubs ({hubs})."


def _availability_phrase(candidate: dict[str, Any]) -> str:
    signals = candidate.get("redrob_signals") or {}
    rr = recruiter_response_rate(candidate)
    notice = int(signals.get("notice_period_days") or 90)
    if is_low_availability(candidate):
        return f"Availability concern: response rate {rr:.0%}, notice {notice}d — JD prioritizes reachable hires."
    if rr >= 0.5 and notice <= 30:
        return f"Strong availability (response {rr:.0%}, {notice}d notice)."
    return f"Moderate availability (response {rr:.0%}, notice {notice}d)."


def _production_phrase(candidate: dict[str, Any]) -> str:
    if is_research_title(candidate) and not has_production_evidence(candidate):
        return "Research-leaning title without clear shipped production ML evidence — JD wants builders."
    if has_production_evidence(candidate):
        return "Career text shows shipped/production ML work."
    return "Production deployment evidence is thin in the profile."


def _jd_mismatch_phrase(candidate: dict[str, Any]) -> str | None:
    if is_cv_speech_heavy_stack(candidate):
        return "Stack skews CV/speech (e.g. YOLO/ASR) — role needs search/retrieval/IR depth, not vision pipelines."
    if is_excess_yoe(candidate) and is_research_title(candidate):
        return "High YOE + scientist title — acceptable only with strong production IR evidence."
    return None


def _company_phrase(candidate: dict[str, Any], rank: int) -> str:
    variant = (rank - 11) % 4
    if is_product_company_background(candidate):
        templates = (
            "Product/SaaS background aligns with in-house AI role.",
            "Built at product companies — fits startup shipping culture vs pure services.",
            "Current path is product engineering, matching JD's in-house builder focus.",
            "SaaS/platform tenure suggests end-to-end ownership beyond client delivery.",
        )
        return templates[variant]
    if is_consulting_heavy(candidate):
        templates = (
            "Consulting-heavy path — JD prefers product engineering depth.",
            "Mostly services/consulting employers — verify hands-on product IR work.",
            "Career skews IT services; ranked below product-company peers.",
            "Heavy consulting background — secondary fit vs product startup hire.",
        )
        return templates[variant]
    templates = (
        "Mixed services/product background.",
        "Blend of services and product roles — moderate startup fit.",
        "Background spans services and product teams.",
        "Career mix of consulting and product — acceptable but not primary profile.",
    )
    return templates[variant]


def _tail_opener(rank: int) -> str:
    openers = ("Depth option", "Pipeline backup", "Extended shortlist", "Lower-priority fit")
    return openers[(rank - 76) % len(openers)]


def build_reasoning(candidate: dict[str, Any], *, rank: int, jd: JDRequirements | None = None) -> str:
    """Generate varied, JD-aware reasoning from whitelisted fields."""
    profile = candidate.get("profile") or {}
    jd = jd or JDRequirements()
    title = str(profile.get("current_title") or "Professional")
    yoe = profile.get("years_of_experience")
    yoe_text = f"{float(yoe):.1f}" if yoe is not None else "?"
    skills = _skill_phrase(candidate)
    loc = _location_phrase(candidate, jd)
    avail = _availability_phrase(candidate)
    prod = _production_phrase(candidate)
    company = _company_phrase(candidate, rank)
    eval_line = _eval_evidence(candidate)
    mismatch = _jd_mismatch_phrase(candidate)

    if rank <= 10:
        parts = [
            f"Top pick: {title} ({yoe_text} YOE) with hands-on {skills}.",
            loc,
            avail,
            prod,
        ]
        if eval_line:
            parts.append(eval_line)
        if mismatch:
            parts.append(mismatch)
        if is_senior_ai_summary_clone(candidate):
            parts.append("Similar summary template to peers — ranked up on availability + product signals.")
        return " ".join(parts)[:500]

    if rank <= 30:
        parts = [
            f"Strong match: {title}, {yoe_text} YOE; core stack {skills}.",
            company,
            avail,
            prod,
            loc,
        ]
        if mismatch:
            parts.append(mismatch)
        return " ".join(parts)[:500]

    if rank <= 50:
        parts = [
            f"Solid bench: {title} ({yoe_text} YOE) — {skills}.",
            avail,
            _company_phrase(candidate, rank),
        ]
        if mismatch:
            parts.append(mismatch)
        if rank > 20 and is_low_availability(candidate):
            parts.append("Deprioritized vs higher-availability peers for this startup role.")
        return " ".join(parts)[:500]

    if rank <= 75:
        parts = [
            f"Reserve: {title} with {yoe_text} YOE; relevant skills include {skills}.",
            prod,
            avail,
        ]
        if mismatch:
            parts.append(mismatch)
        return " ".join(parts)[:500]

    parts = [
        f"{_tail_opener(rank)} #{rank}: {title} ({yoe_text} YOE).",
        _company_phrase(candidate, rank),
        avail,
    ]
    if is_research_title(candidate):
        parts.append("Research title — included only with some IR overlap; not a primary hire profile.")
    if mismatch:
        parts.append(mismatch)
    return " ".join(parts)[:500]
