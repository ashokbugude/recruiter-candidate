"""Post-fusion ranking adjustments — availability, template clones, research titles."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from app.constants import (
    CONSULTING_FIRMS,
    CV_SPEECH_SKILL_MARKERS,
    JD_PREFERRED_CITIES,
    JD_PRIMARY_HUBS,
    JD_SECONDARY_HUBS,
    JD_YOE_SOFT_MAX,
    PRODUCT_COMPANY_INDICATORS,
    REFERENCE_DATE,
    SEARCH_RETRIEVAL_KEYWORDS,
)
from app.modifier_params import DEFAULT_MODIFIER_PARAMS, ModifierParams
from app.features import candidate_profile_text
from app.jd_requirements import JDRequirements

# Behavioral-twin family called out in challenge README / critique.
SENIOR_AI_CLONE_OPENING = "senior ai engineer with"
SENIOR_AI_CLONE_MARKERS: tuple[str, ...] = (
    "production ml",
    "search",
    "retrieval",
    "ranking",
)

LOW_RESPONSE_THRESHOLD = 0.15
LOW_RESPONSE_STEEP_THRESHOLD = 0.20
STALE_ACTIVE_DAYS = 90
TOP_AVAILABILITY_CAP = 30

PRODUCTION_EVIDENCE_KEYWORDS: tuple[str, ...] = (
    "shipped",
    "production",
    "deployed",
    "serving",
    "sla",
    "online",
    "in production",
    "productionized",
)


def days_since_active(signals: dict[str, Any], ref: date) -> int | None:
    value = signals.get("last_active_date")
    if not value:
        return None
    try:
        last = datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        return (ref - last).days
    except ValueError:
        return None


def recruiter_response_rate(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals") or {}
    return float(signals.get("recruiter_response_rate") or 0.0)


def is_low_availability(
    candidate: dict[str, Any],
    *,
    reference_date: date | None = None,
) -> bool:
    """JD: low response or stale profile = not actually available."""
    ref = reference_date or REFERENCE_DATE
    signals = candidate.get("redrob_signals") or {}
    rr = recruiter_response_rate(candidate)
    days = days_since_active(signals, ref)
    if rr < LOW_RESPONSE_THRESHOLD:
        return True
    if days is not None and days > STALE_ACTIVE_DAYS:
        return True
    return False


def is_senior_ai_summary_clone(candidate: dict[str, Any]) -> bool:
    profile = candidate.get("profile") or {}
    summary = str(profile.get("summary") or "").lower().strip()
    if not summary.startswith(SENIOR_AI_CLONE_OPENING):
        return False
    hits = sum(1 for marker in SENIOR_AI_CLONE_MARKERS if marker in summary)
    return hits >= 3


def is_research_title(candidate: dict[str, Any]) -> bool:
    profile = candidate.get("profile") or {}
    title = str(profile.get("current_title") or "").lower()
    headline = str(profile.get("headline") or "").lower()
    combined = f"{title} {headline}"
    return "research" in combined or (
        "scientist" in combined and "applied" not in combined and "research engineer" not in combined
    )


def has_production_evidence(candidate: dict[str, Any]) -> bool:
    text = candidate_profile_text(candidate).lower()
    return any(keyword in text for keyword in PRODUCTION_EVIDENCE_KEYWORDS)


def ir_keyword_hits(candidate: dict[str, Any], *, min_hits: int = 2) -> int:
    text = candidate_profile_text(candidate).lower()
    return sum(1 for keyword in SEARCH_RETRIEVAL_KEYWORDS if keyword in text)


def _skills_blob(candidate: dict[str, Any]) -> str:
    skills = candidate.get("skills") or []
    return " ".join(str(s.get("name") or "").lower() for s in skills)


def is_cv_speech_heavy_stack(candidate: dict[str, Any]) -> bool:
    """JD anti-pattern: CV/speech-heavy stack without IR/search depth."""
    skills = _skills_blob(candidate)
    cv_in_skills = any(marker in skills for marker in CV_SPEECH_SKILL_MARKERS)
    ir_in_skills = sum(1 for keyword in SEARCH_RETRIEVAL_KEYWORDS if keyword in skills)
    if cv_in_skills and ir_in_skills < 2:
        return True
    text = candidate_profile_text(candidate).lower()
    cv_speech = any(marker in text for marker in CV_SPEECH_SKILL_MARKERS)
    return cv_speech and ir_keyword_hits(candidate) < 2


def is_outside_india(candidate: dict[str, Any], jd: JDRequirements) -> bool:
    return location_fit_score(candidate, jd) == 0.0


def is_excess_yoe(candidate: dict[str, Any]) -> bool:
    profile = candidate.get("profile") or {}
    yoe = profile.get("years_of_experience")
    return yoe is not None and float(yoe) > JD_YOE_SOFT_MAX


def is_stretched_scientist(candidate: dict[str, Any]) -> bool:
    """JD ideal 5–9 YOE; scientist titles with 13+ YOE need stronger evidence."""
    profile = candidate.get("profile") or {}
    yoe = profile.get("years_of_experience")
    if yoe is None or float(yoe) <= 13.0:
        return False
    title = str(profile.get("current_title") or "").lower()
    return "scientist" in title


def is_primary_hub(candidate: dict[str, Any]) -> bool:
    location = str((candidate.get("profile") or {}).get("location") or "").lower()
    return any(hub in location for hub in JD_PRIMARY_HUBS)


def is_secondary_hub(candidate: dict[str, Any]) -> bool:
    location = str((candidate.get("profile") or {}).get("location") or "").lower()
    if is_primary_hub(candidate):
        return False
    return any(hub in location for hub in JD_SECONDARY_HUBS)


def is_consulting_heavy(candidate: dict[str, Any]) -> bool:
    career = candidate.get("career_history") or []
    if not career:
        return False
    consulting = 0
    for role in career[:5]:
        company = str(role.get("company") or "").lower()
        if any(firm in company for firm in CONSULTING_FIRMS):
            consulting += 1
    return consulting >= max(2, len(career[:5]) - 1)


def is_product_company_background(candidate: dict[str, Any]) -> bool:
    profile = candidate.get("profile") or {}
    company = str(profile.get("current_company") or "").lower()
    summary = str(profile.get("summary") or "").lower()
    blob = f"{company} {summary}"
    career_text = " ".join(
        str(r.get("company") or "") + " " + str(r.get("description") or "")
        for r in (candidate.get("career_history") or [])[:3]
    ).lower()
    blob = f"{blob} {career_text}"
    return any(indicator in blob for indicator in PRODUCT_COMPANY_INDICATORS)


def location_fit_score(candidate: dict[str, Any], jd: JDRequirements) -> float:
    profile = candidate.get("profile") or {}
    location = str(profile.get("location") or "").lower()
    country = str(profile.get("country") or "").lower()
    if jd.preferred_country.lower() not in country and country not in ("india", "in"):
        return 0.0
    if any(city in location for city in JD_PREFERRED_CITIES):
        return 1.0
    if "india" in country or country == "in":
        return 0.6
    return 0.2


def availability_score(candidate: dict[str, Any], *, reference_date: date | None = None) -> float:
    """Higher = more recruitable (used to break template-twin ties)."""
    ref = reference_date or REFERENCE_DATE
    signals = candidate.get("redrob_signals") or {}
    rr = recruiter_response_rate(candidate)
    days = days_since_active(signals, ref)
    recency = 1.0 if days is None else max(0.0, 1.0 - days / 365.0)
    notice = int(signals.get("notice_period_days") or 90)
    notice_score = 1.0 if notice <= 30 else 0.8 if notice <= 60 else 0.5
    otw = 1.0 if signals.get("open_to_work_flag") else 0.85
    return rr * 0.55 + recency * 0.25 + notice_score * 0.12 + otw * 0.08


def final_score_multiplier(
    candidate: dict[str, Any],
    jd: JDRequirements,
    *,
    reference_date: date | None = None,
    silver_tier: int | None = None,
    params: ModifierParams | None = None,
    career_scores: dict[str, float] | None = None,
) -> float:
    """Multiplicative adjustment on fused reranker score."""
    p = params or DEFAULT_MODIFIER_PARAMS
    ref = reference_date or REFERENCE_DATE
    mult = 1.0
    rr = recruiter_response_rate(candidate)
    tier = int(silver_tier) if silver_tier is not None else 2
    cid = str(candidate.get("candidate_id") or "")

    if is_low_availability(candidate, reference_date=ref):
        mult *= p.low_avail_mult
    elif rr < LOW_RESPONSE_STEEP_THRESHOLD:
        mult *= p.low_rr_steep_mult
    elif rr < 0.35:
        mult *= p.low_rr_moderate_mult

    if is_senior_ai_summary_clone(candidate):
        if rr < LOW_RESPONSE_STEEP_THRESHOLD:
            mult *= p.clone_low_rr_mult
        elif tier == 5 and rr >= 0.35:
            mult *= p.clone_tier5_mult
        else:
            mult *= 0.85 + 0.15 * min(1.0, availability_score(candidate, reference_date=ref))

    if is_research_title(candidate):
        if not has_production_evidence(candidate):
            mult *= p.research_no_prod_mult
        else:
            mult *= p.research_with_prod_mult

    if is_cv_speech_heavy_stack(candidate):
        mult *= p.cv_speech_mult

    if is_outside_india(candidate, jd):
        mult *= p.outside_india_mult

    if is_stretched_scientist(candidate):
        if tier == 5 and has_production_evidence(candidate) and ir_keyword_hits(candidate) >= 3:
            mult *= max(p.stretched_scientist_mult, 0.88)
        else:
            mult *= p.stretched_scientist_mult
    elif is_excess_yoe(candidate) and ir_keyword_hits(candidate) < 3:
        mult *= p.excess_yoe_mult

    if is_consulting_heavy(candidate) and not is_product_company_background(candidate):
        mult *= p.consulting_mult

    mult *= 0.90 + 0.10 * location_fit_score(candidate, jd)

    if is_primary_hub(candidate):
        mult *= p.primary_hub_boost
    elif is_secondary_hub(candidate):
        mult *= p.preferred_hub_boost

    if is_product_company_background(candidate):
        mult *= p.product_boost

    if career_scores and cid:
        score = career_scores.get(cid, 0.0)
        if (
            score >= p.plain_language_career_threshold
            and not is_senior_ai_summary_clone(candidate)
            and tier >= 4
        ):
            mult *= p.plain_language_career_boost

    if tier == 5 and not is_low_availability(candidate, reference_date=ref) and not is_cv_speech_heavy_stack(candidate):
        mult *= p.tier5_boost

    return max(0.05, min(1.25, mult))


def apply_top_availability_cap(
    ranked_ids: list[str],
    candidate_lookup: dict[str, dict[str, Any]],
    *,
    cap: int = TOP_AVAILABILITY_CAP,
    top_k: int = 100,
    reference_date: date | None = None,
) -> list[str]:
    """Keep ranks 1..cap free of low-availability profiles when alternatives exist."""
    ref = reference_date or REFERENCE_DATE
    available: list[str] = []
    deferred: list[str] = []
    for cid in ranked_ids:
        cand = candidate_lookup.get(cid, {})
        if is_low_availability(cand, reference_date=ref):
            deferred.append(cid)
        else:
            available.append(cid)

    if not deferred:
        return ranked_ids[:top_k]

    head = available[:cap]
    remainder = available[cap:] + deferred
    reordered = head + remainder
    return reordered[:top_k]


def apply_clone_cap(
    ranked_ids: list[str],
    candidate_lookup: dict[str, dict[str, Any]],
    *,
    final: dict[str, float],
    max_clones: int = 10,
    cap_zone: int = 30,
    top_k: int = 100,
) -> list[str]:
    """Limit template-clone summaries in ranks 1..cap_zone."""
    if max_clones <= 0:
        return ranked_ids[:top_k]

    def is_clone(cid: str) -> bool:
        return is_senior_ai_summary_clone(candidate_lookup.get(cid, {}))

    head = ranked_ids[:cap_zone]
    head_clones = [cid for cid in head if is_clone(cid)]
    if len(head_clones) <= max_clones:
        return ranked_ids[:top_k]

    kept_clones = set(
        sorted(head_clones, key=lambda cid: (-final.get(cid, 0.0), cid))[:max_clones]
    )
    zone: list[str] = []
    used: set[str] = set()

    for cid in ranked_ids:
        if len(zone) >= cap_zone:
            break
        if is_clone(cid) and cid not in kept_clones:
            continue
        zone.append(cid)
        used.add(cid)

    for cid in ranked_ids:
        if len(zone) >= cap_zone:
            break
        if cid in used or is_clone(cid):
            continue
        zone.append(cid)
        used.add(cid)

    tail = [cid for cid in ranked_ids if cid not in used]
    return (zone + tail)[:top_k]


def fusion_to_submission_scores(ranked_ids: list[str], fused: dict[str, float]) -> dict[str, float]:
    """Rank-monotonic scores with model-informed spread (safe after cap/reorder)."""
    if not ranked_ids:
        return {}
    n = len(ranked_ids)
    values = [float(fused.get(cid, 0.0)) for cid in ranked_ids]
    lo, hi = min(values), max(values)
    band = 0.88 / max(n - 1, 1)

    scores: dict[str, float] = {}
    prev = 1.0
    for index, cid in enumerate(ranked_ids):
        rank_base = 0.99 - index * band
        if hi > lo + 1e-9:
            nudge = ((values[index] - lo) / (hi - lo)) * band * 0.35
        else:
            nudge = values[index] * 1e-6
        score = rank_base + nudge
        if index > 0:
            score = min(score, prev - 1e-4)
        score = max(0.01, score)
        scores[cid] = round(score, 4)
        prev = score
    return scores


def summary_clone_fingerprint(summary: str) -> str:
    """Normalize clone summaries for grouping."""
    lowered = summary.lower().strip()
    if not lowered.startswith(SENIOR_AI_CLONE_OPENING):
        return ""
    return re.sub(r"\d+(\.\d+)?", "N", lowered[:120])
