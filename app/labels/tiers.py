"""Heuristic relevance tier assignment (proxy silver labels)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.constants import (
    CONSULTING_FIRMS,
    JD_PREFERRED_CITIES,
    JD_PREFERRED_COUNTRY,
    JD_YOE_MAX,
    JD_YOE_MIN,
    JD_YOE_SOFT_MAX,
    JD_YOE_SOFT_MIN,
    ML_PRODUCTION_KEYWORDS,
    PRODUCT_COMPANY_INDICATORS,
    SEARCH_RETRIEVAL_KEYWORDS,
    TRAP_TITLE_KEYWORDS,
    count_search_retrieval_hits,
    is_junior_title,
    is_known_template,
    matches_senior_ai_title,
)
from app.labels.honeypots import HoneypotResult


@dataclass(frozen=True)
class TierResult:
    candidate_id: str
    tier: int
    label_source: str
    confidence: float
    reasons: tuple[str, ...]


def assign_heuristic_tier(
    candidate: dict[str, Any],
    honeypot: HoneypotResult,
) -> TierResult:
    """Assign relevance tier 0–5 from profile heuristics aligned to JD intent."""
    cid = str(candidate.get("candidate_id") or "")
    profile = candidate.get("profile") or {}
    skills = candidate.get("skills") or []
    career = candidate.get("career_history") or []
    signals = candidate.get("redrob_signals") or {}

    if honeypot.is_honeypot:
        return TierResult(
            candidate_id=cid,
            tier=0,
            label_source="honeypot_subtle",
            confidence=0.95,
            reasons=honeypot.rules_hit,
        )

    title = str(profile.get("current_title") or "").lower()
    headline = str(profile.get("headline") or "").lower()
    summary = str(profile.get("summary") or "").lower()
    location = str(profile.get("location") or "").lower()
    country = str(profile.get("country") or "")
    yoe = float(profile.get("years_of_experience") or 0)
    industry = str(profile.get("current_industry") or "").lower()
    company = str(profile.get("current_company") or "").lower()

    profile_text = " ".join([title, headline, summary, _career_text(career)])
    reasons: list[str] = []

    has_r4 = "R4" in honeypot.rules_hit
    has_r7 = "R7" in honeypot.rules_hit
    junior = is_junior_title(title, headline)
    max_tier = 3 if has_r4 else 5

    # Tier 1: trap titles with stuffed AI skills
    trap_title = any(trap in title for trap in TRAP_TITLE_KEYWORDS)
    ai_skill_count = _count_relevant_skills(skills)
    template_roles = sum(
        1 for role in career if is_known_template(str(role.get("description") or ""))
    )

    if trap_title and ai_skill_count >= 6:
        reasons.append("trap_title_ai_stuffing")
        return TierResult(cid, 1, "heuristic_trap", 0.85, tuple(reasons))

    if has_r7 or (
        template_roles >= len(career) and len(career) > 0 and ai_skill_count >= 4
    ):
        reasons.append("recycled_templates")
        return TierResult(cid, 1, "heuristic_template", 0.80, tuple(reasons))

    if honeypot.is_trap and "R11" in honeypot.rules_hit:
        reasons.append("title_skill_mismatch")
        return TierResult(cid, 1, "heuristic_trap", 0.82, tuple(reasons))

    # Score components for tier 2–5
    title_score = _title_fit_score(title, headline)
    search_score = _search_retrieval_score(profile_text)
    ml_score = _keyword_hits(profile_text, ML_PRODUCTION_KEYWORDS)
    product_score = _product_company_score(career, company, industry)
    location_score = _location_score(country, location)
    yoe_score = _yoe_score(yoe)
    behavioral_score = _behavioral_score(signals)
    skill_trust = _skill_trust_score(skills, signals)

    senior_title = matches_senior_ai_title(title, headline)
    search_engineer = any(k in title for k in ("search", "recommendation", "ranking", "retrieval"))

    composite = (
        0.22 * title_score
        + 0.20 * search_score
        + 0.15 * ml_score
        + 0.15 * product_score
        + 0.10 * location_score
        + 0.08 * yoe_score
        + 0.05 * behavioral_score
        + 0.05 * skill_trust
    )

    consulting_only = _consulting_only(career)
    if consulting_only:
        composite -= 0.15
        reasons.append("consulting_only")

    def _cap(tier: int) -> int:
        return min(tier, max_tier)

    # Tier 5: ideal senior AI / search engineer at product co, India, active
    if (
        max_tier >= 5
        and senior_title
        and not junior
        and yoe >= JD_YOE_MIN
        and search_score >= 0.35
        and product_score >= 0.5
        and location_score >= 0.7
        and yoe_score >= 0.6
        and behavioral_score >= 0.45
        and not trap_title
        and not has_r4
    ):
        reasons.extend(["senior_ai", "search_retrieval", "product_company"])
        if search_engineer:
            reasons.append("search_engineer_title")
        return TierResult(
            cid, 5, "heuristic_senior_ai", min(0.98, composite + 0.2), tuple(reasons)
        )

    # Tier 4: strong ML / search fit
    if (
        max_tier >= 4
        and (senior_title or search_engineer)
        and not junior
        and yoe >= JD_YOE_SOFT_MIN
        and (search_score >= 0.25 or ml_score >= 0.35)
        and product_score >= 0.35
        and yoe_score >= 0.5
        and not trap_title
        and not has_r4
    ):
        reasons.append("strong_ml_search")
        return TierResult(cid, _cap(4), "heuristic_ml_search", min(0.90, composite + 0.1), tuple(reasons))

    # Tier 3: relevant ML engineer / adjacent (R4-capped profiles may land here)
    if (
        max_tier >= 3
        and ("engineer" in title or "scientist" in title)
        and not junior
        and (ml_score >= 0.2 or search_score >= 0.15)
        and yoe >= 3
        and not trap_title
    ):
        reasons.append("relevant_ml")
        if has_r4:
            reasons.append("salary_inversion_cap")
        return TierResult(cid, _cap(3), "heuristic_relevant", min(0.80, composite), tuple(reasons))

    # Tier 2: software/data transitioning to ML
    if (
        any(k in title for k in ("software", "backend", "data", "full stack", "cloud", "devops"))
        and (ml_score >= 0.1 or ai_skill_count >= 3)
        and not trap_title
    ):
        reasons.append("ml_transition")
        return TierResult(cid, _cap(2), "heuristic_transition", min(0.70, composite), tuple(reasons))

    # Tier 1: weak / keyword-only
    if trap_title or ai_skill_count >= 8 or junior:
        if junior:
            reasons.append("junior_title")
        reasons.append("weak_or_stuffed")
        return TierResult(cid, 1, "heuristic_weak", 0.60, tuple(reasons))

    return TierResult(cid, _cap(2), "heuristic_default", 0.40, ("default_adjacent",))


def _career_text(career: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for role in career:
        parts.append(str(role.get("title") or ""))
        parts.append(str(role.get("description") or ""))
    return " ".join(parts).lower()


def _search_retrieval_score(text: str) -> float:
    hits = count_search_retrieval_hits(text)
    return min(1.0, hits / max(3, len(SEARCH_RETRIEVAL_KEYWORDS) // 4))


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> float:
    hits = sum(1 for kw in keywords if kw in text)
    return min(1.0, hits / max(3, len(keywords) // 4))


def _title_fit_score(title: str, headline: str) -> float:
    if is_junior_title(title, headline):
        return 0.15
    if matches_senior_ai_title(title, headline):
        return 1.0
    combined = f"{title} {headline}"
    if any(k in combined for k in ("machine learning", "data scientist", "ml ", " ai ")):
        return 0.7
    if "engineer" in combined:
        return 0.45
    return 0.2


def _product_company_score(
    career: list[dict[str, Any]], company: str, industry: str
) -> float:
    score = 0.0
    consulting_roles = 0
    for role in career:
        comp = str(role.get("company") or "").lower()
        desc = str(role.get("description") or "").lower()
        size = str(role.get("company_size") or "")
        if any(firm in comp for firm in CONSULTING_FIRMS):
            consulting_roles += 1
        elif any(
            ind in desc or ind in str(role.get("industry") or "").lower()
            for ind in PRODUCT_COMPANY_INDICATORS
        ):
            score += 0.35
        elif size in {"1-10", "11-50", "51-200"}:
            score += 0.25
    if any(ind in company or ind in industry for ind in PRODUCT_COMPANY_INDICATORS):
        score += 0.3
    if consulting_roles == len(career) and len(career) > 0:
        return 0.1
    return min(1.0, score)


def _consulting_only(career: list[dict[str, Any]]) -> bool:
    if not career:
        return False
    for role in career:
        comp = str(role.get("company") or "").lower()
        if not any(firm in comp for firm in CONSULTING_FIRMS):
            return False
    return True


def _location_score(country: str, location: str) -> float:
    if country.lower() != JD_PREFERRED_COUNTRY.lower():
        return 0.3
    if any(city in location for city in JD_PREFERRED_CITIES):
        return 1.0
    return 0.75


def _yoe_score(yoe: float) -> float:
    if JD_YOE_MIN <= yoe <= JD_YOE_MAX:
        return 1.0
    if JD_YOE_SOFT_MIN <= yoe <= JD_YOE_SOFT_MAX:
        return 0.7
    if yoe < JD_YOE_SOFT_MIN:
        return max(0.2, yoe / JD_YOE_SOFT_MIN * 0.5)
    return max(0.3, 1.0 - (yoe - JD_YOE_SOFT_MAX) / 10)


def _behavioral_score(signals: dict[str, Any]) -> float:
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    response = float(signals.get("recruiter_response_rate") or 0)
    saved = min(1.0, int(signals.get("saved_by_recruiters_30d") or 0) / 8)
    github = signals.get("github_activity_score")
    gh = 0.5 if github == -1 else min(1.0, max(0.0, float(github or 0) / 60))
    return 0.35 * open_to_work + 0.30 * response + 0.20 * saved + 0.15 * gh


def _skill_trust_score(skills: list[dict[str, Any]], signals: dict[str, Any]) -> float:
    if not skills:
        return 0.0
    assessments = signals.get("skill_assessment_scores") or {}
    prof_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    trusts: list[float] = []
    for skill in skills:
        prof = prof_map.get(str(skill.get("proficiency") or ""), 0.3)
        duration = min(1.0, int(skill.get("duration_months") or 0) / 24)
        endorse = min(1.0, math.log1p(int(skill.get("endorsements") or 0)) / 3)
        name = str(skill.get("name") or "")
        assess = float(assessments.get(name, 0) or 0) / 100 if name in assessments else 1.0
        trusts.append(prof * duration * endorse * assess)
    return sum(trusts) / len(trusts)


def _count_relevant_skills(skills: list[dict[str, Any]]) -> int:
    count = 0
    for skill in skills:
        name = str(skill.get("name") or "").lower()
        if any(k in name for k in ("ml", "ai", "nlp", "pytorch", "tensorflow", "llm", "rag", "embedding")):
            count += 1
    return count
