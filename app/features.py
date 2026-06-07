"""45-feature extractor for LightGBM learning-to-rank."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from rapidfuzz import fuzz

from app.constants import (
    CONSULTING_FIRMS,
    JD_PREFERRED_CITIES,
    PRODUCT_COMPANY_INDICATORS,
    REFERENCE_DATE,
    TRAP_TITLE_KEYWORDS,
    count_search_retrieval_hits,
    is_junior_title,
    is_known_template,
    matches_senior_ai_title,
)
from app.jd_requirements import JDRequirements, skill_matches_jd, title_matches_jd_targets
from app.labels.honeypots import detect_honeypot

FEATURE_NAMES: tuple[str, ...] = (
    # Title fit (5)
    "title_fit_score",
    "matches_senior_ai_title",
    "is_junior_title",
    "search_engineer_title",
    "trap_title_flag",
    # Coherence (5)
    "template_match_ratio",
    "all_templates_flag",
    "summary_title_mismatch",
    "career_desc_unique_ratio",
    "honeypot_score",
    # Skills (8)
    "jd_skill_match_count",
    "jd_skill_match_ratio",
    "skill_trust_weighted",
    "avg_skill_duration",
    "max_endorsement",
    "assessment_avg",
    "stuffing_flag",
    "ai_skill_count",
    # Career (6)
    "product_company_ratio",
    "consulting_only_flag",
    "career_progression",
    "search_keywords_in_history",
    "career_months_total",
    "current_role_months",
    # Experience (4)
    "yoe",
    "yoe_in_band",
    "band_distance",
    "yoe_score",
    # Education (4)
    "education_tier_score",
    "cs_ai_field",
    "has_postgrad",
    "education_count",
    # Location & profile (6)
    "india_flag",
    "preferred_city_match",
    "relocate_flag",
    "remote_ok",
    "profile_completeness",
    "salary_mid_lpa",
    # Behavioral (7)
    "response_rate",
    "recency_score",
    "open_to_work",
    "saved_by_recruiters_norm",
    "github_score_norm",
    "notice_period_score",
    "interview_rate",
)

assert len(FEATURE_NAMES) == 45


def candidate_profile_text(candidate: dict[str, Any]) -> str:
    """Concatenate searchable profile text for BM25 / display."""
    profile = candidate.get("profile") or {}
    parts = [
        str(profile.get("headline") or ""),
        str(profile.get("summary") or ""),
        str(profile.get("current_title") or ""),
        str(profile.get("current_company") or ""),
    ]
    for skill in candidate.get("skills") or []:
        parts.append(str(skill.get("name") or ""))
    for role in candidate.get("career_history") or []:
        parts.append(str(role.get("title") or ""))
        parts.append(str(role.get("description") or ""))
    return " ".join(p for p in parts if p)


def extract_features(
    candidate: dict[str, Any],
    jd: JDRequirements,
    *,
    reference_date: date | None = None,
) -> dict[str, float]:
    """Extract 45 numeric features for one candidate."""
    ref = reference_date or REFERENCE_DATE
    profile = candidate.get("profile") or {}
    skills = candidate.get("skills") or []
    career = candidate.get("career_history") or []
    education = candidate.get("education") or []
    signals = candidate.get("redrob_signals") or {}

    title = str(profile.get("current_title") or "")
    headline = str(profile.get("headline") or "")
    summary = str(profile.get("summary") or "")
    location = str(profile.get("location") or "").lower()
    country = str(profile.get("country") or "")
    yoe = float(profile.get("years_of_experience") or 0)

    title_lower = title.lower()
    profile_text = candidate_profile_text(candidate).lower()
    hp = detect_honeypot(candidate, reference_date=ref)

    # Title fit
    title_fit = 1.0 if title_matches_jd_targets(title, jd) else 0.0
    if matches_senior_ai_title(title, headline):
        title_fit = max(title_fit, 1.0)
    elif "engineer" in title_lower or "scientist" in title_lower:
        title_fit = max(title_fit, 0.5)

    search_eng_title = float(
        any(k in title_lower for k in ("search", "recommendation", "ranking", "retrieval"))
    )
    trap_title = float(any(t in title_lower for t in TRAP_TITLE_KEYWORDS))

    # Coherence
    descriptions = [str(r.get("description") or "") for r in career]
    template_hits = sum(1 for d in descriptions if d and is_known_template(d))
    template_ratio = template_hits / max(1, len(descriptions))
    all_templates = float(descriptions and template_hits == len(descriptions))
    unique_desc_ratio = len({d.strip() for d in descriptions if d}) / max(1, len(descriptions))
    title_summary_sim = fuzz.partial_ratio(title_lower, summary.lower()) / 100.0
    summary_title_mismatch = 1.0 - title_summary_sim if trap_title else max(0.0, 0.5 - title_summary_sim)

    # Skills
    matched = [s for s in skills if skill_matches_jd(str(s.get("name") or ""), jd)]
    jd_match_count = float(len(matched))
    jd_match_ratio = len(matched) / max(1, len(skills))
    prof_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    assessments = signals.get("skill_assessment_scores") or {}
    trusts: list[float] = []
    durations: list[float] = []
    endorsements: list[int] = []
    assess_scores: list[float] = []
    ai_count = 0
    for skill in skills:
        name = str(skill.get("name") or "").lower()
        if any(k in name for k in ("ml", "ai", "nlp", "pytorch", "llm", "rag", "embedding")):
            ai_count += 1
        prof = prof_map.get(str(skill.get("proficiency") or ""), 0.3)
        duration = min(1.0, int(skill.get("duration_months") or 0) / 24)
        endorse = min(1.0, math.log1p(int(skill.get("endorsements") or 0)) / 3)
        sname = str(skill.get("name") or "")
        assess = float(assessments.get(sname, 0) or 0) / 100 if sname in assessments else 1.0
        trusts.append(prof * duration * endorse * assess)
        durations.append(float(skill.get("duration_months") or 0))
        endorsements.append(int(skill.get("endorsements") or 0))
        if sname in assessments:
            assess_scores.append(float(assessments[sname]))
    skill_trust = (sum(trusts) / len(trusts)) if trusts else 0.0
    stuffing = float(trap_title and ai_count >= 8)
    skill_trust = skill_trust - 0.5 * stuffing

    # Career
    consulting_roles = 0
    product_roles = 0
    career_months = 0
    current_months = 0
    for role in career:
        comp = str(role.get("company") or "").lower()
        months = int(role.get("duration_months") or 0)
        career_months += months
        if role.get("is_current"):
            current_months = months
        if any(f in comp for f in CONSULTING_FIRMS):
            consulting_roles += 1
        elif any(ind in comp or ind in str(role.get("description") or "").lower() for ind in PRODUCT_COMPANY_INDICATORS):
            product_roles += 1
    product_ratio = product_roles / max(1, len(career))
    consulting_only = float(career and consulting_roles == len(career))
    progression = 0.0
    if len(career) >= 2:
        progression = float(career[0].get("is_current", False)) + min(1.0, len(career) / 5)
        progression /= 2
    search_kw_hist = min(1.0, count_search_retrieval_hits(profile_text) / 4)

    # Experience band
    yoe_in_band = float(jd.yoe_min <= yoe <= jd.yoe_max)
    if yoe < jd.yoe_min:
        band_distance = jd.yoe_min - yoe
    elif yoe > jd.yoe_max:
        band_distance = yoe - jd.yoe_max
    else:
        band_distance = 0.0
    if jd.yoe_min <= yoe <= jd.yoe_max:
        yoe_score = 1.0
    elif jd.yoe_min - 1 <= yoe <= jd.yoe_max + 3:
        yoe_score = 0.7
    else:
        yoe_score = max(0.2, 1.0 - band_distance / 10)

    # Education
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.35, "unknown": 0.25}
    tier_scores = []
    cs_ai = 0.0
    postgrad = 0.0
    for edu in education:
        tier = str(edu.get("tier") or "unknown")
        tier_scores.append(tier_map.get(tier, 0.25))
        field = str(edu.get("field_of_study") or "").lower()
        degree = str(edu.get("degree") or "").lower()
        if any(k in field for k in ("computer", "ai", "machine learning", "data science")):
            cs_ai = 1.0
        if any(k in degree for k in ("m.", "master", "phd", "doctor")):
            postgrad = 1.0
    edu_tier_score = max(tier_scores) if tier_scores else 0.0

    # Location
    india = float(country.lower() == jd.preferred_country.lower())
    city_match = float(any(c in location for c in JD_PREFERRED_CITIES))
    relocate = float(signals.get("willing_to_relocate", False))
    work_mode = str(signals.get("preferred_work_mode") or "").lower()
    remote_ok = float(work_mode in {"remote", "hybrid", "flexible"})
    completeness = float(signals.get("profile_completeness_score") or 0) / 100.0
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    sal_mid = (float(salary.get("min") or 0) + float(salary.get("max") or 0)) / 2

    # Behavioral
    response = float(signals.get("recruiter_response_rate") or 0)
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days = (ref - last_active).days
        recency = 1.0 if days <= 30 else 0.85 if days <= 90 else 0.7 if days <= 180 else 0.5
    else:
        recency = 0.5
    open_to_work = float(signals.get("open_to_work_flag", False))
    saved = min(1.0, int(signals.get("saved_by_recruiters_30d") or 0) / 8)
    gh_raw = signals.get("github_activity_score")
    github = 0.5 if gh_raw == -1 else min(1.0, max(0.0, float(gh_raw or 0) / 60))
    notice = int(signals.get("notice_period_days") or 90)
    notice_score = 1.0 if notice <= 30 else 0.85 if notice <= 60 else 0.65
    interview = float(signals.get("interview_completion_rate") or 0)

    values = {
        "title_fit_score": title_fit,
        "matches_senior_ai_title": float(matches_senior_ai_title(title, headline)),
        "is_junior_title": float(is_junior_title(title, headline)),
        "search_engineer_title": search_eng_title,
        "trap_title_flag": trap_title,
        "template_match_ratio": template_ratio,
        "all_templates_flag": all_templates,
        "summary_title_mismatch": summary_title_mismatch,
        "career_desc_unique_ratio": unique_desc_ratio,
        "honeypot_score": hp.score,
        "jd_skill_match_count": jd_match_count,
        "jd_skill_match_ratio": jd_match_ratio,
        "skill_trust_weighted": skill_trust,
        "avg_skill_duration": sum(durations) / len(durations) if durations else 0.0,
        "max_endorsement": float(max(endorsements)) if endorsements else 0.0,
        "assessment_avg": sum(assess_scores) / len(assess_scores) if assess_scores else 0.0,
        "stuffing_flag": stuffing,
        "ai_skill_count": float(ai_count),
        "product_company_ratio": product_ratio,
        "consulting_only_flag": consulting_only,
        "career_progression": progression,
        "search_keywords_in_history": search_kw_hist,
        "career_months_total": float(career_months),
        "current_role_months": float(current_months),
        "yoe": yoe,
        "yoe_in_band": yoe_in_band,
        "band_distance": band_distance,
        "yoe_score": yoe_score,
        "education_tier_score": edu_tier_score,
        "cs_ai_field": cs_ai,
        "has_postgrad": postgrad,
        "education_count": float(len(education)),
        "india_flag": india,
        "preferred_city_match": city_match,
        "relocate_flag": relocate,
        "remote_ok": remote_ok,
        "profile_completeness": completeness,
        "salary_mid_lpa": sal_mid,
        "response_rate": response,
        "recency_score": recency,
        "open_to_work": open_to_work,
        "saved_by_recruiters_norm": saved,
        "github_score_norm": github,
        "notice_period_score": notice_score,
        "interview_rate": interview,
    }
    return {name: float(values[name]) for name in FEATURE_NAMES}


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
