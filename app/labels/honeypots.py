"""12-rule honeypot and trap detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.constants import (
    AI_SKILL_NAMES,
    KNOWN_TEMPLATE_HASHES,
    REFERENCE_DATE,
    SUBTLE_HONEYPOT_RULES,
    SUBTLE_HONEYPOT_SINGLE_THRESHOLD,
    SUBTLE_HONEYPOT_WEIGHTS,
    career_description_hash,
)

@dataclass(frozen=True)
class HoneypotResult:
    """Detection result separating subtle honeypots (tier 0) from template traps."""

    candidate_id: str
    score: float
    rules_hit: tuple[str, ...] = field(default_factory=tuple)
    is_honeypot: bool = False
    is_trap: bool = False
    should_exclude: bool = False


def detect_honeypot(
    candidate: dict[str, Any],
    *,
    reference_date: date | None = None,
) -> HoneypotResult:
    """Evaluate honeypot/trap rules with reproducible reference date."""
    ref = reference_date or REFERENCE_DATE
    hits: list[str] = []

    profile = candidate.get("profile") or {}
    skills: list[dict[str, Any]] = candidate.get("skills") or []
    education: list[dict[str, Any]] = candidate.get("education") or []
    career: list[dict[str, Any]] = candidate.get("career_history") or []
    signals = candidate.get("redrob_signals") or {}

    # R1: expert skill with zero duration
    if any(
        s.get("proficiency") == "expert" and int(s.get("duration_months") or 0) == 0 for s in skills
    ):
        hits.append("R1")

    # R2: claimed tenure exceeds calendar time in role (+6 month buffer)
    for role in career:
        if _tenure_exceeds_calendar(role, ref):
            hits.append("R2")
            break

    # R3: education end before start
    for edu in education:
        start = int(edu.get("start_year") or 0)
        end = int(edu.get("end_year") or 0)
        if end < start:
            hits.append("R3")
            break

    # R4: salary min > max (trap signal — not a subtle honeypot alone)
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    sal_min = float(salary.get("min") or 0)
    sal_max = float(salary.get("max") or 0)
    if sal_min > sal_max > 0:
        hits.append("R4")

    # R5: 10+ expert skills with avg duration < 6 months
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 10:
        avg_duration = sum(int(s.get("duration_months") or 0) for s in expert_skills) / len(
            expert_skills
        )
        if avg_duration < 6:
            hits.append("R5")

    # R6: yoe inconsistent with career history sum
    yoe = float(profile.get("years_of_experience") or 0)
    career_months = sum(int(r.get("duration_months") or 0) for r in career)
    if career_months > 0 and yoe < (career_months / 12.0) - 2:
        hits.append("R6")

    # R7: all career descriptions match known recycled templates
    descriptions = [str(r.get("description") or "").strip() for r in career if r.get("description")]
    if descriptions:
        template_hits = sum(
            1 for d in descriptions if career_description_hash(d)[:12] in KNOWN_TEMPLATE_HASHES
        )
        if template_hits == len(descriptions):
            hits.append("R7")

    # R8: github == -1 but claims OSS / infra expertise (trap modifier, not tier 0)
    github_score = signals.get("github_activity_score")
    if github_score == -1:
        summary = str(profile.get("summary") or "").lower()
        oss_claim = any(
            token in summary
            for token in ("open source", "open-source", "oss contributor", "github star")
        )
        skill_claim = any(
            "open source" in str(s.get("name") or "").lower()
            for s in skills
            if s.get("proficiency") in {"advanced", "expert"}
        )
        if oss_claim or skill_claim:
            hits.append("R8")

    # R9: assessment > 90 with beginner proficiency on same skill
    assessments = signals.get("skill_assessment_scores") or {}
    for skill in skills:
        if skill.get("proficiency") != "beginner":
            continue
        name = str(skill.get("name") or "")
        score = float(assessments.get(name, 0) or 0)
        if score > 90:
            hits.append("R9")
            break

    # R10: inactive >365 days (weak trap signal — excluded from subtle honeypot tier 0)
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active and (ref - last_active).days > 365:
        hits.append("R10")

    # R11: marketing/non-AI title with 8+ AI skills
    title = str(profile.get("current_title") or "").lower()
    summary = str(profile.get("summary") or "").lower()
    ai_skill_count = _count_ai_skills(skills)
    non_ai_title = any(
        trap in title
        for trap in ("marketing manager", "hr manager", "content writer", "accountant", "sales")
    )
    if non_ai_title and ai_skill_count >= 8 and "engineer" not in title:
        hits.append("R11")
    elif "marketing manager" in summary and ai_skill_count >= 8 and "engineer" not in title:
        hits.append("R11")

    # R12 reserved for Gemini tier-0 (offline Phase 2)

    subtle_hits = [rule for rule in hits if rule in SUBTLE_HONEYPOT_RULES]
    subtle_score = sum(SUBTLE_HONEYPOT_WEIGHTS.get(rule, 0.0) for rule in subtle_hits)
    is_honeypot = _is_subtle_honeypot(subtle_hits, subtle_score)
    is_trap = "R7" in hits or "R11" in hits or ("R4" in hits and "R7" in hits)

    # Full diagnostic score (all rules) for logging / features
    all_weights = {
        "R1": 0.35,
        "R2": 0.40,
        "R3": 0.45,
        "R4": 0.35,
        "R5": 0.40,
        "R6": 0.30,
        "R7": 0.60,
        "R8": 0.25,
        "R9": 0.40,
        "R10": 0.15,
        "R11": 0.45,
    }
    score = min(1.0, sum(all_weights.get(rule, 0.2) for rule in hits))

    # Runtime hard exclude: subtle honeypots OR recycled-template profiles (R7)
    should_exclude = is_honeypot or "R7" in hits

    return HoneypotResult(
        candidate_id=str(candidate.get("candidate_id") or ""),
        score=round(score, 4),
        rules_hit=tuple(hits),
        is_honeypot=is_honeypot,
        is_trap=is_trap,
        should_exclude=should_exclude,
    )


def _is_subtle_honeypot(subtle_hits: list[str], subtle_score: float) -> bool:
    """Tier-0 ground-truth-style honeypots only — not template/salary traps."""
    if not subtle_hits:
        return False
    if any(SUBTLE_HONEYPOT_WEIGHTS.get(rule, 0) >= SUBTLE_HONEYPOT_SINGLE_THRESHOLD for rule in subtle_hits):
        return True
    if len(subtle_hits) >= 2 and subtle_score >= 0.55:
        return True
    return False


def _tenure_exceeds_calendar(role: dict[str, Any], ref: date) -> bool:
    """True when claimed duration exceeds calendar months in role (+6 month buffer)."""
    start = _parse_date(role.get("start_date"))
    if start is None:
        return False
    claimed_months = int(role.get("duration_months") or 0)
    calendar_months = (ref.year - start.year) * 12 + (ref.month - start.month)
    if ref.day < start.day:
        calendar_months -= 1
    calendar_months = max(0, calendar_months)
    return claimed_months > calendar_months + 6


def _count_ai_skills(skills: list[dict[str, Any]]) -> int:
    count = 0
    for skill in skills:
        name = str(skill.get("name") or "").lower()
        if name in AI_SKILL_NAMES or any(k in name for k in ("ml", "ai", "llm", "nlp", "pytorch")):
            count += 1
    return count


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
