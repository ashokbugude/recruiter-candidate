"""Ranking modifier unit tests."""

from __future__ import annotations

from app.modifier_params import ModifierParams
from app.ranking_modifiers import (
    apply_clone_cap,
    apply_top_availability_cap,
    final_score_multiplier,
    fusion_to_submission_scores,
    is_cv_speech_heavy_stack,
    is_low_availability,
    is_primary_hub,
    is_secondary_hub,
    is_senior_ai_summary_clone,
    is_stretched_scientist,
)
from app.jd_requirements import build_heuristic_requirements
from app.config import PROJECT_ROOT


def _clone_candidate(response_rate: float, candidate_id: str = "C1") -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "summary": (
                "Senior AI engineer with 5 years of hands-on experience building production ML systems, "
                "with a focus on search, retrieval, and ranking at scale."
            ),
            "current_title": "Senior AI Engineer",
        },
        "redrob_signals": {
            "recruiter_response_rate": response_rate,
            "last_active_date": "2026-05-01",
            "notice_period_days": 30,
            "open_to_work_flag": True,
        },
    }


def test_senior_ai_clone_detection() -> None:
    assert is_senior_ai_summary_clone(_clone_candidate(0.8))
    assert not is_senior_ai_summary_clone({"profile": {"summary": "Backend engineer in fintech."}})


def test_low_availability_by_response_rate() -> None:
    assert is_low_availability(_clone_candidate(0.07))
    assert not is_low_availability(_clone_candidate(0.80))


def test_final_score_multiplier_penalizes_low_rr_clone() -> None:
    jd = build_heuristic_requirements(
        (PROJECT_ROOT / "artifacts" / "job_description.txt").read_text(encoding="utf-8")
    )
    high = final_score_multiplier(_clone_candidate(0.80), jd)
    low = final_score_multiplier(_clone_candidate(0.07), jd)
    assert low < high * 0.5


def test_apply_top_availability_cap() -> None:
    lookup = {
        "good": _clone_candidate(0.80, "good"),
        "bad": _clone_candidate(0.05, "bad"),
    }
    ranked = apply_top_availability_cap(["bad", "good"], lookup, cap=1, top_k=2)
    assert ranked[0] == "good"


def test_fusion_to_submission_scores_monotonic() -> None:
    ids = ["a", "b", "c"]
    fused = {"a": 3.0, "b": 2.0, "c": 1.0}
    scores = fusion_to_submission_scores(ids, fused)
    assert scores["a"] > scores["b"] > scores["c"]


def test_cv_speech_stack_detected() -> None:
    cand = {
        "profile": {"summary": "Built search systems at scale"},
        "skills": [{"name": "YOLO"}, {"name": "Kubeflow"}, {"name": "Hadoop"}],
        "career_history": [{"description": "search ranking retrieval elasticsearch"}],
    }
    assert is_cv_speech_heavy_stack(cand)


def test_fusion_to_submission_scores_breaks_ties() -> None:
    ids = ["a", "b", "c"]
    fused = {"a": 1.0, "b": 1.0, "c": 1.0}
    scores = fusion_to_submission_scores(ids, fused)
    assert scores["a"] > scores["b"] > scores["c"]


def test_fusion_to_submission_scores_reflects_model_spread() -> None:
    ids = ["a", "b", "c"]
    fused = {"a": 1.25, "b": 0.9, "c": 0.5}
    scores = fusion_to_submission_scores(ids, fused)
    assert scores["a"] > scores["b"] > scores["c"]
    assert scores["a"] - scores["c"] > 0.3


def test_fusion_to_submission_scores_monotonic_after_reorder() -> None:
    """Post-reorder rank order may invert fused magnitudes — scores stay decreasing."""
    ids = ["low", "high"]
    fused = {"low": 0.95, "high": 1.2}
    scores = fusion_to_submission_scores(ids, fused)
    assert scores["low"] > scores["high"]


def test_fusion_to_submission_scores_collapsed_spread_uses_nudge() -> None:
    ids = ["a", "b", "c"]
    fused = {"a": 1.25, "b": 1.25, "c": 1.249}
    scores = fusion_to_submission_scores(ids, fused)
    assert scores["a"] > scores["b"] > scores["c"]
    assert len({scores[c] for c in ids}) == 3


def test_fusion_to_submission_scores_never_exceed_one() -> None:
    ids = [f"c{i}" for i in range(24)]
    fused = {cid: float(i) for i, cid in enumerate(ids)}
    scores = fusion_to_submission_scores(ids, fused)
    assert all(score <= 0.9999 for score in scores.values())
    assert scores[ids[0]] > scores[ids[-1]]


def test_research_with_prod_less_punitive_than_no_prod() -> None:
    jd = build_heuristic_requirements(
        (PROJECT_ROOT / "artifacts" / "job_description.txt").read_text(encoding="utf-8")
    )
    p = ModifierParams()
    base = {
        "candidate_id": "R1",
        "profile": {"current_title": "Research Scientist", "summary": "shipped production search systems"},
        "skills": [],
        "career_history": [{"description": "search retrieval elasticsearch ranking"}],
        "redrob_signals": {"recruiter_response_rate": 0.8, "last_active_date": "2026-05-01"},
    }
    no_prod = dict(base)
    no_prod["profile"] = {"current_title": "Research Scientist", "summary": "papers only"}
    with_prod = final_score_multiplier(base, jd, params=p)
    without_prod = final_score_multiplier(no_prod, jd, params=p)
    assert with_prod > without_prod


def test_stretched_scientist_tier5_floor() -> None:
    jd = build_heuristic_requirements("AI scientist production search retrieval ranking")
    p = ModifierParams(stretched_scientist_mult=0.6)
    cand = {
        "candidate_id": "S1",
        "profile": {
            "current_title": "Principal Scientist",
            "years_of_experience": 15,
            "summary": "shipped production search retrieval ranking systems",
        },
        "skills": [],
        "career_history": [{"description": "search retrieval ranking elasticsearch"}],
        "redrob_signals": {"recruiter_response_rate": 0.8},
    }
    tier5 = final_score_multiplier(cand, jd, silver_tier=5, params=p)
    tier4 = final_score_multiplier(cand, jd, silver_tier=4, params=p)
    assert tier5 > tier4


def test_plain_language_career_boost_gated() -> None:
    jd = build_heuristic_requirements("search retrieval")
    p = ModifierParams(plain_language_career_boost=1.1, plain_language_career_threshold=0.5)
    cand = {"candidate_id": "P1", "profile": {"summary": "unique backend profile"}, "redrob_signals": {"recruiter_response_rate": 0.8}}
    low = final_score_multiplier(cand, jd, silver_tier=5, params=p, career_scores={"P1": 0.3})
    high = final_score_multiplier(cand, jd, silver_tier=5, params=p, career_scores={"P1": 0.7})
    assert high > low
    clone = _clone_candidate(0.8, "clone")
    clone_score = final_score_multiplier(clone, jd, silver_tier=5, params=p, career_scores={"clone": 0.9})
    assert clone_score < high


def test_secondary_hub_not_double_primary() -> None:
    from app.constants import JD_PRIMARY_HUBS, JD_SECONDARY_HUBS

    primary_hub = next(iter(JD_PRIMARY_HUBS))
    secondary_hub = next(iter(JD_SECONDARY_HUBS))
    primary = {"profile": {"location": f"{primary_hub}, India"}}
    secondary = {"profile": {"location": f"{secondary_hub}, India"}}
    assert is_primary_hub(primary)
    assert is_secondary_hub(secondary)
    assert not is_secondary_hub(primary)


def test_apply_clone_cap_demotes_excess() -> None:
    lookup = {f"c{i}": _clone_candidate(0.8, f"c{i}") for i in range(5)}
    for name in ("normal1", "normal2", "normal3"):
        lookup[name] = {
            "candidate_id": name,
            "profile": {"summary": f"Platform engineer {name}"},
            "redrob_signals": {"recruiter_response_rate": 0.8},
        }
    ranked = [f"c{i}" for i in range(5)] + ["normal1", "normal2", "normal3"]
    final = {cid: 1.0 - i * 0.01 for i, cid in enumerate(ranked)}
    capped = apply_clone_cap(ranked, lookup, final=final, max_clones=2, cap_zone=5, top_k=6)
    head_clones = sum(1 for cid in capped[:5] if is_senior_ai_summary_clone(lookup[cid]))
    assert head_clones <= 2


def test_is_stretched_scientist_requires_high_yoe() -> None:
    assert is_stretched_scientist({"profile": {"current_title": "Scientist", "years_of_experience": 14}})
    assert not is_stretched_scientist({"profile": {"current_title": "Scientist", "years_of_experience": 8}})
