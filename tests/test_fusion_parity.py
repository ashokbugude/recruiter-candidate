"""Cache path vs rank_with_modifiers must return identical order."""

from __future__ import annotations

from datetime import date

from app.fusion import FusionScoreCache, rank_from_fusion_cache
from app.jd_requirements import build_heuristic_requirements
from app.modifier_params import ModifierParams
from app.ranking_core import rank_with_modifiers


def _minimal_fixtures():
    jd = build_heuristic_requirements("Senior AI Engineer search retrieval ranking")
    rerank_ids = ["a", "b", "c", "d"]
    final = {"a": 0.9, "b": 0.85, "c": 0.8, "d": 0.75}
    feature_lookup = {
        cid: {"silver_tier": 5 if cid in ("a", "b") else 3}
        for cid in rerank_ids
    }
    candidate_lookup = {
        cid: {
            "candidate_id": cid,
            "profile": {"summary": f"Engineer {cid}", "location": "Bangalore, India", "country": "India"},
            "redrob_signals": {"recruiter_response_rate": 0.8, "last_active_date": "2026-05-01"},
        }
        for cid in rerank_ids
    }
    return jd, rerank_ids, final, feature_lookup, candidate_lookup


def test_fusion_cache_matches_rank_with_modifiers() -> None:
    jd, rerank_ids, final, feature_lookup, candidate_lookup = _minimal_fixtures()
    mod = ModifierParams(top_availability_cap=10, clone_top30_max=10)
    ref = date(2026, 6, 1)

    via_core = rank_with_modifiers(
        rerank_ids,
        dict(final),
        jd=jd,
        candidate_lookup=candidate_lookup,
        feature_lookup=feature_lookup,
        modifier_params=mod,
        career_scores=None,
        reference_date=ref,
        top_k=4,
    )

    cache = FusionScoreCache(
        rerank_ids=tuple(rerank_ids),
        ce_norm=final,
        ltr_norm=final,
        rrf_norm=final,
        stage2_raw=final,
    )
    via_cache = rank_from_fusion_cache(
        cache,
        jd=jd,
        candidate_lookup=candidate_lookup,
        feature_lookup=feature_lookup,
        reference_date=ref,
        rerank_pool_size=4,
        top_k=4,
        ce_weight=1.0,
        ltr_weight=0.0,
        rrf_weight=0.0,
        modifier_params=mod,
        career_scores=None,
    )
    assert via_cache == via_core


def test_fuse_scores_weights_sum() -> None:
    from app.artifacts_check import validate_fusion_weights

    validate_fusion_weights({"rerank_ce_weight": 0.5, "rerank_ltr_weight": 0.3, "rerank_rrf_weight": 0.2})
