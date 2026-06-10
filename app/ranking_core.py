"""Shared post-fusion ranking — modifiers, caps, tier reorder."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.jd_requirements import JDRequirements
from app.modifier_params import ModifierParams
from app.ranking_modifiers import (
    apply_clone_cap,
    apply_top_availability_cap,
    final_score_multiplier,
    is_senior_ai_summary_clone,
)


def apply_modifiers_to_scores(
    final: dict[str, float],
    rerank_ids: list[str],
    *,
    jd: JDRequirements,
    candidate_lookup: dict[str, dict],
    feature_lookup: dict[str, dict],
    modifier_params: ModifierParams,
    career_scores: dict[str, float] | None,
    reference_date: date,
) -> dict[str, float]:
    adjusted = dict(final)
    for cid in rerank_ids:
        candidate = candidate_lookup.get(cid, {})
        row = feature_lookup.get(cid, {})
        adjusted[cid] = final.get(cid, 0.0) * final_score_multiplier(
            candidate,
            jd,
            reference_date=reference_date,
            silver_tier=int(row.get("silver_tier") or 2),
            params=modifier_params,
            career_scores=career_scores,
        )
    return adjusted


def sort_by_score_and_tier(
    final: dict[str, float],
    feature_lookup: dict[str, dict],
) -> list[str]:
    def sort_key(cid: str) -> tuple:
        tier = int(feature_lookup.get(cid, {}).get("silver_tier") or 2)
        return (-final.get(cid, 0.0), -tier, cid)

    return sorted(final.keys(), key=sort_key)


def reorder_top10_by_tier(
    ranked_ids: list[str],
    feature_lookup: dict[str, dict],
    final: dict[str, float],
) -> list[str]:
    """Within ranks 1–10, prefer higher silver tier then fused score."""
    if len(ranked_ids) <= 10:
        return ranked_ids

    def head_key(cid: str) -> tuple:
        tier = int(feature_lookup.get(cid, {}).get("silver_tier") or 2)
        return (-tier, -final.get(cid, 0.0), cid)

    head = sorted(ranked_ids[:10], key=head_key)
    return head + ranked_ids[10:]


def rank_with_modifiers(
    rerank_ids: list[str],
    final: dict[str, float],
    *,
    jd: JDRequirements,
    candidate_lookup: dict[str, dict],
    feature_lookup: dict[str, dict],
    modifier_params: ModifierParams,
    career_scores: dict[str, float] | None,
    reference_date: date,
    top_k: int = 100,
) -> list[str]:
    """Full post-fusion path shared by pipeline and fusion cache tuning."""
    adjusted = apply_modifiers_to_scores(
        final,
        rerank_ids,
        jd=jd,
        candidate_lookup=candidate_lookup,
        feature_lookup=feature_lookup,
        modifier_params=modifier_params,
        career_scores=career_scores,
        reference_date=reference_date,
    )
    ranked = sort_by_score_and_tier(adjusted, feature_lookup)
    ranked = apply_top_availability_cap(
        ranked,
        candidate_lookup,
        cap=modifier_params.top_availability_cap,
        top_k=top_k,
        reference_date=reference_date,
    )
    ranked = apply_clone_cap(
        ranked,
        candidate_lookup,
        final=adjusted,
        max_clones=modifier_params.clone_top30_max,
        cap_zone=30,
        top_k=top_k,
    )
    ranked = reorder_top10_by_tier(ranked, feature_lookup, adjusted)
    return ranked[:top_k]
