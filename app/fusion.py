"""Stage-3 fusion — cache expensive scores and tune blend weights offline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.artifact_names import FUSION_PARAMS as FUSION_PARAMS_NAME
from app.artifact_names import MODIFIER_PARAMS as MODIFIER_PARAMS_NAME
from app.artifacts_check import validate_fusion_weights
from app.behavioral import behavioral_multiplier
from app.career_recall import load_career_scores
from app.config import Settings
from app.feature_store import load_silver_tiers
from app.jd_requirements import JDRequirements, load_or_build_jd_requirements
from app.ltr import load_ltr_model, score_candidates_by_id
from app.modifier_params import ModifierParams, load_modifier_params
from app.rank_data import load_candidates_lookup, load_feature_lookup
from app.ranking_core import rank_with_modifiers
from app.ranking_modifiers import is_low_availability, is_senior_ai_summary_clone
from app.recall import hybrid_recall, normalize_scores
from app.reranker import rerank_candidates
from app.traps import research_title_penalty, should_hard_exclude, trap_penalty

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionScoreCache:
    """Precomputed scores for fast weight / pool-size search."""

    rerank_ids: tuple[str, ...]
    ce_norm: dict[str, float]
    ltr_norm: dict[str, float]
    rrf_norm: dict[str, float]
    stage2_raw: dict[str, float]


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    ranked_labels = labels[order][:k]
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ranked_labels))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def fuse_scores(
    rerank_ids: list[str],
    *,
    ce_norm: dict[str, float],
    ltr_norm: dict[str, float],
    rrf_norm: dict[str, float],
    ce_weight: float,
    ltr_weight: float,
    rrf_weight: float,
) -> dict[str, float]:
    final: dict[str, float] = {}
    for cid in rerank_ids:
        final[cid] = (
            ce_weight * ce_norm.get(cid, 0.0)
            + ltr_weight * ltr_norm.get(cid, 0.0)
            + rrf_weight * rrf_norm.get(cid, 0.0)
        )
    return final


def rank_ids_from_fusion(
    cache: FusionScoreCache,
    *,
    rerank_pool_size: int,
    top_k: int,
    ce_weight: float,
    ltr_weight: float,
    rrf_weight: float,
) -> list[str]:
    rerank_ids = list(cache.rerank_ids[:rerank_pool_size])
    if not rerank_ids:
        return []
    final = fuse_scores(
        rerank_ids,
        ce_norm=cache.ce_norm,
        ltr_norm=cache.ltr_norm,
        rrf_norm=cache.rrf_norm,
        ce_weight=ce_weight,
        ltr_weight=ltr_weight,
        rrf_weight=rrf_weight,
    )
    return sorted(final, key=lambda cid: (-final[cid], cid))[:top_k]


def evaluate_proxy_ranking(
    ranked_ids: list[str],
    silver_tiers: dict[str, int],
    *,
    honeypot_tier: int = 0,
) -> dict[str, float]:
    top10 = ranked_ids[:10]
    labels10 = np.array([silver_tiers.get(cid, 2) for cid in top10], dtype=np.float64)
    scores10 = np.arange(len(labels10), 0, -1, dtype=np.float64)
    tier5 = sum(1 for cid in top10 if silver_tiers.get(cid, 2) == 5)
    honeypots_top10 = sum(1 for cid in top10 if silver_tiers.get(cid, 2) == honeypot_tier)
    honeypots_top100 = sum(1 for cid in ranked_ids[:100] if silver_tiers.get(cid, 2) == honeypot_tier)
    return {
        "ndcg_at_10": ndcg_at_k(labels10, scores10, k=10),
        "tier5_in_top_10": float(tier5),
        "honeypots_in_top_10": float(honeypots_top10),
        "honeypots_in_top_100": float(honeypots_top100),
    }


def proxy_objective(metrics: dict[str, float]) -> float:
    """Maximize NDCG@10; heavily penalize honeypots in top 10."""
    penalty = metrics["honeypots_in_top_10"] * 0.5 + metrics["honeypots_in_top_100"] * 0.05
    return metrics["ndcg_at_10"] - penalty


def build_fusion_cache(
    settings: Settings,
    artifacts_dir: Path,
    candidates_path: Path,
    *,
    max_rerank_pool: int = 800,
) -> FusionScoreCache:
    """Run recall, LTR, and cross-encoder once (up to max_rerank_pool candidates)."""
    jd_path = settings.job_description_path
    jd = load_or_build_jd_requirements(
        jd_path,
        artifacts_dir / "jd_requirements.json",
        settings=settings,
    )
    jd_text = jd_path.read_text(encoding="utf-8")
    feature_lookup = load_feature_lookup(artifacts_dir / "candidate_features.parquet")
    candidate_lookup = load_candidates_lookup(candidates_path)

    recall_pool, rrf_scores = hybrid_recall(
        jd,
        jd_text,
        artifacts_dir,
        bm25_k=settings.bm25_recall_k,
        dense_k=settings.dense_recall_k,
        pool_size=settings.recall_pool_size,
        rrf_k=settings.rrf_k,
        career_rrf_weight=settings.career_rrf_weight,
    )

    pool_ids = [cid for cid, _ in recall_pool if cid in feature_lookup]
    pool_ids = [cid for cid in pool_ids if not should_hard_exclude(feature_lookup[cid])]

    model = load_ltr_model(artifacts_dir / "ltr_model.lgb")
    ltr_raw = score_candidates_by_id(model, feature_lookup, pool_ids)

    stage2: dict[str, float] = {}
    for cid in pool_ids:
        row = feature_lookup[cid]
        candidate = candidate_lookup.get(cid, {})
        base = float(ltr_raw.get(cid, 0.0))
        penalty = trap_penalty(row) + research_title_penalty(row, candidate)
        behavior = behavioral_multiplier(candidate, reference_date=settings.reference_date)
        stage2[cid] = (base - penalty) * behavior

    rerank_ids = sorted(stage2, key=lambda cid: (-stage2[cid], cid))[:max_rerank_pool]
    rerank_candidates_list = [candidate_lookup[cid] for cid in rerank_ids if cid in candidate_lookup]
    jd_summary = f"{jd.role_title}. {' '.join(jd.must_have_skills[:10])}. YOE {jd.yoe_min}-{jd.yoe_max}."
    logger.info("Fusion cache: cross-encoder scoring %d candidates", len(rerank_candidates_list))
    ce_raw = rerank_candidates(jd_summary, rerank_candidates_list)

    ce_norm = normalize_scores({cid: ce_raw.get(cid, 0.0) for cid in rerank_ids})
    ltr_norm = normalize_scores({cid: stage2.get(cid, 0.0) for cid in rerank_ids})
    rrf_norm = normalize_scores({cid: rrf_scores.get(cid, 0.0) for cid in rerank_ids})

    logger.info(
        "Fusion cache ready: recall_pool=%d stage2=%d ce=%d",
        len(pool_ids),
        len(stage2),
        len(rerank_ids),
    )
    return FusionScoreCache(
        rerank_ids=tuple(rerank_ids),
        ce_norm=ce_norm,
        ltr_norm=ltr_norm,
        rrf_norm=rrf_norm,
        stage2_raw=stage2,
    )


def load_fusion_params(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    params = json.loads(path.read_text(encoding="utf-8"))
    validate_fusion_weights(params)
    return params


def apply_fusion_params(settings: Settings, params: dict[str, Any]) -> Settings:
    updates: dict[str, Any] = {}
    for key in (
        "rerank_ce_weight",
        "rerank_ltr_weight",
        "rerank_rrf_weight",
        "rerank_pool_size",
        "career_rrf_weight",
    ):
        if key in params:
            updates[key] = params[key]
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def settings_with_fusion_params(settings: Settings, artifacts_dir: Path) -> Settings:
    params = load_fusion_params(artifacts_dir / FUSION_PARAMS_NAME)
    if params:
        logger.info("Loaded fusion params from %s", artifacts_dir / FUSION_PARAMS_NAME)
        settings = apply_fusion_params(settings, params)
    mod = load_modifier_params(artifacts_dir / MODIFIER_PARAMS_NAME)
    if (artifacts_dir / MODIFIER_PARAMS_NAME).exists():
        logger.info("Loaded modifier params from %s", artifacts_dir / MODIFIER_PARAMS_NAME)
        settings = settings.model_copy(update={"top_availability_cap": mod.top_availability_cap})
    return settings


def rank_from_fusion_cache(
    cache: FusionScoreCache,
    *,
    jd: JDRequirements,
    candidate_lookup: dict[str, dict],
    feature_lookup: dict[str, dict],
    reference_date,
    rerank_pool_size: int,
    top_k: int,
    ce_weight: float,
    ltr_weight: float,
    rrf_weight: float,
    modifier_params: ModifierParams,
    career_scores: dict[str, float] | None = None,
) -> list[str]:
    """Full post-fusion ranking path used for modifier tuning."""
    rerank_ids = list(cache.rerank_ids[:rerank_pool_size])
    final = fuse_scores(
        rerank_ids,
        ce_norm=cache.ce_norm,
        ltr_norm=cache.ltr_norm,
        rrf_norm=cache.rrf_norm,
        ce_weight=ce_weight,
        ltr_weight=ltr_weight,
        rrf_weight=rrf_weight,
    )
    return rank_with_modifiers(
        rerank_ids,
        final,
        jd=jd,
        candidate_lookup=candidate_lookup,
        feature_lookup=feature_lookup,
        modifier_params=modifier_params,
        career_scores=career_scores,
        reference_date=reference_date,
        top_k=top_k,
    )


def evaluate_full_ranking(
    ranked_ids: list[str],
    silver_tiers: dict[str, int],
    candidate_lookup: dict[str, dict],
    *,
    reference_date,
    honeypot_tier: int = 0,
) -> dict[str, float]:
    metrics = evaluate_proxy_ranking(ranked_ids, silver_tiers, honeypot_tier=honeypot_tier)
    top30 = ranked_ids[:30]
    metrics["low_avail_top_30"] = float(
        sum(1 for cid in top30 if is_low_availability(candidate_lookup.get(cid, {}), reference_date=reference_date))
    )
    metrics["tier5_in_top_30"] = float(sum(1 for cid in top30 if silver_tiers.get(cid, 2) == 5))
    metrics["tier5_in_top_100"] = float(sum(1 for cid in ranked_ids[:100] if silver_tiers.get(cid, 2) == 5))
    metrics["template_clones_top_30"] = float(
        sum(
            1
            for cid in top30
            if is_senior_ai_summary_clone(candidate_lookup.get(cid, {}))
        )
    )
    labels30 = [silver_tiers.get(cid, 2) for cid in top30]
    import numpy as np

    metrics["ndcg_at_30"] = ndcg_at_k(
        np.array(labels30, dtype=float),
        np.arange(len(labels30), 0, -1, dtype=float),
        k=30,
    )
    return metrics


def update_live_metrics(
    artifacts_dir: Path,
    ranked_ids: list[str],
    candidates_path: Path,
    *,
    reference_date,
) -> dict[str, float]:
    """Persist live pipeline metrics into fusion_params.json and modifier_params.json."""
    silver = load_silver_tiers(artifacts_dir / "labels_silver.parquet")
    candidate_lookup = load_candidates_lookup(candidates_path)
    metrics = evaluate_full_ranking(
        ranked_ids,
        silver,
        candidate_lookup,
        reference_date=reference_date,
    )

    fusion_path = artifacts_dir / FUSION_PARAMS_NAME
    if fusion_path.exists():
        fusion_data = json.loads(fusion_path.read_text(encoding="utf-8"))
        fusion_data["live_ndcg_at_10"] = round(metrics["ndcg_at_10"], 4)
        fusion_data["live_tier5_in_top_10"] = metrics["tier5_in_top_10"]
        fusion_data["live_tier5_in_top_30"] = metrics["tier5_in_top_30"]
        fusion_data["live_tier5_in_top_100"] = metrics["tier5_in_top_100"]
        fusion_data["live_template_clones_top_30"] = metrics.get("template_clones_top_30", 0)
        fusion_data["live_ndcg_at_30"] = round(metrics.get("ndcg_at_30", 0), 4)
        fusion_path.write_text(json.dumps(fusion_data, indent=2), encoding="utf-8")

    mod_path = artifacts_dir / MODIFIER_PARAMS_NAME
    if mod_path.exists():
        mod_data = json.loads(mod_path.read_text(encoding="utf-8"))
        mod_data["live_ndcg_at_10"] = metrics["ndcg_at_10"]
        mod_data["live_tier5_in_top_10"] = metrics["tier5_in_top_10"]
        mod_data["live_tier5_in_top_30"] = metrics["tier5_in_top_30"]
        mod_data["live_tier5_in_top_100"] = metrics["tier5_in_top_100"]
        mod_path.write_text(json.dumps(mod_data, indent=2), encoding="utf-8")

    logger.info(
        "Live metrics: NDCG@10=%.4f tier5@30=%.0f clones@30=%.0f",
        metrics["ndcg_at_10"],
        metrics["tier5_in_top_30"],
        metrics.get("template_clones_top_30", 0),
    )
    return metrics
