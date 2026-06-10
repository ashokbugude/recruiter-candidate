"""Fusion weight tuning helpers."""

from __future__ import annotations

from app.fusion import FusionScoreCache, fuse_scores, proxy_objective, rank_ids_from_fusion


def test_fuse_scores_weighted_sum() -> None:
    ids = ["a", "b"]
    final = fuse_scores(
        ids,
        ce_norm={"a": 1.0, "b": 0.0},
        ltr_norm={"a": 0.0, "b": 1.0},
        rrf_norm={"a": 0.5, "b": 0.5},
        ce_weight=0.6,
        ltr_weight=0.3,
        rrf_weight=0.1,
    )
    assert final["a"] > final["b"]


def test_rank_ids_from_fusion_respects_pool_size() -> None:
    cache = FusionScoreCache(
        rerank_ids=("a", "b", "c", "d"),
        ce_norm={"a": 1.0, "b": 0.8, "c": 0.2, "d": 0.0},
        ltr_norm={"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.5},
        rrf_norm={"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},
        stage2_raw={"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0},
    )
    ranked = rank_ids_from_fusion(
        cache,
        rerank_pool_size=2,
        top_k=2,
        ce_weight=1.0,
        ltr_weight=0.0,
        rrf_weight=0.0,
    )
    assert ranked == ["a", "b"]


def test_proxy_objective_penalizes_honeypots() -> None:
    clean = proxy_objective({"ndcg_at_10": 0.9, "honeypots_in_top_10": 0.0, "honeypots_in_top_100": 0.0})
    dirty = proxy_objective({"ndcg_at_10": 0.9, "honeypots_in_top_10": 1.0, "honeypots_in_top_100": 0.0})
    assert clean > dirty
