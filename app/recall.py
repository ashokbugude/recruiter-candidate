"""Hybrid recall — BM25 + dense FAISS + career RRF."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.bm25_index import load_bm25, search_bm25
from app.career_recall import career_recall_ranking
from app.embeddings import encode_text, load_embedding_index, search_faiss
from app.jd_requirements import JDRequirements

logger = logging.getLogger(__name__)


def build_jd_query_text(jd: JDRequirements, jd_text: str) -> str:
    """Build sparse + dense retrieval query from JD."""
    parts = [
        jd.role_title,
        jd_text,
        " ".join(jd.must_have_skills),
        " ".join(jd.nice_to_have_skills),
        " ".join(jd.must_have_keywords),
        " ".join(jd.target_titles),
    ]
    return " ".join(p for p in parts if p)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked ID lists with RRF."""
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, candidate_id in enumerate(ranking, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + rank)
    return scores


def reciprocal_rank_fusion_weighted(
    lists: list[tuple[list[str], float]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked lists with per-list weights."""
    scores: dict[str, float] = {}
    for ranking, weight in lists:
        if weight <= 0 or not ranking:
            continue
        for rank, candidate_id in enumerate(ranking, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + weight * (1.0 / (k + rank))
    return scores


def hybrid_recall(
    jd: JDRequirements,
    jd_text: str,
    artifacts_dir: Path,
    *,
    bm25_k: int = 3000,
    dense_k: int = 3000,
    pool_size: int = 2000,
    rrf_k: int = 60,
    career_rrf_weight: float = 0.65,
) -> list[tuple[str, float]]:
    """Return top recall pool as (candidate_id, rrf_score) pairs."""
    query_text = build_jd_query_text(jd, jd_text)

    bm25 = load_bm25(artifacts_dir / "bm25.pkl")
    bm25_hits = search_bm25(bm25, query_text, top_k=bm25_k)
    bm25_ranking = [cid for cid, _ in bm25_hits]

    index, ids, _matrix = load_embedding_index(artifacts_dir)
    query_vec = encode_text(query_text)
    dense_hits = search_faiss(index, ids, query_vec, top_k=dense_k)
    dense_ranking = [cid for cid, _ in dense_hits]

    career_ranking = career_recall_ranking(artifacts_dir)
    weighted_lists: list[tuple[list[str], float]] = [
        (bm25_ranking, 1.0),
        (dense_ranking, 1.0),
    ]
    if career_ranking:
        weighted_lists.append((career_ranking, career_rrf_weight))

    fused = reciprocal_rank_fusion_weighted(weighted_lists, k=rrf_k)
    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:pool_size]
    logger.info(
        "Hybrid recall: bm25=%d dense=%d career=%d fused=%d pool=%d",
        len(bm25_ranking),
        len(dense_ranking),
        len(career_ranking),
        len(fused),
        len(ranked),
    )
    return ranked


def normalize_scores(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return {key: 1.0 for key in values}
    return {key: (val - lo) / (hi - lo) for key, val in values.items()}
