"""3-stage ranking pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from app.behavioral import behavioral_multiplier
from app.config import Settings
from app.jd_requirements import JDRequirements, load_or_build_jd_requirements
from app.ltr import load_ltr_model, score_candidates_by_id
from app.reasoning import build_reasoning
from app.recall import hybrid_recall, normalize_scores
from app.reranker import rerank_candidates
from app.traps import should_hard_exclude, trap_penalty

logger = logging.getLogger(__name__)


def load_feature_lookup(features_path: Path) -> dict[str, dict[str, Any]]:
    frame = pl.read_parquet(features_path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in frame.iter_rows(named=True):
        lookup[str(row["candidate_id"])] = row
    return lookup


def load_candidates_lookup(candidates_path: Path) -> dict[str, dict[str, Any]]:
    import json

    lookup: dict[str, dict[str, Any]] = {}
    with candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            lookup[str(candidate["candidate_id"])] = candidate
    return lookup


def assign_monotonic_scores(ranks: list[str]) -> dict[str, float]:
    """Map ranks to monotonically decreasing scores (rank 1 → 0.99)."""
    return {cid: max(0.01, 0.99 - (index * 0.008)) for index, cid in enumerate(ranks)}


class RankingPipeline:
    def __init__(self, settings: Settings, artifacts_dir: Path) -> None:
        self.settings = settings
        self.artifacts_dir = artifacts_dir.resolve()
        self.features_path = self.artifacts_dir / "candidate_features.parquet"
        self.ltr_path = self.artifacts_dir / "ltr_model.lgb"
        self.jd_path = settings.job_description_path

    def _load_jd(self) -> tuple[JDRequirements, str]:
        jd = load_or_build_jd_requirements(
            self.jd_path,
            self.artifacts_dir / "jd_requirements.json",
            settings=self.settings,
        )
        jd_text = self.jd_path.read_text(encoding="utf-8")
        return jd, jd_text

    def rank(
        self,
        candidates_path: Path,
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.top_k_output
        jd, jd_text = self._load_jd()
        feature_lookup = load_feature_lookup(self.features_path)
        candidate_lookup = load_candidates_lookup(candidates_path)

        recall_pool = hybrid_recall(
            jd,
            jd_text,
            self.artifacts_dir,
            bm25_k=self.settings.bm25_recall_k,
            dense_k=self.settings.dense_recall_k,
            pool_size=self.settings.recall_pool_size,
            rrf_k=self.settings.rrf_k,
        )
        rrf_scores = dict(recall_pool)

        pool_ids = [cid for cid, _ in recall_pool if cid in feature_lookup]
        pool_ids = [cid for cid in pool_ids if not should_hard_exclude(feature_lookup[cid])]

        model = load_ltr_model(self.ltr_path)
        ltr_raw = score_candidates_by_id(model, feature_lookup, pool_ids)

        stage2: dict[str, float] = {}
        for cid in pool_ids:
            row = feature_lookup[cid]
            candidate = candidate_lookup.get(cid, {})
            base = float(ltr_raw.get(cid, 0.0))
            penalty = trap_penalty(row)
            behavior = behavioral_multiplier(candidate, reference_date=self.settings.reference_date)
            stage2[cid] = (base - penalty) * behavior

        rerank_ids = sorted(stage2, key=lambda cid: (-stage2[cid], cid))[: self.settings.rerank_pool_size]
        rerank_candidates_list = [candidate_lookup[cid] for cid in rerank_ids if cid in candidate_lookup]
        jd_summary = f"{jd.role_title}. {' '.join(jd.must_have_skills[:10])}. YOE {jd.yoe_min}-{jd.yoe_max}."
        ce_raw = rerank_candidates(jd_summary, rerank_candidates_list)

        ce_norm = normalize_scores({cid: ce_raw.get(cid, 0.0) for cid in rerank_ids})
        ltr_norm = normalize_scores({cid: stage2.get(cid, 0.0) for cid in rerank_ids})
        rrf_norm = normalize_scores({cid: rrf_scores.get(cid, 0.0) for cid in rerank_ids})

        final: dict[str, float] = {}
        for cid in rerank_ids:
            final[cid] = (
                self.settings.rerank_ce_weight * ce_norm.get(cid, 0.0)
                + self.settings.rerank_ltr_weight * ltr_norm.get(cid, 0.0)
                + self.settings.rerank_rrf_weight * rrf_norm.get(cid, 0.0)
            )

        ranked_ids = sorted(final, key=lambda cid: (-final[cid], cid))[:top_k]
        score_map = assign_monotonic_scores(ranked_ids)

        results: list[dict[str, Any]] = []
        for rank_index, cid in enumerate(ranked_ids, start=1):
            candidate = candidate_lookup[cid]
            results.append(
                {
                    "candidate_id": cid,
                    "rank": rank_index,
                    "score": round(score_map[cid], 4),
                    "reasoning": build_reasoning(candidate, rank=rank_index),
                }
            )
        logger.info("Ranked top %d candidates from pool=%d rerank=%d", len(results), len(pool_ids), len(rerank_ids))
        return results
