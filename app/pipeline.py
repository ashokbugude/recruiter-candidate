"""3-stage ranking pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.artifact_names import MODIFIER_PARAMS
from app.behavioral import behavioral_multiplier
from app.career_recall import load_career_scores
from app.config import Settings
from app.fusion import fuse_scores
from app.jd_requirements import JDRequirements, load_or_build_jd_requirements
from app.ltr import load_ltr_model, score_candidates_by_id
from app.modifier_params import load_modifier_params
from app.rank_data import load_candidates_lookup, load_feature_lookup
from app.ranking_core import rank_with_modifiers
from app.ranking_modifiers import fusion_to_submission_scores
from app.reasoning import build_reasoning
from app.recall import hybrid_recall, normalize_scores
from app.reranker import rerank_candidates
from app.traps import research_title_penalty, should_hard_exclude, trap_penalty

logger = logging.getLogger(__name__)


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
        modifier_params = load_modifier_params(self.artifacts_dir / MODIFIER_PARAMS)
        career_scores = load_career_scores(str(self.artifacts_dir))
        feature_lookup = load_feature_lookup(self.features_path)
        candidate_lookup = load_candidates_lookup(candidates_path)

        # Small candidate files (sandbox upload ≤ top_k): score all provided IDs.
        # Full candidates.jsonl uses global hybrid recall (same path as team_sarva_automata.csv).
        if len(candidate_lookup) <= self.settings.top_k_output:
            pool_ids = [
                cid
                for cid in candidate_lookup
                if cid in feature_lookup and not should_hard_exclude(feature_lookup[cid])
            ]
            pool_ids.sort()
            rrf_scores = {cid: float(career_scores.get(cid, 0.0)) for cid in pool_ids}
        else:
            recall_pool = hybrid_recall(
                jd,
                jd_text,
                self.artifacts_dir,
                bm25_k=self.settings.bm25_recall_k,
                dense_k=self.settings.dense_recall_k,
                pool_size=self.settings.recall_pool_size,
                rrf_k=self.settings.rrf_k,
                career_rrf_weight=self.settings.career_rrf_weight,
            )
            rrf_scores = dict(recall_pool)
            pool_ids = [cid for cid, _ in recall_pool if cid in feature_lookup and cid in candidate_lookup]
            pool_ids = [cid for cid in pool_ids if not should_hard_exclude(feature_lookup[cid])]

        model = load_ltr_model(self.ltr_path)
        ltr_raw = score_candidates_by_id(model, feature_lookup, pool_ids)

        stage2: dict[str, float] = {}
        for cid in pool_ids:
            row = feature_lookup[cid]
            candidate = candidate_lookup.get(cid, {})
            base = float(ltr_raw.get(cid, 0.0))
            penalty = trap_penalty(row) + research_title_penalty(row, candidate)
            behavior = behavioral_multiplier(candidate, reference_date=self.settings.reference_date)
            stage2[cid] = (base - penalty) * behavior

        rerank_ids = sorted(stage2, key=lambda cid: (-stage2[cid], cid))[: self.settings.rerank_pool_size]
        rerank_candidates_list = [candidate_lookup[cid] for cid in rerank_ids if cid in candidate_lookup]
        jd_summary = f"{jd.role_title}. {' '.join(jd.must_have_skills[:10])}. YOE {jd.yoe_min}-{jd.yoe_max}."
        ce_raw = rerank_candidates(jd_summary, rerank_candidates_list)

        ce_norm = normalize_scores({cid: ce_raw.get(cid, 0.0) for cid in rerank_ids})
        ltr_norm = normalize_scores({cid: stage2.get(cid, 0.0) for cid in rerank_ids})
        rrf_norm = normalize_scores({cid: rrf_scores.get(cid, 0.0) for cid in rerank_ids})

        final = fuse_scores(
            rerank_ids,
            ce_norm=ce_norm,
            ltr_norm=ltr_norm,
            rrf_norm=rrf_norm,
            ce_weight=self.settings.rerank_ce_weight,
            ltr_weight=self.settings.rerank_ltr_weight,
            rrf_weight=self.settings.rerank_rrf_weight,
        )

        ranked_ids = rank_with_modifiers(
            rerank_ids,
            final,
            jd=jd,
            candidate_lookup=candidate_lookup,
            feature_lookup=feature_lookup,
            modifier_params=modifier_params,
            career_scores=career_scores,
            reference_date=self.settings.reference_date,
            top_k=top_k,
        )

        # Recompute adjusted scores for submission score mapping
        from app.ranking_core import apply_modifiers_to_scores

        adjusted = apply_modifiers_to_scores(
            final,
            rerank_ids,
            jd=jd,
            candidate_lookup=candidate_lookup,
            feature_lookup=feature_lookup,
            modifier_params=modifier_params,
            career_scores=career_scores,
            reference_date=self.settings.reference_date,
        )
        score_map = fusion_to_submission_scores(ranked_ids, adjusted)

        results: list[dict[str, Any]] = []
        for rank_index, cid in enumerate(ranked_ids, start=1):
            candidate = candidate_lookup[cid]
            results.append(
                {
                    "candidate_id": cid,
                    "rank": rank_index,
                    "score": score_map.get(cid, round(0.99 - (rank_index - 1) * 0.008, 4)),
                    "reasoning": build_reasoning(candidate, rank=rank_index, jd=jd),
                }
            )
        logger.info("Ranked top %d candidates from pool=%d rerank=%d", len(results), len(pool_ids), len(rerank_ids))
        return results
