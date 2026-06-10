#!/usr/bin/env python3
"""Tune fusion weights + post-fusion modifiers; verify with live RankingPipeline."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifact_names import FUSION_PARAMS, MODIFIER_PARAMS, TUNING_REPORT_MODIFIERS  # noqa: E402
from app.career_recall import load_career_scores  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.fusion import (  # noqa: E402
    build_fusion_cache,
    evaluate_full_ranking,
    settings_with_fusion_params,
)
from app.feature_store import load_silver_tiers  # noqa: E402
from app.fusion import rank_from_fusion_cache  # noqa: E402
from app.jd_requirements import load_or_build_jd_requirements  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.modifier_params import ModifierParams, save_modifier_params  # noqa: E402
from app.pipeline import RankingPipeline  # noqa: E402
from app.rank_data import load_candidates_lookup, load_feature_lookup  # noqa: E402

logger = logging.getLogger(__name__)

POOL_SIZES = (500, 600, 700, 800)
CAP_SIZES = (28, 30, 32, 35)
CLONE_CAPS = (6, 8, 10, 12)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def verify_live_pipeline(
    settings,
    artifacts_dir: Path,
    candidates_path: Path,
    silver: dict[str, int],
    candidate_lookup: dict,
) -> dict:
    """Run live RankingPipeline and evaluate same metrics as cache path."""
    settings = settings_with_fusion_params(settings, artifacts_dir)
    pipeline = RankingPipeline(settings, artifacts_dir)
    results = pipeline.rank(candidates_path, top_k=100)
    ranked_ids = [r["candidate_id"] for r in results]
    return evaluate_full_ranking(
        ranked_ids,
        silver,
        candidate_lookup,
        reference_date=settings.reference_date,
    )


def tune_modifiers(
    settings,
    artifacts_dir: Path,
    candidates_path: Path,
    *,
    n_trials: int = 120,
    max_rerank_pool: int = 800,
    verify_live: bool = True,
) -> dict:
    silver = load_silver_tiers(artifacts_dir / "labels_silver.parquet")
    jd = load_or_build_jd_requirements(
        settings.job_description_path,
        artifacts_dir / "jd_requirements.json",
        settings=settings,
    )
    feature_lookup = load_feature_lookup(artifacts_dir / "candidate_features.parquet")
    candidate_lookup = load_candidates_lookup(candidates_path)
    career_scores = load_career_scores(str(artifacts_dir))
    cache = build_fusion_cache(settings, artifacts_dir, candidates_path, max_rerank_pool=max_rerank_pool)

    def objective(trial: optuna.Trial) -> float:
        ce_w = trial.suggest_float("rerank_ce_weight", 0.30, 0.55)
        ltr_w = trial.suggest_float("rerank_ltr_weight", 0.45, 0.52)
        rrf_w = 1.0 - ce_w - ltr_w
        if rrf_w < 0.05 or rrf_w > 0.25:
            raise optuna.TrialPruned()

        mod = ModifierParams(
            tier5_boost=trial.suggest_float("tier5_boost", 1.02, 1.12),
            low_avail_mult=trial.suggest_float("low_avail_mult", 0.25, 0.45),
            low_rr_steep_mult=trial.suggest_float("low_rr_steep_mult", 0.45, 0.65),
            clone_tier5_mult=trial.suggest_float("clone_tier5_mult", 0.88, 0.98),
            research_with_prod_mult=trial.suggest_float("research_with_prod_mult", 0.85, 0.98),
            plain_language_career_boost=trial.suggest_float("plain_language_career_boost", 1.02, 1.12),
            plain_language_career_threshold=trial.suggest_float("plain_language_career_threshold", 0.50, 0.65),
            stretched_scientist_mult=trial.suggest_float("stretched_scientist_mult", 0.55, 0.75),
            cv_speech_mult=trial.suggest_float("cv_speech_mult", 0.30, 0.50),
            primary_hub_boost=trial.suggest_float("primary_hub_boost", 1.00, 1.10),
            preferred_hub_boost=trial.suggest_float("preferred_hub_boost", 1.00, 1.06),
            clone_top30_max=trial.suggest_categorical("clone_top30_max", list(CLONE_CAPS)),
            top_availability_cap=trial.suggest_categorical("top_availability_cap", list(CAP_SIZES)),
        )
        pool_size = trial.suggest_categorical("rerank_pool_size", list(POOL_SIZES))
        career_w = trial.suggest_float("career_rrf_weight", 0.50, 0.80)

        trial_settings = settings.model_copy(update={"career_rrf_weight": career_w})
        ranked = rank_from_fusion_cache(
            cache,
            jd=jd,
            candidate_lookup=candidate_lookup,
            feature_lookup=feature_lookup,
            reference_date=trial_settings.reference_date,
            rerank_pool_size=pool_size,
            top_k=100,
            ce_weight=ce_w,
            ltr_weight=ltr_w,
            rrf_weight=rrf_w,
            modifier_params=mod,
            career_scores=career_scores,
        )
        metrics = evaluate_full_ranking(
            ranked,
            silver,
            candidate_lookup,
            reference_date=trial_settings.reference_date,
        )

        if metrics["honeypots_in_top_10"] > 0 or metrics["low_avail_top_30"] > 0:
            raise optuna.TrialPruned()

        trial.set_user_attr("ndcg_at_10", metrics["ndcg_at_10"])
        trial.set_user_attr("tier5_in_top_10", metrics["tier5_in_top_10"])
        trial.set_user_attr("tier5_in_top_30", metrics["tier5_in_top_30"])

        return (
            metrics["ndcg_at_10"]
            + 0.02 * metrics["tier5_in_top_10"]
            + 0.008 * metrics["tier5_in_top_30"]
            + 0.003 * metrics["tier5_in_top_100"]
            - 0.01 * metrics.get("template_clones_top_30", 0)
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["rerank_rrf_weight"] = round(1.0 - best["rerank_ce_weight"] - best["rerank_ltr_weight"], 4)
    mod = ModifierParams(
        tier5_boost=best["tier5_boost"],
        low_avail_mult=best["low_avail_mult"],
        low_rr_steep_mult=best["low_rr_steep_mult"],
        clone_tier5_mult=best["clone_tier5_mult"],
        research_with_prod_mult=best["research_with_prod_mult"],
        plain_language_career_boost=best["plain_language_career_boost"],
        plain_language_career_threshold=best["plain_language_career_threshold"],
        stretched_scientist_mult=best["stretched_scientist_mult"],
        cv_speech_mult=best["cv_speech_mult"],
        primary_hub_boost=best["primary_hub_boost"],
        preferred_hub_boost=best["preferred_hub_boost"],
        clone_top30_max=best["clone_top30_max"],
        top_availability_cap=best["top_availability_cap"],
    )

    cache_metrics = evaluate_full_ranking(
        rank_from_fusion_cache(
            cache,
            jd=jd,
            candidate_lookup=candidate_lookup,
            feature_lookup=feature_lookup,
            reference_date=settings.reference_date,
            rerank_pool_size=best["rerank_pool_size"],
            top_k=100,
            ce_weight=best["rerank_ce_weight"],
            ltr_weight=best["rerank_ltr_weight"],
            rrf_weight=best["rerank_rrf_weight"],
            modifier_params=mod,
            career_scores=career_scores,
        ),
        silver,
        candidate_lookup,
        reference_date=settings.reference_date,
    )

    fusion_payload = {
        "rerank_ce_weight": best["rerank_ce_weight"],
        "rerank_ltr_weight": best["rerank_ltr_weight"],
        "rerank_rrf_weight": best["rerank_rrf_weight"],
        "rerank_pool_size": best["rerank_pool_size"],
        "career_rrf_weight": best["career_rrf_weight"],
        "proxy_objective": study.best_value,
        "cache_ndcg_at_10": cache_metrics["ndcg_at_10"],
        "tier5_in_top_10": cache_metrics["tier5_in_top_10"],
        "tier5_in_top_30": cache_metrics["tier5_in_top_30"],
        "tier5_in_top_100": cache_metrics["tier5_in_top_100"],
        "low_avail_top_30": cache_metrics["low_avail_top_30"],
        "n_trials": n_trials,
        "tuned_with_modifiers": True,
    }

    save_modifier_params(artifacts_dir / MODIFIER_PARAMS, mod)
    (artifacts_dir / FUSION_PARAMS).write_text(json.dumps(fusion_payload, indent=2), encoding="utf-8")

    live_metrics: dict | None = None
    if verify_live:
        settings = settings.model_copy(update={"career_rrf_weight": best["career_rrf_weight"]})
        live_metrics = verify_live_pipeline(settings, artifacts_dir, candidates_path, silver, candidate_lookup)
        fusion_payload["live_ndcg_at_10"] = live_metrics["ndcg_at_10"]
        fusion_payload["live_tier5_in_top_10"] = live_metrics["tier5_in_top_10"]
        fusion_payload["live_tier5_in_top_30"] = live_metrics["tier5_in_top_30"]
        fusion_payload["live_tier5_in_top_100"] = live_metrics["tier5_in_top_100"]
        (artifacts_dir / FUSION_PARAMS).write_text(json.dumps(fusion_payload, indent=2), encoding="utf-8")
        save_modifier_params(
            artifacts_dir / MODIFIER_PARAMS,
            mod,
            extra={
                "live_ndcg_at_10": live_metrics["ndcg_at_10"],
                "live_tier5_in_top_10": live_metrics["tier5_in_top_10"],
                "live_tier5_in_top_30": live_metrics["tier5_in_top_30"],
                "live_tier5_in_top_100": live_metrics["tier5_in_top_100"],
            },
        )

    top10_ids = (
        [r["candidate_id"] for r in RankingPipeline(settings_with_fusion_params(settings, artifacts_dir), artifacts_dir).rank(candidates_path, top_k=10)]
        if verify_live
        else rank_from_fusion_cache(
            cache, jd=jd, candidate_lookup=candidate_lookup, feature_lookup=feature_lookup,
            reference_date=settings.reference_date, rerank_pool_size=best["rerank_pool_size"], top_k=10,
            ce_weight=best["rerank_ce_weight"], ltr_weight=best["rerank_ltr_weight"],
            rrf_weight=best["rerank_rrf_weight"], modifier_params=mod, career_scores=career_scores,
        )
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "best_params": {**best, **{k: v for k, v in fusion_payload.items() if k.startswith("rerank")}},
        "cache_metrics": cache_metrics,
        "live_metrics": live_metrics,
        "top10_fingerprint": top10_ids,
    }
    (artifacts_dir / TUNING_REPORT_MODIFIERS).write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "Cache: NDCG@10=%.4f tier5@30=%.0f | Live: %s",
        cache_metrics["ndcg_at_10"],
        cache_metrics["tier5_in_top_30"],
        f"NDCG@10={live_metrics['ndcg_at_10']:.4f}" if live_metrics else "skipped",
    )
    return fusion_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune fusion + modifier params (full ranking path).")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--max-pool", type=int, default=800)
    parser.add_argument(
        "--verify-live",
        action="store_true",
        default=True,
        help="Re-run live RankingPipeline after tuning (default: on).",
    )
    parser.add_argument("--no-verify-live", action="store_true", help="Skip live pipeline verification.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    artifacts_dir = (args.artifacts or settings.artifacts_dir).resolve()
    candidates_path = (args.candidates or settings.candidates_path).resolve()

    if not (artifacts_dir / "ltr_model.lgb").exists():
        logger.error("Train LTR first: python scripts/train_ltr.py")
        return 1

    tune_modifiers(
        settings,
        artifacts_dir,
        candidates_path,
        n_trials=args.trials,
        max_rerank_pool=args.max_pool,
        verify_live=not args.no_verify_live,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
