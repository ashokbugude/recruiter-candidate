#!/usr/bin/env python3
"""Train LightGBM LambdaRank model on silver labels."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.ltr import BEST_PARAMS_NAME, LTR_MODEL_NAME, ltr_feature_columns, prepare_ltr_matrix  # noqa: E402

logger = logging.getLogger(__name__)


def train_ltr(
    features_path: Path,
    output_path: Path,
    *,
    params_path: Path | None = None,
) -> lgb.Booster:
    frame = pl.read_parquet(features_path)
    if "silver_tier" not in frame.columns:
        raise ValueError("Feature parquet must include silver_tier column")

    matrix, labels, _ids = prepare_ltr_matrix(frame)
    group = [matrix.shape[0]]

    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.2,
        "verbosity": -1,
    }
    num_round = 300
    if params_path and params_path.exists():
        tuned = json.loads(params_path.read_text(encoding="utf-8"))
        num_round = int(tuned.get("num_boost_round", 300))
        default_params.update({k: v for k, v in tuned.items() if k != "num_boost_round"})
        logger.info("Loaded tuned params from %s", params_path)

    train_set = lgb.Dataset(
        matrix,
        label=labels,
        group=group,
        feature_name=ltr_feature_columns(),
        free_raw_data=False,
    )
    booster = lgb.train(default_params, train_set, num_boost_round=num_round)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    logger.info("Saved LTR model to %s (%d rows, %d features)", output_path, matrix.shape[0], matrix.shape[1])
    return booster


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank on candidate features.")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--params", type=Path, default=None, help="Optional best_params.json from tune.py")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    features_path = (args.features or settings.artifact_path("candidate_features.parquet")).resolve()
    out_path = (args.out or settings.artifact_path(LTR_MODEL_NAME)).resolve()
    params_path = (args.params or settings.artifact_path(BEST_PARAMS_NAME)).resolve()

    if not features_path.exists():
        logger.error("Features not found: %s — run preprocess first", features_path)
        return 1

    train_ltr(features_path, out_path, params_path=params_path if params_path.exists() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
