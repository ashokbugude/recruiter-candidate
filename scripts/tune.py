#!/usr/bin/env python3
"""Optuna hyperparameter tuning for LightGBM LambdaRank (proxy NDCG@10)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.ltr import (  # noqa: E402
    BEST_PARAMS_NAME,
    prepare_ltr_training_data,
    split_ltr_groups,
)

logger = logging.getLogger(__name__)


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    ranked_labels = labels[order][:k]
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ranked_labels))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_params(
    matrix_train: np.ndarray,
    labels_train: np.ndarray,
    group_train: list[int],
    matrix_val: np.ndarray,
    labels_val: np.ndarray,
    feature_names: list[str],
    params: dict,
) -> float:
    train_set = lgb.Dataset(
        matrix_train,
        label=labels_train,
        group=group_train,
        feature_name=feature_names,
    )
    num_round = int(params.get("num_boost_round", 200))
    train_params = {k: v for k, v in params.items() if k != "num_boost_round"}
    booster = lgb.train(train_params, train_set, num_boost_round=num_round)
    scores = booster.predict(matrix_val)
    return ndcg_at_k(labels_val, scores, k=10)


def tune_ltr(
    features_path: Path,
    output_path: Path,
    *,
    n_trials: int = 50,
) -> dict:
    frame = pl.read_parquet(features_path)
    matrix, labels, groups, feature_names = prepare_ltr_training_data(frame)
    train = split_ltr_groups(matrix, labels, groups, val_group_frac=0.2)
    matrix_train, labels_train, group_train, matrix_val, labels_val, _group_val = train
    logger.info(
        "Tuning on %d train / %d val rows (%d features)",
        matrix_train.shape[0],
        matrix_val.shape[0],
        len(feature_names),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 50],
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
            "num_boost_round": trial.suggest_int("num_boost_round", 100, 400),
        }
        return evaluate_params(
            matrix_train,
            labels_train,
            group_train,
            matrix_val,
            labels_val,
            feature_names,
            params,
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    report_path = output_path.with_name("tuning_report.json")
    report_path.write_text(
        json.dumps({"best_ndcg_at_10": study.best_value, "best_params": best, "n_trials": n_trials}, indent=2),
        encoding="utf-8",
    )
    logger.info("Best validation proxy NDCG@10=%.4f saved to %s", study.best_value, output_path)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune LightGBM LTR hyperparameters with Optuna.")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    features_path = (args.features or settings.artifact_path("candidate_features.parquet")).resolve()
    out_path = (args.out or settings.artifact_path(BEST_PARAMS_NAME)).resolve()

    if not features_path.exists():
        logger.error("Features not found: %s", features_path)
        return 1

    tune_ltr(features_path, out_path, n_trials=args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
