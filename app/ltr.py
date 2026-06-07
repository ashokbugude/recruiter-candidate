"""LightGBM LambdaRank scorer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from app.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

LTR_MODEL_NAME = "ltr_model.lgb"
BEST_PARAMS_NAME = "best_params.json"
EXTRA_FEATURE_NAMES: tuple[str, ...] = ("gemini_tier_norm",)


def ltr_feature_columns() -> list[str]:
    return list(FEATURE_NAMES) + list(EXTRA_FEATURE_NAMES)


def prepare_ltr_matrix(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix and labels from feature parquet."""
    columns = ltr_feature_columns()
    work = frame.with_columns(
        (pl.col("gemini_tier").fill_null(-1).cast(pl.Float64) / 5.0).alias("gemini_tier_norm")
    )
    missing = [name for name in columns if name not in work.columns]
    if missing:
        raise ValueError(f"Missing LTR columns: {missing}")
    matrix = work.select(columns).to_numpy()
    labels = work["silver_tier"].fill_null(2).to_numpy()
    ids = work["candidate_id"].cast(pl.Utf8).to_list()
    return matrix, labels, ids


def load_ltr_model(path: Path) -> lgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"LTR model not found: {path}. Run: python scripts/train_ltr.py")
    return lgb.Booster(model_file=str(path))


def load_best_params(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def predict_ltr_scores(model: lgb.Booster, feature_matrix: np.ndarray) -> np.ndarray:
    return model.predict(feature_matrix)


def score_candidates_by_id(
    model: lgb.Booster,
    feature_lookup: dict[str, dict],
    candidate_ids: list[str],
) -> dict[str, float]:
    """Predict LTR scores for a subset of candidate IDs."""
    columns = ltr_feature_columns()
    rows: list[list[float]] = []
    valid_ids: list[str] = []
    for cid in candidate_ids:
        row = feature_lookup.get(cid)
        if row is None:
            continue
        gemini = float(row.get("gemini_tier", -1))
        if gemini < 0:
            gemini = 2.0
        values = [float(row.get(name, 0.0)) for name in FEATURE_NAMES]
        values.append(gemini / 5.0)
        rows.append(values)
        valid_ids.append(cid)
    if not rows:
        return {}
    scores = predict_ltr_scores(model, np.asarray(rows, dtype=np.float64))
    return dict(zip(valid_ids, scores.tolist(), strict=False))
