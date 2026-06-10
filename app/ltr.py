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
LAMBDARANK_MAX_GROUP_SIZE = 10_000
DEFAULT_LTR_GROUP_SIZE = 1_000
DEFAULT_LTR_TRAIN_SEED = 42


def should_use_gemini_tier_feature(frame: pl.DataFrame) -> bool:
    """Skip gemini_tier_norm when it duplicates silver_tier (silver_fallback labels)."""
    if "gemini_tier" not in frame.columns or "silver_tier" not in frame.columns:
        return True
    same = frame.filter(pl.col("gemini_tier") == pl.col("silver_tier")).height
    ratio = same / max(frame.height, 1)
    return ratio < 0.95


def ltr_feature_columns(*, include_gemini_tier: bool = True) -> list[str]:
    columns = list(FEATURE_NAMES)
    if include_gemini_tier:
        columns.extend(EXTRA_FEATURE_NAMES)
    return columns


def prepare_ltr_matrix(
    frame: pl.DataFrame,
    *,
    include_gemini_tier: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build feature matrix and labels from feature parquet."""
    use_gemini = (
        should_use_gemini_tier_feature(frame) if include_gemini_tier is None else include_gemini_tier
    )
    columns = ltr_feature_columns(include_gemini_tier=use_gemini)
    work = frame.with_columns(
        (pl.col("gemini_tier").fill_null(-1).cast(pl.Float64) / 5.0).alias("gemini_tier_norm")
    )
    missing = [name for name in columns if name not in work.columns]
    if missing:
        raise ValueError(f"Missing LTR columns: {missing}")
    matrix = work.select(columns).to_numpy()
    labels = work["silver_tier"].fill_null(2).to_numpy()
    ids = work["candidate_id"].cast(pl.Utf8).to_list()
    return matrix, labels, ids, columns


def build_ltr_groups(n_rows: int, *, group_size: int = DEFAULT_LTR_GROUP_SIZE) -> list[int]:
    """Split rows into LambdaRank query groups (each must be <= 10K)."""
    if n_rows <= 0:
        return []
    if group_size <= 0 or group_size > LAMBDARANK_MAX_GROUP_SIZE:
        raise ValueError(f"group_size must be 1..{LAMBDARANK_MAX_GROUP_SIZE}")
    groups: list[int] = []
    remaining = n_rows
    while remaining > 0:
        size = min(group_size, remaining)
        groups.append(size)
        remaining -= size
    return groups


def prepare_ltr_training_data(
    frame: pl.DataFrame,
    *,
    group_size: int = DEFAULT_LTR_GROUP_SIZE,
    seed: int = DEFAULT_LTR_TRAIN_SEED,
    include_gemini_tier: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Shuffle candidates and build matrix/labels with valid LambdaRank groups."""
    matrix, labels, _ids, columns = prepare_ltr_matrix(frame, include_gemini_tier=include_gemini_tier)
    if not should_use_gemini_tier_feature(frame) and include_gemini_tier is not False:
        logger.warning(
            "gemini_tier_norm excluded from LTR — identical to silver_tier (silver_fallback labels)"
        )
    order = np.random.default_rng(seed).permutation(matrix.shape[0])
    matrix = matrix[order]
    labels = labels[order]
    groups = build_ltr_groups(matrix.shape[0], group_size=group_size)
    logger.info(
        "LTR training data: %d rows in %d groups (group_size<=%d, %d features)",
        matrix.shape[0],
        len(groups),
        group_size,
        matrix.shape[1],
    )
    return matrix, labels, groups, columns


def split_ltr_groups(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: list[int],
    *,
    val_group_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray, list[int]]:
    """Hold out trailing query groups for validation."""
    if not groups:
        raise ValueError("groups must not be empty")
    n_val = max(1, int(len(groups) * val_group_frac))
    train_groups = groups[:-n_val]
    val_groups = groups[-n_val:]
    train_rows = sum(train_groups)
    matrix_train, labels_train = matrix[:train_rows], labels[:train_rows]
    matrix_val, labels_val = matrix[train_rows:], labels[train_rows:]
    return matrix_train, labels_train, train_groups, matrix_val, labels_val, val_groups


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
    *,
    include_gemini_tier: bool | None = None,
) -> dict[str, float]:
    """Predict LTR scores for a subset of candidate IDs."""
    if include_gemini_tier is None:
        include_gemini_tier = model.num_feature() == len(FEATURE_NAMES) + len(EXTRA_FEATURE_NAMES)
    columns = ltr_feature_columns(include_gemini_tier=include_gemini_tier)
    rows: list[list[float]] = []
    valid_ids: list[str] = []
    for cid in candidate_ids:
        row = feature_lookup.get(cid)
        if row is None:
            continue
        values = [float(row.get(name, 0.0)) for name in FEATURE_NAMES]
        if include_gemini_tier:
            gemini = float(row.get("gemini_tier", -1))
            if gemini < 0:
                gemini = 2.0
            values.append(gemini / 5.0)
        rows.append(values)
        valid_ids.append(cid)
    if not rows:
        return {}
    scores = predict_ltr_scores(model, np.asarray(rows, dtype=np.float64))
    return dict(zip(valid_ids, scores.tolist(), strict=False))
