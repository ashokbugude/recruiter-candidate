"""Phase 3/4 pipeline unit tests (sample data, no full 100K required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.behavioral import behavioral_multiplier
from app.config import PROJECT_ROOT
from app.feature_store import build_features_frame, write_features_parquet
from app.jd_requirements import build_heuristic_requirements
from app.ltr import prepare_ltr_matrix
from app.recall import reciprocal_rank_fusion
from app.traps import should_hard_exclude, trap_penalty


@pytest.fixture(scope="module")
def sample_candidates() -> list[dict]:
    path = PROJECT_ROOT / "challenge" / "sample_candidates.json"
    if not path.exists():
        pytest.skip("sample_candidates.json not available")
    return json.loads(path.read_text(encoding="utf-8"))[:20]


@pytest.fixture(scope="module")
def jd_requirements():
    jd_text = (PROJECT_ROOT / "artifacts" / "job_description.txt").read_text(encoding="utf-8")
    return build_heuristic_requirements(jd_text)


class TestRecallFusion:
    def test_rrf_combines_lists(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
        assert fused["a"] > fused["c"]
        assert fused["b"] >= fused["c"]


class TestTrapsAndBehavior:
    def test_trap_penalty_increases_for_excluded(self) -> None:
        assert trap_penalty({"should_exclude": True}) > trap_penalty({"should_exclude": False})

    def test_behavioral_multiplier_bounds(self, sample_candidates) -> None:
        value = behavioral_multiplier(sample_candidates[0])
        assert 0.55 <= value <= 1.15


class TestLTRTrainingMatrix:
    def test_prepare_ltr_matrix_shape(self, sample_candidates, jd_requirements, tmp_path) -> None:
        frame = build_features_frame(iter(sample_candidates), jd_requirements)
        frame = frame.with_columns(pl.lit(2).alias("silver_tier"), pl.lit(2).alias("gemini_tier"))
        out = tmp_path / "features.parquet"
        write_features_parquet(frame, out)
        matrix, labels, ids = prepare_ltr_matrix(pl.read_parquet(out))
        assert matrix.shape[0] == len(sample_candidates)
        assert matrix.shape[1] == 46
        assert len(labels) == len(ids)


class TestPhase4ModulesExist:
    @pytest.mark.parametrize(
        "filepath",
        [
            "app/recall.py",
            "app/traps.py",
            "app/behavioral.py",
            "app/ltr.py",
            "app/reranker.py",
            "app/pipeline.py",
            "app/reasoning.py",
            "scripts/train_ltr.py",
            "scripts/tune.py",
        ],
    )
    def test_files_exist(self, filepath: str) -> None:
        assert (PROJECT_ROOT / filepath).is_file()
