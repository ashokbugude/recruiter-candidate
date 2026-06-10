"""Career recall score loading tests."""

from __future__ import annotations

import json
from pathlib import Path

from app.career_recall import career_recall_ranking, load_career_scores


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    load_career_scores.cache_clear()
    assert load_career_scores(str(tmp_path)) == {}
    assert career_recall_ranking(tmp_path) == []


def test_valid_file_sorted_descending(tmp_path: Path) -> None:
    load_career_scores.cache_clear()
    payload = {"scores": {"a": 0.5, "b": 0.9, "c": 0.7}}
    (tmp_path / "career_recall_scores.json").write_text(json.dumps(payload), encoding="utf-8")
    scores = load_career_scores(str(tmp_path))
    assert scores["b"] == 0.9
    ranked = career_recall_ranking(tmp_path, top_k=2)
    assert ranked == ["b", "c"]


def test_lru_cache_path_normalization(tmp_path: Path) -> None:
    load_career_scores.cache_clear()
    payload = {"scores": {"x": 1.0}}
    (tmp_path / "career_recall_scores.json").write_text(json.dumps(payload), encoding="utf-8")
    a = load_career_scores(str(tmp_path))
    b = load_career_scores(str(tmp_path.resolve()))
    assert a == b
