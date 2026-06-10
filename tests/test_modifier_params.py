"""Modifier params load/save and unknown-key warnings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.modifier_params import (
    ModifierParams,
    load_modifier_params,
    modifier_params_from_dict,
    save_modifier_params,
)


def test_unknown_json_keys_warn(caplog) -> None:
    caplog.set_level(logging.WARNING)
    params = modifier_params_from_dict({"tier5_boost": 1.1, "bogus_field": 99})
    assert params.tier5_boost == 1.1
    assert any("bogus_field" in r.message for r in caplog.records)


def test_round_trip_save_load(tmp_path: Path) -> None:
    path = tmp_path / "modifier_params.json"
    original = ModifierParams(tier5_boost=1.08, clone_top30_max=8)
    save_modifier_params(path, original, extra={"live_ndcg_at_10": 0.99})
    loaded = load_modifier_params(path)
    assert loaded.tier5_boost == 1.08
    assert loaded.clone_top30_max == 8
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["live_ndcg_at_10"] == 0.99
