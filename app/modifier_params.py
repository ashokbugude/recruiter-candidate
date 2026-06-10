"""Tunable post-fusion modifier weights (loaded from artifacts/modifier_params.json)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.artifact_names import MODIFIER_PARAMS as MODIFIER_PARAMS_NAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModifierParams:
    tier5_boost: float = 1.06
    low_avail_mult: float = 0.35
    low_rr_steep_mult: float = 0.55
    low_rr_moderate_mult: float = 0.75
    clone_low_rr_mult: float = 0.40
    clone_tier5_mult: float = 0.92
    research_no_prod_mult: float = 0.45
    research_with_prod_mult: float = 0.91
    plain_language_career_boost: float = 1.076
    plain_language_career_threshold: float = 0.58
    cv_speech_mult: float = 0.42
    outside_india_mult: float = 0.38
    excess_yoe_mult: float = 0.88
    stretched_scientist_mult: float = 0.72
    consulting_mult: float = 0.80
    product_boost: float = 1.05
    primary_hub_boost: float = 1.04
    preferred_hub_boost: float = 1.03
    clone_top30_max: int = 10
    top_availability_cap: int = 30


DEFAULT_MODIFIER_PARAMS = ModifierParams()


def modifier_params_from_dict(data: dict[str, Any]) -> ModifierParams:
    allowed = {f.name for f in ModifierParams.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = [k for k in data if k not in allowed and k not in (
        "proxy_objective", "ndcg_at_10", "tier5_in_top_10", "tier5_in_top_30",
        "tier5_in_top_100", "n_trials", "live_ndcg_at_10", "cache_ndcg_at_10",
        "live_tier5_in_top_10", "live_tier5_in_top_30", "live_tier5_in_top_100",
        "live_template_clones_top_30", "live_ndcg_at_30",
    )]
    if unknown:
        logger.warning("Ignoring unknown keys in modifier_params.json: %s", ", ".join(unknown))
    return ModifierParams(**{k: v for k, v in data.items() if k in allowed})


def load_modifier_params(path: Path) -> ModifierParams:
    if not path.exists():
        return DEFAULT_MODIFIER_PARAMS
    return modifier_params_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_modifier_params(path: Path, params: ModifierParams, extra: dict[str, Any] | None = None) -> None:
    payload = asdict(params)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
