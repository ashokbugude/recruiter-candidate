"""Build and persist silver label parquet for LTR training."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from app.constants import REFERENCE_DATE
from app.labels.honeypots import detect_honeypot
from app.labels.tiers import TierResult, assign_heuristic_tier

logger = logging.getLogger(__name__)

LIST_SAMPLE_SIZE = 50


@dataclass(frozen=True)
class SilverLabelRow:
    candidate_id: str
    tier: int
    label_source: str
    confidence: float
    honeypot_score: float
    is_honeypot: bool
    is_trap: bool
    should_exclude: bool
    rules_hit: str
    reasons: str


def build_silver_labels(
    candidates: Iterator[dict[str, Any]],
    *,
    reference_date=None,
) -> tuple[list[SilverLabelRow], dict[str, list[str]]]:
    """
    Generate silver labels and curated candidate lists.

    Returns:
        rows: one SilverLabelRow per candidate
        lists: named ID lists for downstream validation
    """
    ref = reference_date or REFERENCE_DATE
    rows: list[SilverLabelRow] = []
    lists: dict[str, list[str]] = {
        "senior_ai_tier5": [],
        "strong_ml_tier4": [],
        "relevant_tier3": [],
        "transition_tier2": [],
        "weak_tier1": [],
        "honeypot_tier0": [],
        "search_recommendation_engineers": [],
        "keyword_stuffers": [],
        "template_trap_candidates": [],
        "manual_review_sample": [],
    }

    for candidate in candidates:
        honeypot = detect_honeypot(candidate, reference_date=ref)
        tier_result = assign_heuristic_tier(candidate, honeypot)
        row = _to_row(tier_result, honeypot)
        rows.append(row)
        _update_lists(candidate, tier_result, honeypot, lists)

    lists["manual_review_sample"] = _stratified_sample(rows, sample_size=100)

    return rows, lists


def write_silver_labels(
    rows: list[SilverLabelRow],
    lists: dict[str, list[str]],
    *,
    parquet_path: Path,
    lists_path: Path,
    list_sample_size: int = LIST_SAMPLE_SIZE,
) -> None:
    """Write labels parquet and compact candidate_lists.json (counts + samples)."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    lists_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pl.DataFrame(
        {
            "candidate_id": [r.candidate_id for r in rows],
            "tier": pl.Series([r.tier for r in rows], dtype=pl.Int8),
            "label_source": [r.label_source for r in rows],
            "confidence": pl.Series([r.confidence for r in rows], dtype=pl.Float64),
            "honeypot_score": pl.Series([r.honeypot_score for r in rows], dtype=pl.Float64),
            "is_honeypot": pl.Series([r.is_honeypot for r in rows], dtype=pl.Boolean),
            "is_trap": pl.Series([r.is_trap for r in rows], dtype=pl.Boolean),
            "should_exclude": pl.Series([r.should_exclude for r in rows], dtype=pl.Boolean),
            "rules_hit": [r.rules_hit for r in rows],
            "reasons": [r.reasons for r in rows],
        }
    )
    frame.write_parquet(parquet_path)
    logger.info("Wrote %d silver labels to %s", len(rows), parquet_path)

    summary = {
        "reference_date": REFERENCE_DATE.isoformat(),
        "counts": {
            name: len(ids)
            for name, ids in lists.items()
            if name != "manual_review_sample"
        },
        "manual_review_sample_count": len(lists["manual_review_sample"]),
        "tier_distribution": _tier_distribution(rows),
        "samples": _compact_list_samples(lists, sample_size=list_sample_size),
        "manual_review_sample": lists["manual_review_sample"],
    }
    lists_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote candidate lists to %s", lists_path)


def _compact_list_samples(lists: dict[str, list[str]], *, sample_size: int) -> dict[str, list[str]]:
    """Store sample IDs only — full lists live in parquet."""
    samples: dict[str, list[str]] = {}
    for name, ids in lists.items():
        if name == "manual_review_sample":
            continue
        samples[name] = sorted(ids)[:sample_size]
    return samples


def _to_row(tier: TierResult, honeypot) -> SilverLabelRow:
    return SilverLabelRow(
        candidate_id=tier.candidate_id,
        tier=tier.tier,
        label_source=tier.label_source,
        confidence=tier.confidence,
        honeypot_score=honeypot.score,
        is_honeypot=honeypot.is_honeypot,
        is_trap=honeypot.is_trap,
        should_exclude=honeypot.should_exclude,
        rules_hit="|".join(honeypot.rules_hit),
        reasons="|".join(tier.reasons),
    )


def _update_lists(
    candidate: dict[str, Any],
    tier: TierResult,
    honeypot,
    lists: dict[str, list[str]],
) -> None:
    cid = tier.candidate_id
    title = str((candidate.get("profile") or {}).get("current_title") or "").lower()

    if tier.tier == 5:
        lists["senior_ai_tier5"].append(cid)
    elif tier.tier == 4:
        lists["strong_ml_tier4"].append(cid)
    elif tier.tier == 3:
        lists["relevant_tier3"].append(cid)
    elif tier.tier == 2:
        lists["transition_tier2"].append(cid)
    elif tier.tier == 1:
        lists["weak_tier1"].append(cid)
    if tier.tier == 0:
        lists["honeypot_tier0"].append(cid)

    if any(k in title for k in ("search", "recommendation", "ranking", "retrieval")):
        lists["search_recommendation_engineers"].append(cid)

    if tier.label_source == "heuristic_trap" or "trap_title_ai_stuffing" in tier.reasons:
        lists["keyword_stuffers"].append(cid)

    if "recycled_templates" in tier.reasons or "R7" in honeypot.rules_hit:
        lists["template_trap_candidates"].append(cid)


def _tier_distribution(rows: list[SilverLabelRow]) -> dict[str, int]:
    dist: dict[str, int] = {str(i): 0 for i in range(6)}
    for row in rows:
        dist[str(row.tier)] += 1
    return dist


def _stratified_sample(rows: list[SilverLabelRow], sample_size: int) -> list[str]:
    """Pick a deterministic stratified sample for manual review."""
    by_tier: dict[int, list[str]] = {i: [] for i in range(6)}
    for row in rows:
        by_tier[row.tier].append(row.candidate_id)

    targets = {0: 8, 1: 12, 2: 15, 3: 20, 4: 25, 5: 20}
    selected: list[str] = []
    for tier, target in targets.items():
        pool = sorted(by_tier[tier])
        step = max(1, len(pool) // max(1, target))
        picked = pool[::step][:target]
        selected.extend(picked)

    if len(selected) < sample_size:
        extras = sorted(by_tier[4] + by_tier[3])
        for cid in extras:
            if cid not in selected:
                selected.append(cid)
            if len(selected) >= sample_size:
                break
    return sorted(selected)[:sample_size]
