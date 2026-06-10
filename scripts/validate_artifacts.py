#!/usr/bin/env python3
"""Validate extracted preprocess and rank-ready artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.artifact_names import CAREER_SCORES, FUSION_PARAMS, MODIFIER_PARAMS  # noqa: E402
from app.artifacts_check import validate_fusion_weights  # noqa: E402
from app.artifacts_manifest import assert_ready_for_preprocess_lock, missing_artifacts  # noqa: E402
from app.bm25_index import load_bm25  # noqa: E402
from app.features import FEATURE_NAMES  # noqa: E402

REQUIRED = (
    "jd_requirements.json",
    "gemini_tiers.parquet",
    "labels_silver.parquet",
    "candidate_features.parquet",
    "bge_embeddings.npy",
    "faiss.index",
    "candidate_id_index.json",
    "bm25.pkl",
)

RANK_READY = (
    "ltr_model.lgb",
    FUSION_PARAMS,
    MODIFIER_PARAMS,
    CAREER_SCORES,
)


def validate(artifacts_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED:
        path = artifacts_dir / name
        if not path.exists():
            errors.append(f"Missing: {name}")
        else:
            print(f"OK  {name} ({path.stat().st_size / 1_048_576:.2f} MB)")

    if errors:
        return errors, warnings

    try:
        assert_ready_for_preprocess_lock(artifacts_dir)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(str(exc))
        return errors, warnings

    jd = json.loads((artifacts_dir / "jd_requirements.json").read_text(encoding="utf-8"))
    print(
        f"JD source: {jd.get('source')} | role: {jd.get('role_title')} | "
        f"must_have_skills: {len(jd.get('must_have_skills', []))}"
    )
    if jd.get("source") != "gemini_pro":
        warnings.append(f"JD source is '{jd.get('source')}' — gemini_pro is preferred for accuracy")

    feat = pl.read_parquet(artifacts_dir / "candidate_features.parquet")
    missing_feats = [c for c in FEATURE_NAMES if c not in feat.columns]
    if missing_feats:
        errors.append(f"Feature parquet missing columns: {missing_feats}")
    print(f"Feature columns: {len(feat.columns)}")

    emb = np.load(artifacts_dir / "bge_embeddings.npy", mmap_mode="r")
    print(f"Embeddings shape: {emb.shape}")
    if emb.shape != (100_000, 384):
        errors.append(f"bge_embeddings.npy shape {emb.shape}, expected (100000, 384)")

    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    print(f"FAISS ntotal: {index.ntotal}")
    if index.ntotal != 100_000:
        errors.append(f"faiss.index ntotal {index.ntotal}, expected 100000")

    id_index = json.loads((artifacts_dir / "candidate_id_index.json").read_text(encoding="utf-8"))
    print(f"candidate_id_index entries: {len(id_index)}")
    if len(id_index) != 100_000:
        errors.append(f"candidate_id_index has {len(id_index)} entries, expected 100000")

    g = pl.read_parquet(artifacts_dir / "gemini_tiers.parquet")
    src = g.group_by("label_source").len().sort("len", descending=True)
    print("Label sources:", dict(zip(src["label_source"].to_list(), src["len"].to_list(), strict=False)))
    if g.filter(pl.col("label_source") == "gemini_flash").height == 0:
        warnings.append("No gemini_flash labels (silver_fallback only) — OK if GEMINI_LABELS=False")

    silver = pl.read_parquet(artifacts_dir / "labels_silver.parquet")
    tier_dist = silver.group_by("tier").len().sort("tier")
    print("Silver tier counts:", dict(zip(tier_dist["tier"].to_list(), tier_dist["len"].to_list(), strict=False)))

    bm25 = load_bm25(artifacts_dir / "bm25.pkl")
    print(f"BM25 corpus_size: {bm25.corpus_size}")
    if bm25.corpus_size != 100_000:
        errors.append(f"bm25 corpus_size {bm25.corpus_size}, expected 100000")

    for name in RANK_READY:
        path = artifacts_dir / name
        if not path.exists():
            errors.append(f"Rank-ready artifact missing: {name} — run train_ltr / tune_modifiers / build_career_recall_scores")
        else:
            print(f"OK  {name}")

    fusion_path = artifacts_dir / FUSION_PARAMS
    if fusion_path.exists():
        try:
            validate_fusion_weights(json.loads(fusion_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Invalid {FUSION_PARAMS}: {exc}")

    modifier_path = artifacts_dir / MODIFIER_PARAMS
    if modifier_path.exists():
        try:
            json.loads(modifier_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid {MODIFIER_PARAMS}: {exc}")

    career_path = artifacts_dir / CAREER_SCORES
    if career_path.exists():
        payload = json.loads(career_path.read_text(encoding="utf-8"))
        n_scores = len(payload.get("scores", {}))
        print(f"Career recall scores: {n_scores}")
        if n_scores < 1000:
            errors.append(f"{CAREER_SCORES} has only {n_scores} entries (expected >= 1000)")

    remaining = missing_artifacts(artifacts_dir, include_ltr=True)
    if remaining:
        warnings.append(f"Optional missing: {', '.join(remaining)}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate preprocess artifacts.")
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args()
    artifacts_dir = args.artifacts.resolve()

    print(f"Validating {artifacts_dir}\n")
    errors, warnings = validate(artifacts_dir)

    if warnings:
        print("\nWARNINGS:")
        for item in warnings:
            print(f"  - {item}")
    if errors:
        print("\nERRORS:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
