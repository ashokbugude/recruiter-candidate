"""Track preprocessed artifacts so Gemini is not re-run (and credits are not re-spent)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

MANIFEST_NAME = "preprocess_manifest.json"

# Files to commit under artifacts/ for full offline reuse (no Gemini, no re-encode).
TRACKED_ARTIFACTS: tuple[str, ...] = (
    "jd_requirements.json",
    "gemini_tiers.parquet",
    "labels_silver.parquet",
    "candidate_features.parquet",
    "bge_embeddings.npy",
    "faiss.index",
    "candidate_id_index.json",
    "bm25.pkl",
    "ltr_model.lgb",
)


def manifest_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / MANIFEST_NAME


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "exists": True,
        "bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def load_manifest(artifacts_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(artifacts_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_locked(artifacts_dir: Path) -> bool:
    manifest = load_manifest(artifacts_dir)
    return bool(manifest and manifest.get("locked"))


def missing_artifacts(artifacts_dir: Path, *, include_ltr: bool = False) -> list[str]:
    names = list(TRACKED_ARTIFACTS)
    if not include_ltr:
        names = [n for n in names if n != "ltr_model.lgb"]
    return [name for name in names if not (artifacts_dir / name).exists()]


def row_counts(artifacts_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, col in (
        ("candidate_features.parquet", "candidate_id"),
        ("gemini_tiers.parquet", "candidate_id"),
        ("labels_silver.parquet", "candidate_id"),
    ):
        path = artifacts_dir / name
        if path.exists():
            frame = pl.read_parquet(path)
            counts[name] = frame.height
    return counts


def build_manifest(artifacts_dir: Path, *, locked: bool = False) -> dict[str, Any]:
    files = {name: _file_meta(artifacts_dir / name) for name in TRACKED_ARTIFACTS}
    return {
        "version": 1,
        "locked": locked,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": row_counts(artifacts_dir),
        "files": files,
        "missing": missing_artifacts(artifacts_dir, include_ltr=True),
    }


def write_manifest(artifacts_dir: Path, *, locked: bool = False) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(artifacts_dir, locked=locked)
    path = manifest_path(artifacts_dir)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote artifact manifest to %s (locked=%s)", path, locked)
    return manifest


def assert_ready_for_preprocess_lock(artifacts_dir: Path) -> None:
    missing = missing_artifacts(artifacts_dir, include_ltr=False)
    if missing:
        raise FileNotFoundError(
            "Cannot lock artifacts — missing: "
            + ", ".join(missing)
            + ". Finish preprocess first (especially embeddings)."
        )
    counts = row_counts(artifacts_dir)
    for name, expected in (
        ("candidate_features.parquet", 100_000),
        ("gemini_tiers.parquet", 100_000),
        ("labels_silver.parquet", 100_000),
    ):
        if counts.get(name, 0) < expected:
            raise ValueError(f"{name} has {counts.get(name, 0)} rows; expected {expected}.")
