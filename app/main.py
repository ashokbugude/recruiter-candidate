"""FastAPI sandbox — rank sample or uploaded candidates (CPU, no network)."""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.artifacts_check import ArtifactValidationError, validate_artifacts
from app.config import get_settings
from app.fusion import settings_with_fusion_params
from app.logging_setup import configure_logging
from app.pipeline import RankingPipeline

logger = logging.getLogger(__name__)

_artifacts_ok = False
_artifact_warnings: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifacts_ok, _artifact_warnings
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        _artifact_warnings = validate_artifacts(settings.artifacts_dir.resolve())
        for msg in _artifact_warnings:
            logger.warning("%s", msg)
        _artifacts_ok = True
    except ArtifactValidationError as exc:
        logger.error("Artifact validation failed: %s", exc)
        _artifacts_ok = False
    yield


app = FastAPI(
    title="Redrob Candidate Ranker",
    description="Hybrid recall + LTR + cross-encoder ranker (offline artifacts, CPU-only).",
    version="1.0.0",
    lifespan=lifespan,
)


class RankRequest(BaseModel):
    candidates_path: str | None = Field(default=None, description="Server path to candidates.jsonl")
    top_k: int = Field(default=100, ge=1, le=100)


@app.get("/health")
def health() -> dict:
    if not _artifacts_ok:
        raise HTTPException(status_code=503, detail="Artifacts missing or invalid")
    return {"status": "ok", "warnings": _artifact_warnings}


@app.post("/rank")
def rank_candidates(body: RankRequest | None = None) -> dict:
    if not _artifacts_ok:
        raise HTTPException(status_code=503, detail="Artifacts missing or invalid")

    settings = get_settings()
    artifacts_dir = settings.artifacts_dir.resolve()
    settings = settings_with_fusion_params(settings, artifacts_dir)

    candidates_path = settings.candidates_path
    if body and body.candidates_path:
        candidates_path = Path(body.candidates_path)

    if not candidates_path.exists():
        raise HTTPException(status_code=400, detail=f"Candidates not found: {candidates_path}")

    pipeline = RankingPipeline(settings, artifacts_dir)
    top_k = body.top_k if body else 100
    results = pipeline.rank(candidates_path, top_k=top_k)
    return {"count": len(results), "results": results}


@app.post("/rank/upload")
async def rank_upload(file: UploadFile = File(...)) -> dict:
    """Rank candidates from uploaded JSONL (max sample size for sandbox)."""
    if not _artifacts_ok:
        raise HTTPException(status_code=503, detail="Artifacts missing or invalid")

    settings = get_settings()
    artifacts_dir = settings.artifacts_dir.resolve()
    settings = settings_with_fusion_params(settings, artifacts_dir)

    raw = await file.read()
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Empty JSONL upload")
    if len(lines) > 500:
        raise HTTPException(status_code=400, detail="Sandbox limit: 500 candidates per upload")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(lines))
        tmp_path = Path(tmp.name)

    try:
        pipeline = RankingPipeline(settings, artifacts_dir)
        results = pipeline.rank(tmp_path, top_k=min(100, len(lines)))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"count": len(results), "results": results}


@app.get("/rank/sample")
def rank_sample() -> dict:
    """Rank bundled sample_candidates.json (<=100 rows)."""
    if not _artifacts_ok:
        raise HTTPException(status_code=503, detail="Artifacts missing or invalid")

    settings = get_settings()
    sample_path = settings.sample_candidates_path
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="sample_candidates.json missing")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for row in json.loads(sample_path.read_text(encoding="utf-8")):
            tmp.write(json.dumps(row) + "\n")
        tmp_path = Path(tmp.name)

    try:
        settings = settings_with_fusion_params(settings, settings.artifacts_dir)
        pipeline = RankingPipeline(settings, settings.artifacts_dir)
        results = pipeline.rank(tmp_path, top_k=100)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"count": len(results), "results": results}
