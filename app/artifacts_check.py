"""Artifact validation for rank.py and FastAPI sandbox."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.artifact_names import CAREER_SCORES, FUSION_PARAMS, MODIFIER_PARAMS, RANK_OPTIONAL_ARTIFACTS, RANK_REQUIRED_ARTIFACTS

logger = logging.getLogger(__name__)


class ArtifactValidationError(Exception):
    """Raised when required artifacts are missing or invalid."""


def validate_fusion_weights(params: dict) -> None:
    ce = float(params.get("rerank_ce_weight", 0))
    ltr = float(params.get("rerank_ltr_weight", 0))
    rrf = float(params.get("rerank_rrf_weight", 0))
    total = ce + ltr + rrf
    if abs(total - 1.0) > 0.01:
        raise ArtifactValidationError(
            f"Fusion weights must sum to ~1.0 (got ce={ce:.4f} ltr={ltr:.4f} rrf={rrf:.4f} total={total:.4f})"
        )


def validate_artifacts(artifacts_dir: Path, *, strict: bool = True) -> list[str]:
    """Validate artifacts for ranking. Returns warnings; raises on missing required files."""
    artifacts_dir = artifacts_dir.resolve()
    missing = [name for name in RANK_REQUIRED_ARTIFACTS if not (artifacts_dir / name).exists()]
    if missing:
        raise ArtifactValidationError(
            "Missing artifacts: "
            + ", ".join(missing)
            + ". Run: python scripts/preprocess.py && python scripts/train_ltr.py"
        )

    warnings: list[str] = []
    fusion_path = artifacts_dir / FUSION_PARAMS
    if fusion_path.exists():
        try:
            validate_fusion_weights(json.loads(fusion_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(f"Invalid {FUSION_PARAMS}: {exc}") from exc

    modifier_path = artifacts_dir / MODIFIER_PARAMS
    if modifier_path.exists():
        try:
            json.loads(modifier_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(f"Invalid {MODIFIER_PARAMS}: {exc}") from exc

    career_path = artifacts_dir / CAREER_SCORES
    if career_path.exists():
        payload = json.loads(career_path.read_text(encoding="utf-8"))
        scores = payload.get("scores", {})
        if len(scores) < 1000:
            warnings.append(f"{CAREER_SCORES} has only {len(scores)} entries (expected >= 1000)")
    elif strict:
        warnings.append(f"{CAREER_SCORES} missing — career recall degraded to BM25+dense only")

    for name in RANK_OPTIONAL_ARTIFACTS:
        if not (artifacts_dir / name).exists():
            logger.debug("Optional artifact missing: %s", name)

    return warnings
