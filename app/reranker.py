"""Cross-encoder reranker — BGE reranker-base."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Deterministic CE scoring for tune vs live parity (see README).
np.random.seed(42)
try:
    import torch

    torch.manual_seed(42)
except ImportError:
    pass

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _load_reranker(model_name: str = RERANKER_MODEL_NAME):
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder reranker: %s", model_name)
    return CrossEncoder(model_name, max_length=512)


def build_pair_text(jd_summary: str, candidate: dict[str, Any]) -> tuple[str, str]:
    from app.features import candidate_profile_text

    jd_text = jd_summary[:1500]
    profile = candidate_profile_text(candidate)[:1500]
    return jd_text, profile


def rerank_candidates(
    jd_summary: str,
    candidates: list[dict[str, Any]],
    *,
    batch_size: int = 16,
) -> dict[str, float]:
    """Score candidate profiles against JD summary; higher is better."""
    if not candidates:
        return {}
    model = _load_reranker()
    pairs = [build_pair_text(jd_summary, candidate) for candidate in candidates]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return {
        str(candidate["candidate_id"]): float(score)
        for candidate, score in zip(candidates, scores, strict=False)
    }
