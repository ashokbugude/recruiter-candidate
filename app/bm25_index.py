"""BM25 sparse retrieval index (offline preprocessing)."""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.progress import ProgressTracker

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9+#./-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Artifacts:
    bm25: BM25Okapi
    candidate_ids: list[str]
    corpus_size: int


def build_bm25_index(candidates: list[dict[str, Any]]) -> BM25Artifacts:
    from app.features import candidate_profile_text

    total = len(candidates)
    log_every = max(total // 20, 1) if total else 1
    progress = ProgressTracker(
        logger, label="BM25 tokenization", total=total, log_every=log_every, unit="candidates"
    )

    ids: list[str] = []
    corpus: list[list[str]] = []
    for candidate in candidates:
        ids.append(str(candidate["candidate_id"]))
        corpus.append(tokenize(candidate_profile_text(candidate)))
        progress.tick()

    progress.finish(message="tokenization complete")
    logger.info("BM25: building index for %d documents...", total)
    bm25 = BM25Okapi(corpus)
    logger.info("BM25: index ready (%d docs)", total)
    return BM25Artifacts(bm25=bm25, candidate_ids=ids, corpus_size=len(ids))


def save_bm25(artifacts: BM25Artifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": artifacts.bm25,
        "candidate_ids": artifacts.candidate_ids,
        "corpus_size": artifacts.corpus_size,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({"corpus_size": artifacts.corpus_size, "candidate_count": len(artifacts.candidate_ids)}),
        encoding="utf-8",
    )
    logger.info("Saved BM25 index to %s (%d docs)", path, artifacts.corpus_size)


def load_bm25(path: Path) -> BM25Artifacts:
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found: {path}. Run preprocess --step bm25")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return BM25Artifacts(
        bm25=payload["bm25"],
        candidate_ids=payload["candidate_ids"],
        corpus_size=payload["corpus_size"],
    )


def search_bm25(
    artifacts: BM25Artifacts,
    query: str,
    *,
    top_k: int = 3000,
) -> list[tuple[str, float]]:
    """Return top-k (candidate_id, score) pairs."""
    tokens = tokenize(query)
    scores = artifacts.bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(artifacts.candidate_ids[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]
