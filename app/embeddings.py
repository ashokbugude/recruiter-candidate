"""BGE bi-encoder embeddings and FAISS index (offline preprocessing)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


def build_profile_text(candidate: dict[str, Any]) -> str:
    from app.features import candidate_profile_text

    return candidate_profile_text(candidate)


def resolve_embedding_device(device: str | None = None) -> str:
    """Pick CUDA when available (e.g. Colab GPU), else CPU."""
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def default_batch_size(device: str) -> int:
    return 256 if device == "cuda" else 64


def encode_candidates(
    candidates: list[dict[str, Any]],
    *,
    model_name: str = BGE_MODEL_NAME,
    batch_size: int | None = None,
    show_progress: bool = True,
    device: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Encode candidates to normalized embedding matrix."""
    from sentence_transformers import SentenceTransformer

    resolved_device = resolve_embedding_device(device)
    effective_batch = batch_size or default_batch_size(resolved_device)
    logger.info("Encoding on device=%s batch_size=%d", resolved_device, effective_batch)

    model = SentenceTransformer(model_name, device=resolved_device)
    texts = [build_profile_text(c) for c in candidates]
    ids = [str(c["candidate_id"]) for c in candidates]
    embeddings = model.encode(
        texts,
        batch_size=effective_batch,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        device=resolved_device,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    return matrix, ids


def save_embeddings(matrix: np.ndarray, ids: list[str], *, npy_path: Path, index_path: Path) -> None:
    """Persist embedding matrix, FAISS index, and ID mapping."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, matrix)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)

    faiss.write_index(index, str(index_path))
    mapping_path = index_path.with_name("candidate_id_index.json")
    mapping_path.write_text(json.dumps(ids, indent=0), encoding="utf-8")
    logger.info(
        "Saved embeddings %s shape=%s, FAISS index %s (%d vectors)",
        npy_path,
        matrix.shape,
        index_path,
        index.ntotal,
    )


def encode_text(text: str, *, model_name: str = BGE_MODEL_NAME) -> np.ndarray:
    """Encode a single query string (e.g. JD text) for FAISS search."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vector = model.encode([text], normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)[0]


def search_faiss(
    index: faiss.Index,
    ids: list[str],
    query_vector: np.ndarray,
    *,
    top_k: int = 3000,
) -> list[tuple[str, float]]:
    """Return top-k (candidate_id, score) from FAISS inner-product index."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(query, min(top_k, index.ntotal))
    results: list[tuple[str, float]] = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0:
            continue
        results.append((ids[int(idx)], float(score)))
    return results


def load_embedding_index(
    artifacts_dir: Path,
) -> tuple[faiss.Index, list[str], np.ndarray]:
    """Load FAISS index, ID list, and embedding matrix."""
    index_path = artifacts_dir / "faiss.index"
    npy_path = artifacts_dir / "bge_embeddings.npy"
    mapping_path = artifacts_dir / "candidate_id_index.json"

    if not index_path.exists() or not npy_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            f"Embedding artifacts missing in {artifacts_dir}. Run: python scripts/preprocess.py --step embeddings"
        )

    index = faiss.read_index(str(index_path))
    matrix = np.load(npy_path)
    ids = json.loads(mapping_path.read_text(encoding="utf-8"))
    return index, ids, matrix
