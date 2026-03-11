"""Evaluation utilities for trained embeddings."""

from __future__ import annotations

import numpy as np


def cosine_similarity(vecs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of vecs and a query vector."""
    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    return vecs_norm @ query_norm


def nearest_neighbors(
    embeddings: np.ndarray,
    id_to_token: list[str],
    token_id: int,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k nearest neighbors for a token id."""
    sims = cosine_similarity(embeddings, embeddings[token_id])
    sims[token_id] = -1.0
    top_ids = np.argsort(-sims)[:top_k]
    return [(id_to_token[idx], float(sims[idx])) for idx in top_ids]
