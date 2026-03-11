"""Negative sampling utilities for word2vec."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def build_unigram_distribution(
    counts: dict[str, int],
    token_to_id: dict[str, int],
    power: float = 0.75,
) -> np.ndarray:
    """Build a unigram distribution over vocab ids.

    Args:
        counts: token -> count
        token_to_id: token -> id
        power: exponent for smoothing (standard word2vec uses 0.75)

    Returns:
        probs: shape (vocab_size,), sums to 1
    """
    vocab_size = len(token_to_id)
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for token, idx in token_to_id.items():
        freqs[idx] = counts.get(token, 0)
    freqs = np.power(freqs, power)
    total = freqs.sum()
    if total == 0:
        raise ValueError("Unigram distribution is empty; check counts and vocab.")
    return freqs / total


def sample_negative(
    rng: np.random.Generator,
    probs: np.ndarray,
    k: int,
    avoid: Iterable[int] | None = None,
) -> np.ndarray:
    """Sample negative ids, optionally avoiding given ids."""
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    neg_ids = rng.choice(len(probs), size=k, replace=True, p=probs).astype(np.int64)
    if avoid is None:
        return neg_ids
    avoid_set = set(avoid)
    if not avoid_set:
        return neg_ids
    mask = np.isin(neg_ids, list(avoid_set))
    while mask.any():
        resampled = rng.choice(len(probs), size=mask.sum(), replace=True, p=probs)
        neg_ids[mask] = resampled
        mask = np.isin(neg_ids, list(avoid_set))
    return neg_ids
