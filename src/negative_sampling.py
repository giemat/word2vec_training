"""Negative sampling utilities for word2vec."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def build_unigram_distribution(
    counts: dict[str, int],
    token_to_id: dict[str, int],
    power: float = 0.75,
) -> np.ndarray:
    """Build a smoothed unigram distribution over vocab ids.

    Args:
        counts: token -> count for all observed tokens in the corpus.
        token_to_id: token -> id mapping for the *kept* vocabulary
            (e.g. after applying a ``min_freq`` threshold).
        power: exponent for smoothing (standard word2vec uses 0.75).
            The distribution is defined as
            ``p(i) ∝ count(token_i) ** power`` over ``token_to_id``.

    Returns:
        probs: shape ``(vocab_size,)``, a 1D array over ids in
            ``[0, vocab_size)`` that sums to 1. Only tokens present in
            ``token_to_id`` contribute probability mass.
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
    """Sample negative ids, optionally avoiding given ids.

    Sampling is i.i.d. with replacement from the provided unigram
    distribution ``probs``. This means the same negative id can appear
    multiple times for a single (center, context) pair. When ``avoid``
    is given (typically containing the positive context id), any sample
    that falls into this set is resampled until no avoided ids remain.
    """
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
