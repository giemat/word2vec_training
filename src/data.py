"""Dataset utilities for word2vec training (NLTK-based)."""

from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import nltk
from nltk.corpus import brown

TOKEN_RE = re.compile(r"[a-z0-9']+")


class DatasetDict(TypedDict):
    token_ids: list[int]
    pairs: list[tuple[int, int]]
    token_to_id: dict[str, int]
    id_to_token: list[str]
    counts: dict[str, int]


def load_text(path: str | Path) -> str:
    """Load raw text from a UTF-8 file."""
    return Path(path).read_text(encoding="utf-8")


def load_brown_corpus(categories: Sequence[str] | None = None) -> list[list[str]]:
    """Load Brown corpus sentences as token lists."""
    return list(brown.sents(categories=categories))


def normalize_token(token: str) -> str:
    """Lowercase and strip non-word characters from a token."""
    token = token.lower()
    match = TOKEN_RE.search(token)
    return match.group(0) if match else ""


def tokenize_text(text: str) -> list[str]:
    """Tokenize raw text with NLTK, returning normalized tokens."""
    raw_tokens = nltk.word_tokenize(text)
    return [tok for tok in (normalize_token(tok) for tok in raw_tokens) if tok]


def tokenize_sentences(sentences: Iterable[Iterable[str]]) -> list[str]:
    """Normalize tokenized sentences from NLTK corpora."""
    normalized: list[str] = []
    for sentence in sentences:
        for token in sentence:
            cleaned = normalize_token(token)
            if cleaned:
                normalized.append(cleaned)
    return normalized


def build_vocab(tokens: Sequence[str], min_freq: int = 1) -> tuple[dict[str, int], list[str], Counter]:
    """Build vocab mappings and return token counts.

    Returns:
        token_to_id: mapping token -> integer id
        id_to_token: list where index is the token id
        counts: token counts for negative sampling
    """
    counts = Counter(tokens)
    kept = [token for token, count in counts.items() if count >= min_freq]
    kept.sort()
    token_to_id = {token: idx for idx, token in enumerate(kept)}
    id_to_token = list(kept)
    return token_to_id, id_to_token, counts


def tokens_to_ids(tokens: Iterable[str], token_to_id: dict[str, int]) -> list[int]:
    """Map tokens to ids, dropping tokens not in the vocab."""
    return [token_to_id[token] for token in tokens if token in token_to_id]


def generate_skipgram_pairs(token_ids: Sequence[int], window_size: int) -> list[tuple[int, int]]:
    """Generate (center_id, context_id) training pairs."""
    pairs: list[tuple[int, int]] = []
    for center_idx, center_id in enumerate(token_ids):
        start = max(0, center_idx - window_size)
        end = min(len(token_ids), center_idx + window_size + 1)
        for ctx_idx in range(start, end):
            if ctx_idx == center_idx:
                continue
            pairs.append((center_id, token_ids[ctx_idx]))
    return pairs


def build_dataset_from_tokens(
    tokens: Sequence[str],
    *,
    window_size: int = 2,
    min_freq: int = 1,
) -> DatasetDict:
    """Create a dataset dict for the training loop from token stream."""
    token_to_id, id_to_token, counts = build_vocab(tokens, min_freq=min_freq)
    token_ids = tokens_to_ids(tokens, token_to_id)
    pairs = generate_skipgram_pairs(token_ids, window_size=window_size)
    return {
        "token_ids": token_ids,
        "pairs": pairs,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "counts": dict(counts),
    }


def build_dataset_from_text(
    text: str,
    *,
    window_size: int = 2,
    min_freq: int = 1,
) -> DatasetDict:
    """Create a dataset dict from raw text using NLTK tokenization."""
    tokens = tokenize_text(text)
    return build_dataset_from_tokens(tokens, window_size=window_size, min_freq=min_freq)


def build_dataset_from_brown(
    *,
    categories: Sequence[str] | None = None,
    window_size: int = 2,
    min_freq: int = 1,
) -> DatasetDict:
    """Create a dataset dict from the Brown corpus."""
    sentences = load_brown_corpus(categories=categories)
    tokens = tokenize_sentences(sentences)
    return build_dataset_from_tokens(tokens, window_size=window_size, min_freq=min_freq)


def save_vocab(token_to_id: dict[str, int], path: str | Path) -> None:
    """Save vocab mapping as JSON."""
    Path(path).write_text(json.dumps(token_to_id, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    dataset = build_dataset_from_brown(categories=["news"], window_size=2, min_freq=2)
    print("vocab_size=", len(dataset["token_to_id"]))
    print("pairs=", dataset["pairs"][:10])


if __name__ == "__main__":
    main()
