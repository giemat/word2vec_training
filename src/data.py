"""Dataset utilities for word2vec training (NLTK-based).

The core public surface of this module is :class:`DatasetDict` plus the
``build_dataset_from_*`` helpers, which prepare everything the training
loop needs:

- a stream of integer token ids,
- skip-gram training pairs,
- vocabulary mappings, and
- token counts for negative sampling.
"""

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
    """Container for all dataset pieces consumed by the training loop.

    Keys:
        token_ids: Linearized sequence of token ids produced from the
            original token stream *after* applying ``min_freq``. Only
            tokens present in ``token_to_id`` appear here.
        pairs: List of ``(center_id, context_id)`` skip-gram training
            pairs built from ``token_ids`` using a symmetric window.
        token_to_id: Mapping from token string to integer id. The id
            range is ``[0, vocab_size)`` and matches the rows of the
            embedding matrices.
        id_to_token: List of tokens where ``id_to_token[i]`` is the
            token string for id ``i``.
        counts: Raw token counts over the original token stream,
            including tokens that may be dropped by ``min_freq``.
            Negative sampling only uses counts for tokens present in
            ``token_to_id`` via their assigned ids.
    """

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
        token_to_id: Mapping token -> integer id for tokens whose
            frequency is at least ``min_freq``.
        id_to_token: List where index is the token id; contains only
            tokens kept in the vocab.
        counts: Raw token counts over *all* tokens, before the
            ``min_freq`` filter. This is later combined with
            ``token_to_id`` when building the unigram distribution used
            for negative sampling.
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
    """Generate (center_id, context_id) training pairs.

    For each position ``i`` in ``token_ids``, this iterates over a
    symmetric context window ``[i - window_size, i + window_size]``,
    clipped to the sequence bounds, and emits pairs for all positions
    except ``i`` itself.
    """
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
    """Create a :class:`DatasetDict` from a pre-tokenized stream.

    The same ``min_freq`` threshold is applied both to the embedding
    vocabulary and to the ``token_ids`` sequence used for generating
    skip-gram pairs. The returned ``counts`` still reflect the full
    token stream and are later restricted to the kept vocab when
    computing negative sampling probabilities.
    """
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
    """Create a :class:`DatasetDict` from raw text.

    This first tokenizes the input with NLTK, normalizes tokens, and
    then defers to :func:`build_dataset_from_tokens`. See that function
    for details on ``min_freq`` and window semantics.
    """
    tokens = tokenize_text(text)
    return build_dataset_from_tokens(tokens, window_size=window_size, min_freq=min_freq)


def build_dataset_from_brown(
    *,
    categories: Sequence[str] | None = None,
    window_size: int = 2,
    min_freq: int = 1,
) -> DatasetDict:
    """Create a :class:`DatasetDict` from the Brown corpus.

    Sentences from the selected Brown categories are normalized and
    flattened into a single token stream before building vocab and
    skip-gram pairs. See :func:`build_dataset_from_tokens` for
    ``min_freq`` and window semantics.
    """
    sentences = load_brown_corpus(categories=categories)
    tokens = tokenize_sentences(sentences)
    return build_dataset_from_tokens(tokens, window_size=window_size, min_freq=min_freq)


def save_vocab(token_to_id: dict[str, int], path: str | Path) -> None:
    """Save vocab mapping as a single JSON file.

    This is a low-level helper that writes directly to ``path`` and is
    useful in small experiments or notebooks. For the main training
    pipeline, prefer :func:`io_utils.save_vocab`, which saves the vocab
    alongside model weights in a dedicated output directory.
    """
    Path(path).write_text(json.dumps(token_to_id, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    dataset = build_dataset_from_brown(categories=["news"], window_size=2, min_freq=2)
    print("vocab_size=", len(dataset["token_to_id"]))
    print("pairs=", dataset["pairs"][:10])


if __name__ == "__main__":
    main()
