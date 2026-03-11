"""Training loop for word2vec (skip-gram with negative sampling)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data import DatasetDict
from io_utils import save_model, save_vocab
from model import Word2VecModel
from negative_sampling import build_unigram_distribution, sample_negative


@dataclass
class TrainingConfig:
    embedding_dim: int = 100
    window_size: int = 2
    negative_samples: int = 5
    lr: float = 0.025
    epochs: int = 3
    seed: int = 13


def train_skipgram(
    dataset: DatasetDict,
    config: TrainingConfig,
    *,
    output_dir: str | None = None,
) -> Word2VecModel:
    """Train skip-gram with negative sampling on a prepared dataset."""
    rng = np.random.default_rng(config.seed)
    vocab_size = len(dataset["token_to_id"])
    model = Word2VecModel.initialize(vocab_size, config.embedding_dim, rng)

    probs = build_unigram_distribution(dataset["counts"], dataset["token_to_id"], power=0.75)
    pairs = dataset["pairs"]

    for epoch in range(config.epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        for center_id, context_id in pairs:
            neg_ids = sample_negative(rng, probs, config.negative_samples, avoid=[context_id])
            total_loss += model.train_step(center_id, context_id, neg_ids, config.lr)
        avg_loss = total_loss / max(1, len(pairs))
        print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f}")

    if output_dir is not None:
        save_model(model, output_dir)
        save_vocab(dataset["token_to_id"], output_dir)

    return model
