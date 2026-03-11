"""Word2vec skip-gram with negative sampling (NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Word2VecModel:
    """Embeddings and update logic for skip-gram with negative sampling."""

    w_in: np.ndarray
    w_out: np.ndarray

    @classmethod
    def initialize(
        cls,
        vocab_size: int,
        embedding_dim: int,
        rng: np.random.Generator,
    ) -> "Word2VecModel":
        scale = 0.5 / embedding_dim
        w_in = rng.uniform(-scale, scale, size=(vocab_size, embedding_dim)).astype(np.float32)
        w_out = rng.uniform(-scale, scale, size=(vocab_size, embedding_dim)).astype(np.float32)
        return cls(w_in=w_in, w_out=w_out)

    def train_step(
        self,
        center_id: int,
        context_id: int,
        negative_ids: np.ndarray,
        lr: float,
    ) -> float:
        """Update embeddings for a single (center, context) pair.

        Returns:
            loss: float
        """
        v_in = self.w_in[center_id]
        v_out_pos = self.w_out[context_id]

        s_pos = float(np.dot(v_in, v_out_pos))
        sig_pos = _sigmoid_scalar(s_pos)
        loss = -np.log(sig_pos + 1e-12)

        grad_pos = sig_pos - 1.0
        grad_in = grad_pos * v_out_pos
        self.w_out[context_id] -= lr * grad_pos * v_in

        if negative_ids.size:
            v_out_neg = self.w_out[negative_ids]
            s_neg = np.dot(v_out_neg, v_in)
            sig_neg = _sigmoid_array(s_neg)
            loss -= np.sum(np.log(1.0 - sig_neg + 1e-12))
            grad_neg = sig_neg
            grad_in += np.sum(grad_neg[:, None] * v_out_neg, axis=0)
            self.w_out[negative_ids] -= lr * (grad_neg[:, None] * v_in[None, :])

        self.w_in[center_id] -= lr * grad_in
        return float(loss)


def _sigmoid_scalar(x: float) -> float:
    x = float(np.clip(x, -10.0, 10.0))
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-x))
