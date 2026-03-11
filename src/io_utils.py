"""I/O helpers for word2vec artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import Word2VecModel


def save_model(model: Word2VecModel, output_dir: str | Path) -> None:
    """Save embedding matrices to a directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "w_in.npy", model.w_in)
    np.save(out_dir / "w_out.npy", model.w_out)


def load_model(output_dir: str | Path) -> Word2VecModel:
    """Load embedding matrices from a directory."""
    out_dir = Path(output_dir)
    w_in = np.load(out_dir / "w_in.npy")
    w_out = np.load(out_dir / "w_out.npy")
    return Word2VecModel(w_in=w_in, w_out=w_out)


def save_vocab(token_to_id: dict[str, int], output_dir: str | Path) -> None:
    """Save vocab mapping to a directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "token_to_id.json").write_text(
        json.dumps(token_to_id, indent=2, sort_keys=True),
        encoding="utf-8",
    )
