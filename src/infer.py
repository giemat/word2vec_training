"""Inference-only entry point for trained word2vec embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval import nearest_neighbors
from io_utils import load_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query nearest neighbors from saved embeddings.")
    parser.add_argument("--load-dir", type=str, default="artifacts", help="Directory with embeddings.")
    parser.add_argument(
        "--queries",
        type=str,
        default="government,news,market",
        help="Comma-separated query tokens.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Nearest neighbors to show.")
    parser.add_argument(
        "--use-w-out",
        action="store_true",
        help="Use output embeddings instead of input embeddings.",
    )
    return parser.parse_args()


def _load_vocab(load_dir: str | Path) -> tuple[dict[str, int], list[str]]:
    vocab_path = Path(load_dir) / "token_to_id.json"
    token_to_id = json.loads(vocab_path.read_text(encoding="utf-8"))
    id_to_token = [None] * (max(token_to_id.values()) + 1)
    for token, idx in token_to_id.items():
        id_to_token[idx] = token
    return token_to_id, id_to_token


def main() -> None:
    args = _parse_args()
    model = load_model(args.load_dir)
    token_to_id, id_to_token = _load_vocab(args.load_dir)

    embeddings = model.w_out if args.use_w_out else model.w_in
    queries = [q for q in args.queries.split(",") if q]
    for query in queries:
        if query not in token_to_id:
            print(f"{query}: <missing>")
            continue
        token_id = token_to_id[query]
        neighbors = nearest_neighbors(embeddings, id_to_token, token_id, top_k=args.top_k)
        print(f"{query}: {neighbors}")


if __name__ == "__main__":
    main()
