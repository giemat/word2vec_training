"""Simple entry point to train and inspect word2vec embeddings."""

from __future__ import annotations

import argparse

from data import build_dataset_from_brown, build_dataset_from_text, load_text
from eval import nearest_neighbors
from train import TrainingConfig, train_skipgram


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train word2vec with NLTK data.")
    parser.add_argument("--text-path", type=str, default=None, help="Path to a UTF-8 text file.")
    parser.add_argument(
        "--categories",
        type=str,
        default="news",
        help="Comma-separated Brown categories (ignored if --text-path set).",
    )
    parser.add_argument("--window-size", type=int, default=2, help="Context window size.")
    parser.add_argument("--min-freq", type=int, default=5, help="Minimum token frequency.")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Embedding dimension.")
    parser.add_argument("--negative-samples", type=int, default=5, help="Number of negatives.")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Where to save outputs.")
    parser.add_argument(
        "--queries",
        type=str,
        default="government,news,market",
        help="Comma-separated query tokens for neighbors.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Nearest neighbors to show.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.text_path:
        text = load_text(args.text_path)
        dataset = build_dataset_from_text(text, window_size=args.window_size, min_freq=args.min_freq)
    else:
        categories = [c for c in (args.categories.split(",") if args.categories else []) if c]
        dataset = build_dataset_from_brown(
            categories=categories or None,
            window_size=args.window_size,
            min_freq=args.min_freq,
        )

    config = TrainingConfig(
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        negative_samples=args.negative_samples,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    output_dir = args.output_dir or None
    model = train_skipgram(dataset, config, output_dir=output_dir)

    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    queries = [q for q in args.queries.split(",") if q]
    for query in queries:
        if query not in token_to_id:
            continue
        neighbors = nearest_neighbors(model.w_in, id_to_token, token_to_id[query], top_k=args.top_k)
        print(f"{query}: {neighbors}")


if __name__ == "__main__":
    main()
