"""Small smoke test for dataset + training pipeline."""

from __future__ import annotations

from data import build_dataset_from_text
from eval import nearest_neighbors
from train import TrainingConfig, train_skipgram


def main() -> None:
    text = """
    The quick brown fox jumps over the lazy dog.
    The fox is quick and the dog is lazy.
    """
    dataset = build_dataset_from_text(text, window_size=2, min_freq=1)
    config = TrainingConfig(embedding_dim=20, window_size=2, negative_samples=3, lr=0.05, epochs=5)
    model = train_skipgram(dataset, config)

    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    token_id = token_to_id["fox"]
    print(nearest_neighbors(model.w_in, id_to_token, token_id, top_k=3))


if __name__ == "__main__":
    main()
