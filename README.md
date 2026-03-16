# word2vec_training
JetBrains Hallucination Detection Task

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The project implements a standard skip-gram with negative sampling variant on top of NLTK data, with a small but complete training and evaluation pipeline.

## Quick Start

### 1) Install dependencies and NLTK data

```bash
python -m pip install nltk
python src/install.py  # downloads Brown corpus and tokenizers
```

### 2) Train on the Brown corpus (default)

```bash
python src/run.py
```

This will:

- build a dataset from the Brown `news` category by default,
- train skip-gram word2vec with negative sampling for a few epochs, and
- print nearest neighbors for a small set of query tokens.

## CLI Usage (summary)

The main entry point is `src/run.py`. Common patterns:

- **Train on specific Brown categories**

  ```bash
  python src/run.py --categories news,editorial,government
  ```

- **Train on a local text file**

  ```bash
  python src/run.py --text-path /path/to/text.txt
  ```

- **Control key hyperparameters**

  ```bash
  python src/run.py \
    --window-size 3 \
    --min-freq 2 \
    --embedding-dim 100 \
    --negative-samples 8 \
    --lr 0.02 \
    --epochs 5 \
    --seed 42
  ```

- **Save artifacts (embeddings + vocab)**

  ```bash
  python src/run.py --output-dir artifacts
  ```

  This writes:

  - `artifacts/w_in.npy`
  - `artifacts/w_out.npy`
  - `artifacts/token_to_id.json`

- **Query nearest neighbors after training**

  ```bash
  python src/run.py --queries government,market,bank --top-k 10
  ```

## Inference Only (using saved embeddings)

After training and saving artifacts, you can query nearest neighbors without retraining using `src/infer.py`:

```bash
python src/infer.py --load-dir artifacts --queries government,news --top-k 5
```

By default this uses the input embeddings (`w_in.npy`). To instead use the output embeddings (`w_out.npy`), pass:

```bash
python src/infer.py --load-dir artifacts --queries government --top-k 5 --use-w-out
```
