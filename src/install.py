"""One-time NLTK download helper for this project.

Running this module downloads all corpora and tokenizers required by
the default training and tokenization pipeline:

- ``brown``: main training corpus used by ``data.build_dataset_from_brown``.
- ``punkt`` / ``punkt_tab``: sentence and word tokenizers used by NLTK.
- ``stopwords`` and ``wordnet``: not strictly required for core
  training, but useful for experimentation and future extensions.
"""

import nltk


def main() -> None:
    nltk.download("brown")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")


if __name__ == "__main__":
    main()
