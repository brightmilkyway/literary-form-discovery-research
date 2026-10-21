"""Sentence-BERT semantic similarity (Reimers & Gurevych, 2019).

This is *not* a theory-free structure-discovery algorithm: the pretrained
encoder carries its own semantic priors.  We therefore (a) record the exact
model and its max sequence length, (b) check for truncation explicitly, and
(c) always compare the resulting matrix against the NCD and TF-IDF baselines.

Default model: ``all-MiniLM-L6-v2`` — small, fast on CPU, 256-token limit.
Units longer than the limit are truncated; the number of truncated units is
reported so the reader can judge the risk.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from common import Unit

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def encode_units(units: Sequence[Unit], model_name: str = DEFAULT_MODEL):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(42)
    model = SentenceTransformer(model_name)
    model.eval()
    max_len = int(model.max_seq_length)
    # Rough word->token check: flag units that likely exceed the limit.
    texts = [u.text for u in units]
    with torch.no_grad():
        emb = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    n_words = np.array([u.n_words for u in units])
    # ~1.35 tokens/word is a safe upper heuristic for English prose.
    likely_truncated = int(np.sum(n_words * 1.35 > max_len))
    return model, emb, {
        "model_name": model_name,
        "max_seq_length": max_len,
        "embedding_dim": int(emb.shape[1]),
        "n_units": len(units),
        "likely_truncated_units": likely_truncated,
        "n_words_max": int(n_words.max()),
    }


def similarity_from_embeddings(emb: np.ndarray) -> np.ndarray:
    S = emb @ emb.T
    np.fill_diagonal(S, 1.0)
    return np.clip(S, -1.0, 1.0)
