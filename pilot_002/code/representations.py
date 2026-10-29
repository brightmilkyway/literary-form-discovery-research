"""Label-free representation adapters used by Pilot 002."""

from __future__ import annotations

import json
import os
import re
import zlib
from pathlib import Path

import numpy as np

import common as C

EMBED_DIR = C.DERIVED_DIR / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)


def _cosine(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    out = X @ X.T
    np.fill_diagonal(out, 1.0)
    return np.clip(out, -1.0, 1.0)


def char_tfidf(units: list[C.Unit]) -> tuple[np.ndarray, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                          min_df=1, sublinear_tf=True, norm="l2",
                          lowercase=True)
    X = vec.fit_transform([u.text for u in units])
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 1.0)
    return S, {"features": int(X.shape[1]), "analyzer": "char",
               "ngram_range": [3, 5]}


def word_tfidf(units: list[C.Unit]) -> tuple[np.ndarray, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                          sublinear_tf=True, norm="l2",
                          token_pattern=r"(?u)\b[A-Za-z][A-Za-z’']*\b")
    X = vec.fit_transform([u.text for u in units])
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 1.0)
    return S, {"features": int(X.shape[1]), "analyzer": "word",
               "ngram_range": [1, 2]}


def ncd_similarity(units: list[C.Unit]) -> tuple[np.ndarray, dict]:
    """Symmetric zlib NCD, reported as 1-distance.

    The two concatenation orders are averaged.  We retain the correlation with
    compressed-size difference as a diagnostic rather than silently declaring
    the practical compressor length-free.
    """
    blobs = [u.text.encode("utf-8") for u in units]
    sizes = np.array([len(zlib.compress(x, 9)) for x in blobs], dtype=float)
    n = len(units)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            cxy = len(zlib.compress(blobs[i] + blobs[j], 9))
            cyx = len(zlib.compress(blobs[j] + blobs[i], 9))
            ccat = (cxy + cyx) / 2.0
            value = ((ccat - min(sizes[i], sizes[j])) /
                     max(sizes[i], sizes[j]))
            D[i, j] = D[j, i] = value
    S = 1.0 - D
    np.fill_diagonal(S, 1.0)
    iu = np.triu_indices(n, k=1)
    diff = np.abs(sizes[iu[0]] - sizes[iu[1]])
    corr = float(np.corrcoef(D[iu], diff)[0, 1]) if diff.std() else float("nan")
    return S, {"compressor": "zlib-9", "order": "mean(xy,yx)",
               "corr_distance_compressed_size_difference": corr,
               "mean_compressed_size": float(sizes.mean())}


def zca_fit(X: np.ndarray, ridge: float = 1e-3):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, 0.0)
    scale = 1.0 / np.sqrt(values + ridge * max(values.mean(), 1e-12))
    W = (vectors * scale) @ vectors.T
    return mu, W, values


def apply_white(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    Z = (X - mu) @ W
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return Z


class NeuralWhitened:
    MODELS = {
        "minilm_white": "sentence-transformers/all-MiniLM-L6-v2",
        "bge_white": "BAAI/bge-small-en-v1.5",
    }

    def __init__(self, representation: str, ridge: float = 1e-3):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.representation = representation
        self.model_name = self.MODELS[representation]
        self.tag = representation.replace("_white", "")
        self.ridge = ridge
        # Both encoders were already used by Pilot 001.  Force an offline
        # cache lookup so a reproducible rerun never turns into a network call
        # merely to check for a newer model revision.
        self.model = SentenceTransformer(self.model_name,
                                         local_files_only=True)
        self.model.eval()
        self._fits: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    def _pool_encode(self, texts: list[str]) -> np.ndarray:
        """Paragraph-aware subchunk pooling, with duplicate-text caching."""
        import torch

        unique, inverse, seen = [], [], {}
        for text in texts:
            key = text
            if key not in seen:
                seen[key] = len(unique)
                unique.append(text)
            inverse.append(seen[key])

        subs, owners = [], []
        for owner, text in enumerate(unique):
            pieces = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
            buf, count = [], 0
            for piece in pieces or [text]:
                tokens = piece.split()
                chunks = [" ".join(tokens[i:i + 120])
                          for i in range(0, len(tokens), 120)] if len(tokens) > 120 else [piece]
                for chunk in chunks:
                    nw = len(chunk.split())
                    if buf and count + nw > 120:
                        subs.append("\n\n".join(buf))
                        owners.append(owner)
                        buf, count = [], 0
                    buf.append(chunk)
                    count += nw
            if buf:
                subs.append("\n\n".join(buf))
                owners.append(owner)

        with torch.no_grad():
            E = self.model.encode(subs, batch_size=64, convert_to_numpy=True,
                                  normalize_embeddings=True,
                                  show_progress_bar=False)
        pooled = np.zeros((len(unique), E.shape[1]), dtype=np.float64)
        for row, owner in zip(E, owners):
            pooled[owner] += row
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
        return pooled[np.asarray(inverse)]

    def _cached_embeddings(self, key: str, units: list[C.Unit]) -> np.ndarray:
        path = EMBED_DIR / f"{self.tag}_{key}.npz"
        fingerprint = json.dumps(
            [(u.word_start, u.word_end, len(u.text)) for u in units],
            separators=(",", ":"))
        if path.exists():
            data = np.load(path, allow_pickle=False)
            if str(data["fingerprint"].item()) == fingerprint:
                return data["embeddings"]
        E = self._pool_encode([u.text for u in units])
        np.savez_compressed(path, embeddings=E,
                            fingerprint=np.array(fingerprint))
        return E

    def _fit_scale(self, scale: str):
        if scale in self._fits:
            return self._fits[scale]
        donors = C.donor_units_by_scale(scale)
        E = self._cached_embeddings(f"donor_{scale}", donors)
        mu, W, values = zca_fit(E, self.ridge)
        positive = values[values > 1e-12]
        pr = float(positive.sum() ** 2 / (positive ** 2).sum())
        meta = {"model": self.model_name, "ridge": self.ridge,
                "fit_source": "other_14_dubliners_stories",
                "fit_scale": scale, "fit_units": len(donors),
                "fit_participation_ratio": pr}
        self._fits[scale] = (mu, W, meta)
        return self._fits[scale]

    def matrix(self, scheme: str, units: list[C.Unit]) -> tuple[np.ndarray, dict]:
        E = self._cached_embeddings(f"target_{scheme}", units)
        mu, W, meta = self._fit_scale(units[0].scale)
        Z = apply_white(E, mu, W)
        out = dict(meta)
        out["target_units"] = len(units)
        out["raw_mean_cosine"] = float(_upper_mean(_cosine(E)))
        out["white_mean_cosine"] = float(_upper_mean(_cosine(Z)))
        return _cosine(Z), out


def _upper_mean(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return float("nan")
    return float(matrix[np.triu_indices(len(matrix), k=1)].mean())


def lexical_matrix(name: str, units: list[C.Unit]) -> tuple[np.ndarray, dict]:
    if name == "char_tfidf":
        return char_tfidf(units)
    if name == "word_tfidf":
        return word_tfidf(units)
    if name == "ncd":
        return ncd_similarity(units)
    raise ValueError(name)

