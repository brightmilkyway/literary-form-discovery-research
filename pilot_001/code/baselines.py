"""Simple baselines used to judge whether the heavier methods add anything.

* **TF-IDF cosine** — same bag-of-words idea as TextTiling but at unit level.
* **Token-set Jaccard** — pure lexical overlap with no weighting at all.
* **Length-only** — similarity predicted from unit length alone, to expose how
  much of a "structure" can be explained by size differences.

All three use the exact same units as NCD / SBERT, so their matrices are
directly comparable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from common import Unit, tokens


def _cosine_from_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    S = Xn @ Xn.T
    np.fill_diagonal(S, 1.0)
    return np.clip(S, -1.0, 1.0)


def tfidf_matrix(units: Sequence[Unit]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(tokens(u.text)) for u in units]
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)
    X = vec.fit_transform(docs).toarray()
    return _cosine_from_matrix(X)


def jaccard_matrix(units: Sequence[Unit]) -> np.ndarray:
    sets = [set(tokens(u.text)) for u in units]
    n = len(sets)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            inter = len(a & b)
            union = len(a | b)
            v = inter / union if union else 0.0
            M[i, j] = M[j, i] = v
    return M


def length_similarity(units: Sequence[Unit]) -> np.ndarray:
    """A deliberately content-free 'similarity': 1 - |len_i - len_j| / max."""
    n = len(units)
    L = np.array([u.n_words for u in units], dtype=float)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = 1.0 - abs(L[i] - L[j]) / max(L[i], L[j])
            M[i, j] = M[j, i] = v
    return M


# --------------------------------------------------------------------------
# Matrix comparison
# --------------------------------------------------------------------------

def compare_matrices(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Pearson & Spearman correlation of the upper triangles of two matrices."""
    n = a.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    x, y = a[iu, ju], b[iu, ju]
    out = {}
    if x.std() > 0 and y.std() > 0:
        out["pearson"] = float(np.corrcoef(x, y)[0, 1])
    else:
        out["pearson"] = float("nan")
    try:
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(x, y).statistic)
    except Exception:
        # rank-transform fallback
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        out["spearman"] = float(np.corrcoef(rx, ry)[0, 1])
    out["top10_overlap"] = _top_pair_overlap(a, b, k=10)
    return out


def _top_pair_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> int:
    def top(mat):
        n = mat.shape[0]
        iu, ju = np.triu_indices(n, k=1)
        order = np.argsort(-mat[iu, ju])[:k]
        return set(zip(iu[order].tolist(), ju[order].tolist()))
    return len(top(a) & top(b))


def adjacency_matrix_profile(S: np.ndarray) -> Dict[int, float]:
    n = S.shape[0]
    out: Dict[int, List[float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            out.setdefault(j - i, []).append(float(S[i, j]))
    return {g: float(np.mean(v)) for g, v in sorted(out.items())}
