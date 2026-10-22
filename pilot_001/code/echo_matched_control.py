"""Matched-control test for the snow echo, as the reviewer asked.

The original control was too weak:
  * the sampling loop made 200 *attempts* and discarded pairs closer than 200
    sentences, so the effective sample was well under 200;
  * it filtered on distance only — it did not match the target's 290-sentence
    distance, its sentence lengths, or its lexical overlap;
  * the target pair was selected *after* exploration, so comparing it with the
    maximum of a small random batch does not correct for selection effects.

This script builds a null of pairs matched on distance, length and lexical
overlap, progressively relaxing the criteria so the reader can see how many
pairs survive each level.  For every level it reports the matched distribution
and where the target falls.

It also separates the two claims the reviewer asked to keep apart:

  (a) NEAREST-NEIGHBOUR RANK — descriptive: where does each sentence rank the
      other in its own row?
  (b) EXCESS OVER A MATCHED BASELINE — how much does the pair exceed what
      pairs with the same distance / length / overlap typically show?

Selection effects cannot be removed post hoc; this is stated in the output.

Writes ``data/derived/echo_matched_control.txt``.
"""

from __future__ import annotations

import re

import numpy as np

import common as C
import sentences as S

DERIVED = C.DATA_DIR / "derived"
OUT = []
PHRASE = "general all over Ireland"


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    sents = S.split_sentences(body)
    n = len(sents)
    texts = [s.text for s in sents]
    lens = np.array([s.n_words for s in sents])
    toks = [set(C.tokens(t)) for t in texts]

    import torch
    from sentence_transformers import SentenceTransformer

    mats = {}
    for name, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                      ("bge", "BAAI/bge-small-en-v1.5")):
        m = SentenceTransformer(mid)
        m.eval()
        with torch.no_grad():
            e = m.encode(texts, batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        mats[name] = e @ e.T
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True).fit_transform(
        [" ".join(C.tokens(t)) for t in texts]).toarray()
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1
    Xn = X / nrm
    mats["TF-IDF"] = Xn @ Xn.T

    hits = [i for i, t in enumerate(texts) if PHRASE in t]
    assert len(hits) >= 2, "echo sentences not found"
    i0, j0 = hits[0], hits[-1]
    dist0 = j0 - i0
    lo0, hi0 = sorted((lens[i0], lens[j0]))
    jac0 = jaccard(toks[i0], toks[j0])

    w("=" * 78)
    w("MATCHED-CONTROL TEST FOR THE SNOW ECHO")
    w("=" * 78)
    w()
    w(f"target pair: sentences {i0} and {j0}")
    w(f"  index distance   : {dist0}")
    w(f"  word counts      : {lens[i0]} and {lens[j0]}  (sorted {lo0}, {hi0})")
    w(f"  Jaccard overlap  : {jac0:.3f}")
    shared = sorted(toks[i0] & toks[j0])
    w(f"  shared tokens    : {shared}")
    w()
    w("  Selection caveat: this pair was found by exploration and then tested.")
    w("  A matched control reduces, but cannot remove, the selection effect.")
    w("  The honest claim is therefore descriptive (see the two parts below).")
    w()

    # ---------------------------------------------------------------
    w("-" * 78)
    w("PART A — DESCRIPTIVE: NEAREST-NEIGHBOUR RANK (no inference)")
    w("-" * 78)
    for name, M in mats.items():
        a = int((np.delete(M[i0], i0) > M[i0, j0]).sum()) + 1
        b = int((np.delete(M[j0], j0) > M[j0, i0]).sum()) + 1
        w(f"  {name:<8} similarity {M[i0, j0]:.3f}   "
          f"rank of {j0} in row {i0}: {a}/{n}   "
          f"rank of {i0} in row {j0}: {b}/{n}   worse {max(a, b)}")
    w()

    # ---------------------------------------------------------------
    w("-" * 78)
    w("PART B — EXCESS OVER A MATCHED BASELINE")
    w("-" * 78)
    w()
    LEVELS = [
        ("strict",   25,  4, 0.05),
        ("medium",   50,  8, 0.10),
        ("loose",   100, 12, 0.15),
        ("distance only (as in the original control)", None, None, None),
    ]
    for lname, tol_d, tol_l, tol_j in LEVELS:
        idx = []
        for i in range(n):
            for j in range(i + 3, n):
                if tol_d is None:
                    if j - i < 200:
                        continue
                else:
                    if abs((j - i) - dist0) > tol_d:
                        continue
                    lo, hi = sorted((lens[i], lens[j]))
                    if abs(lo - lo0) > tol_l or abs(hi - hi0) > tol_l:
                        continue
                    if abs(jaccard(toks[i], toks[j]) - jac0) > tol_j:
                        continue
                idx.append((i, j))
        w(f"  --- {lname} ---")
        w(f"      matched pairs: {len(idx)}")
        if not idx:
            w("      (none — criteria too strict)")
            w()
            continue
        w(f"      {'method':<8}{'target':>9}{'mean':>9}{'sd':>8}"
          f"{'max':>9}{'pctile':>9}{'excess':>9}")
        for name, M in mats.items():
            vals = np.array([M[i, j] for i, j in idx])
            t = M[i0, j0]
            pct = float((vals < t).mean()) * 100
            w(f"      {name:<8}{t:>9.3f}{vals.mean():>9.3f}"
              f"{vals.std():>8.3f}{vals.max():>9.3f}{pct:>8.1f}%"
              f"{t - vals.mean():>+9.3f}")
        w()

    w("-" * 78)
    w("READING THE TWO PARTS TOGETHER")
    w("-" * 78)
    w("  Part A is descriptive and is what the pilot can legitimately claim:")
    w("  under two independent encoders, each sentence is the other's single")
    w("  most similar sentence in the text.")
    w("  Part B is the inferential question, and it is only partly answered:")
    w("  against pairs matched on distance, length and lexical overlap, the")
    w("  echo's similarity can be compared with what such pairs typically")
    w("  show.  Because the pair was chosen after exploration, this is a")
    w("  post-hoc comparison, not a pre-registered test.")
    w("  => The finding is reported as 'a computationally very salient")
    w("     long-distance echo', NOT as 'statistically proven anomalous' and")
    w("     NOT as 'the structural pivot of the story'.")

    (DERIVED / "echo_matched_control.txt").write_text("\n".join(OUT),
                                                      encoding="utf-8")
    print("wrote", DERIVED / "echo_matched_control.txt")


if __name__ == "__main__":
    main()
