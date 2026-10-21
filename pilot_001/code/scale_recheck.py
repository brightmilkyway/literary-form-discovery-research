"""Re-check the two 150-word-scale results at sentence scale.

The scale verdict showed that the analysis unit can flip a conclusion: "the
snow is invisible to computation" was an artifact of 150-word units.  Two
other results were obtained at that same scale and must therefore be
re-examined:

  (1) TextTiling's strongest boundaries are ORTHOGONAL to what SBERT/TF-IDF
      measure (similarity percentiles 47-56% = no relation).
  (2) NCD's distance matrix is dominated by unit length (r = +0.689 with the
      compressed-size difference), fixed by equal-length units and residual
      regression.

Both are re-run here on the 764-sentence scale, with two TextTiling variants:
  (a) word-based (w=20, k=6), boundaries mapped to the enclosing sentence;
  (b) sentence-based, i.e. token sequences ARE sentences and blocks are k
      sentences — the natural adaptation of the algorithm to this scale.

Also checks whether NCD at sentence scale recovers the verbatim echo
(sentences 534 and 759), which SBERT recovers as a mutual rank-1 pair.

Writes ``data/derived/scale_recheck.txt``.
"""

from __future__ import annotations

import json
import re

import numpy as np

import common as C
import ncd as NCD
import texttiling as TT

DERIVED = C.DATA_DIR / "derived"
OUT = []


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


class Sent:
    """Minimal stand-in for common.Paragraph."""
    __slots__ = ("index", "text", "start", "end", "n_words")

    def __init__(self, index, text, start, end):
        self.index = index
        self.text = text
        self.start = start
        self.end = end
        self.n_words = len(C.word_span(text))


def sentence_units(body):
    import nltk
    units, pos = [], 0
    for s in nltk.sent_tokenize(body):
        idx = body.find(s, pos)
        if idx < 0:
            idx = pos
        units.append(Sent(len(units), s, idx, idx + len(s)))
        pos = idx + len(s)
    return units


# --------------------------------------------------------------------------
# sentence-based TextTiling (variant b)
# --------------------------------------------------------------------------

def texttiling_sentences(sents, k=6, sd_factor=0.5, suppress=4):
    from collections import Counter
    import numpy as np

    seqs = [Counter(C.tokens(s.text)) for s in sents]
    m = len(seqs)
    G = m - 1

    def cos(a, b):
        if not a or not b:
            return 0.0
        num = sum(a[t] * b[t] for t in (set(a) & set(b)))
        if num == 0:
            return 0.0
        d1 = np.sqrt(sum(v * v for v in a.values()))
        d2 = np.sqrt(sum(v * v for v in b.values()))
        return float(num / (d1 * d2)) if d1 and d2 else 0.0

    sims = np.zeros(G)
    for g in range(G):
        lo = max(0, g - k + 1)
        hi = min(m, g + 1 + k)
        left = Counter()
        for c in seqs[lo:g + 1]:
            left.update(c)
        right = Counter()
        for c in seqs[g + 1:hi]:
            right.update(c)
        sims[g] = cos(left, right)

    # flat smoothing (window 3), as in the main implementation
    if G >= 3:
        s = np.r_[2 * sims[0] - sims[2:0:-1], sims,
                  2 * sims[-1] - sims[-1:-3:-1]]
        wgt = np.ones(3)
        sm = np.convolve(wgt / wgt.sum(), s, mode="same")[2:-2]
    else:
        sm = sims

    depths = np.zeros(G)
    clip = min(max(G // 10, 2), 5)
    for idx in range(clip, G - clip):
        g = sm[idx]
        lp = g
        for x in sm[idx::-1]:
            if x >= lp:
                lp = x
            else:
                break
        rp = g
        for x in sm[idx:]:
            if x >= rp:
                rp = x
            else:
                break
        depths[idx] = lp + rp - 2 * g

    avg, sd = float(depths.mean()), float(depths.std())
    cutoff = avg - sd_factor * sd
    order = sorted(range(G), key=lambda i: -depths[i])
    hp = [i for i in order if depths[i] > cutoff]
    is_b = [0] * G
    for dt in hp:
        is_b[dt] = 1
        for dt2 in hp:
            if dt != dt2 and abs(dt2 - dt) < suppress and is_b[dt2] == 1:
                is_b[dt] = 0
                break
    return sims, sm, depths, is_b, cutoff


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    sents = sentence_units(body)
    n = len(sents)
    w(f"body: {len(body)} chars; sentences: {n}")

    import torch
    from sentence_transformers import SentenceTransformer

    texts = [s.text for s in sents]
    mats = {}
    m1 = SentenceTransformer("all-MiniLM-L6-v2"); m1.eval()
    m2 = SentenceTransformer("BAAI/bge-small-en-v1.5"); m2.eval()
    with torch.no_grad():
        e1 = m1.encode(texts, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
        e2 = m2.encode(texts, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    mats["MiniLM"] = e1 @ e1.T
    mats["bge"] = e2 @ e2.T
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True).fit_transform(
        [" ".join(C.tokens(t)) for t in texts]).toarray()
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1
    Xn = X / nrm
    mats["TF-IDF"] = Xn @ Xn.T
    # Jaccard for completeness
    sets = [set(C.tokens(t)) for t in texts]
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            u = len(a | b)
            J[i, j] = J[j, i] = (len(a & b) / u) if u else 0.0
    mats["Jaccard"] = J

    # ==================================================================
    w("")
    w("=" * 78)
    w("PART 1 — IS TEXTTILING STILL ORTHOGONAL AT SENTENCE SCALE?")
    w("=" * 78)
    w()

    # ---- variant (a): word-based TextTiling, boundaries -> sentences ----
    tt = TT.run_texttiling(body, sents, w=20, k=6, units=None)
    gaps = []
    for b in tt["boundaries"]:
        cp = b["char_pos"]
        k = None
        for i, s in enumerate(sents):
            if s.start <= cp <= s.end:
                k = i
                break
        if k is None:
            continue
        if k - 1 >= 0:
            gaps.append((k - 1, k))
        if k + 1 < n:
            gaps.append((k, k + 1))
    gaps = sorted(set(gaps))
    w(f"VARIANT (a) word-based TextTiling (w=20, k=6): "
      f"{len(tt['boundaries'])} boundaries -> {len(gaps)} straddled "
      f"adjacent sentence pairs")
    for lab, M in mats.items():
        alladj = np.array([M[i, i + 1] for i in range(n - 1)])
        vals = np.array([M[i, j] for i, j in gaps])
        pct = float(np.mean([(alladj < v).mean() for v in vals])) * 100
        w(f"   {lab:<8} at boundaries mean={vals.mean():+.3f} | "
          f"all adjacent mean={alladj.mean():+.3f} | percentile {pct:.0f}%")
    w()

    # ---- variant (b): sentence-based TextTiling ------------------------
    for k in (4, 6, 10):
        sims, sm, depths, is_b, cutoff = texttiling_sentences(sents, k=k)
        gaps_b = [g for g in range(len(is_b)) if is_b[g]]
        pairs = [(g, g + 1) for g in gaps_b]
        w(f"VARIANT (b) sentence-based TextTiling (k={k}): "
          f"{len(gaps_b)} boundaries")
        for lab, M in mats.items():
            alladj = np.array([M[i, i + 1] for i in range(n - 1)])
            vals = np.array([M[i, j] for i, j in pairs])
            if vals.size == 0:
                continue
            pct = float(np.mean([(alladj < v).mean() for v in vals])) * 100
            w(f"   {lab:<8} at boundaries mean={vals.mean():+.3f} | "
              f"all adjacent mean={alladj.mean():+.3f} | percentile {pct:.0f}%")
        w()
    w("   (50% = no relation.  At 150 words the same test gave 47-56% for")
    w("    five representations.)")
    w()

    # ==================================================================
    w("=" * 78)
    w("PART 2 — NCD AT SENTENCE SCALE")
    w("=" * 78)
    w()

    keep = [i for i, s in enumerate(sents) if s.n_words >= 5]
    w(f"sentences with >= 5 words: {len(keep)} of {n}")
    sub = [sents[i] for i in keep]
    res = NCD.ncd_matrix(sub, "zlib")
    conf = NCD.length_confound(res)
    w(f"   r(NCD, compressed-size difference) = "
      f"{conf['corr_ncd_size_difference']:+.3f}")
    w(f"   r(NCD, index distance)             = "
      f"{conf['corr_ncd_index_distance']:+.3f}")
    w(f"   (at 150-word scale the first was +0.689, the second +0.155)")
    w()

    sizes = res.sizes
    M = res.matrix
    m = len(sub)
    iu, ju = np.triu_indices(m, k=1)
    d = M[iu, ju]
    sd = np.abs(sizes[iu] - sizes[ju])
    b1, a1 = np.polyfit(sd, d, 1)
    resid = d - (a1 + b1 * sd)
    R = np.full((m, m), np.nan)
    R[iu, ju] = resid
    R[ju, iu] = resid
    np.fill_diagonal(R, 0.0)
    w(f"   residual fix: D = {a1:.4f} + {b1:.6f}*|size_i - size_j|; "
      f"variance explained by size alone = "
      f"{np.corrcoef(sd, d)[0, 1] ** 2:.3f}")
    w(f"   r(residual, size difference) = "
      f"{np.corrcoef(resid, sd)[0, 1]:+.3f}  (by construction)")
    w()

    # ---- does NCD find the verbatim echo at sentence scale? ------------
    if 534 in keep and 759 in keep:
        i, j = keep.index(534), keep.index(759)
        r_raw = int((M[iu, ju] < M[i, j]).sum()) + 1
        r_res = int((R[iu, ju] < R[i, j]).sum()) + 1
        w("   VERBATIM ECHO at sentence scale, under NCD:")
        w(f"      sentences 534 & 759 -> matrix indices {i}, {j}")
        w(f"      raw NCD distance {M[i, j]:.3f}, rank {r_raw}/{len(iu)} "
          f"(lower = more similar)")
        w(f"      residual NCD {R[i, j]:+.4f}, rank {r_res}/{len(iu)}")
        for lab, S in mats.items():
            a = int((np.delete(S[534], 534) > S[534, 759]).sum()) + 1
            w(f"      {lab} similarity {S[534, 759]:.3f}, mutual rank "
              f"from 534: {a}/{n}")
    w()
    w("   NCD's top-10 closest non-adjacent sentence pairs (raw):")
    pairs = [(float(M[i, j]), keep[i], keep[j]) for i in range(m)
             for j in range(i + 3, m)]
    pairs.sort(key=lambda t: t[0])
    for v, i, j in pairs[:10]:
        w(f"      d={v:.3f}  sents {i}&{j} (gap {j - i})")
        w(f"         {i}: {re.sub(r'\\s+', ' ', sents[i].text)[:95]}")
        w(f"         {j}: {re.sub(r'\\s+', ' ', sents[j].text)[:95]}")

    (DERIVED / "scale_recheck.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "scale_recheck.txt")


if __name__ == "__main__":
    main()
