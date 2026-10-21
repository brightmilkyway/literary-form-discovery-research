"""Alternative explanations for the patterns seen in the main run.

The pilot's own rules (§3, §5) demand that every eye-catching pattern be tested
against the cheapest explanations before it is called interesting.  Here we add
two content-free "register" features that no method was given:

* ``dialogue ratio`` — share of a unit's characters that sit inside “ ”.
* ``quotation-mark density`` — “ per 100 words.

Then we ask: how much of each method's similarity matrix is predicted by unit
length and by dialogue ratio alone?  And are the standout pairs just
dialogue-units matching other dialogue-units?

Writes ``data/derived/alternative_explanations.txt``.
"""

from __future__ import annotations

import json
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def prev(t, n=150):
    s = re.sub(r"\s+", " ", t).strip()
    return s[:n] + ("…" if len(s) > n else "")


def dialogue_ratio(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    inside = 0
    depth = 0
    for ch in text:
        if ch == "\u201c":
            depth += 1
        elif ch == "\u201d":
            depth = max(0, depth - 1)
        elif depth > 0:
            inside += 1
    return inside / total


def quote_density(text: str, n_words: int) -> float:
    return 100.0 * text.count("\u201c") / max(1, n_words)


def sim_from_scalar(x: np.ndarray) -> np.ndarray:
    n = len(x)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = 1.0 - abs(x[i] - x[j]) / (abs(x).max() - abs(x).min() + 1e-9)
            M[i, j] = M[j, i] = v
    return M


def tri(M):
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    return M[iu, ju]


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial(a, b, c):
    """corr(a,b | c) using the standard three-way formula."""
    rab, rac, rbc = corr(a, b), corr(a, c), corr(b, c)
    den = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
    return float((rab - rac * rbc) / den) if den else float("nan")


def main():
    units = json.loads((DERIVED / "units.json").read_text(encoding="utf-8"))["group150"]
    body = C.strip_title(C.load_raw())[0]

    NCD = np.load(DERIVED / "ncd_group150_zlib.npy")
    SB = np.load(DERIVED / "sbert_group150.npy")
    TF = np.load(DERIVED / "tfidf_group150.npy")
    LN = np.load(DERIVED / "length_group150.npy")

    texts = [body[u["start"]:u["end"]] for u in units]
    words = np.array([u["n_words"] for u in units], dtype=float)
    dr = np.array([dialogue_ratio(t) for t in texts])
    qd = np.array([quote_density(t, u["n_words"]) for t, u in zip(texts, units)])

    DR = sim_from_scalar(dr)
    QD = sim_from_scalar(qd)

    w("=" * 78)
    w("ALTERNATIVE EXPLANATIONS — do cheap features already explain the patterns?")
    w("=" * 78)
    w()
    w(f"unit length  : min {words.min():.0f}  median {np.median(words):.0f}  "
      f"max {words.max():.0f} words")
    w(f"dialogue ratio: min {dr.min():.2f}  median {np.median(dr):.2f}  "
      f"max {dr.max():.2f}   (share of chars inside “ ”)")
    w()

    # ---- how much is explained by length & register ----------------------
    w("A. CORRELATION OF EACH METHOD'S SIMILARITY WITH CONTENT-FREE FEATURES")
    w("   (Pearson r on upper triangles; NCD is flipped to a similarity)")
    feats = {"length-sim": LN, "dialogue-ratio-sim": DR, "quote-density-sim": QD}
    w("   method      " + "".join(f"{k:>20}" for k in feats))
    for lab, M in (("NCD(1-d)", 1 - NCD), ("SBERT", SB), ("TF-IDF", TF)):
        row = "".join(f"{corr(tri(M), tri(F)):>20.3f}" for F in feats.values())
        w(f"   {lab:<11}{row}")
    w()

    w("B. PARTIAL CORRELATION: does NCD still track lexical content once")
    w("   unit length is controlled for?")
    ncd, tf, ln = tri(1 - NCD), tri(TF), tri(LN)
    w(f"   r(NCD, TF-IDF)                 = {corr(ncd, tf):+.3f}")
    w(f"   r(NCD, length-sim)             = {corr(ncd, ln):+.3f}")
    w(f"   r(NCD, TF-IDF | length)        = {partial(ncd, tf, ln):+.3f}")
    w(f"   r(SBERT, TF-IDF)               = {corr(tri(SB), tf):+.3f}")
    w(f"   r(SBERT, length-sim)           = {corr(tri(SB), tri(LN)):+.3f}")
    w()

    # ---- are the standout pairs just dialogue units? --------------------
    w("C. ARE THE STANDOUT PAIRS JUST 'DIALOGUE MATCHING DIALOGUE'?")
    def top_pairs(M, k=12, min_gap=3, invert=False):
        n = len(units)
        P = [(float(M[i, j]), i, j) for i in range(n)
             for j in range(i + min_gap, n)]
        P.sort(key=(lambda t: t[0]) if invert else (lambda t: -t[0]))
        return P[:k]
    for lab, P, inv in (("NCD (closest)", top_pairs(NCD, invert=True), True),
                        ("SBERT (closest)", top_pairs(SB, invert=False), False)):
        idx = sorted({i for _, i, _ in P} | {j for _, _, j in P})
        w(f"   {lab}: mean dialogue ratio of the units involved = "
          f"{dr[idx].mean():.2f}  (all units: {dr.mean():.2f}); "
          f"units={idx}")
    w()
    w("   dialogue ratio of every unit (index: ratio):")
    w("   " + "  ".join(f"{i}:{dr[i]:.2f}" for i in range(len(units))))
    w()

    # ---- the one genuinely distant SBERT pair ---------------------------
    w("D. THE ONE GENUINELY DISTANT SBERT PAIR (gap > 40) IN THE TOP-20")
    n = len(units)
    P = [(float(SB[i, j]), i, j) for i in range(n)
         for j in range(i + 3, n) if j - i > 40]
    P.sort(key=lambda t: -t[0])
    for s, i, j in P[:5]:
        w(f"   cos={s:.3f}  units {i}&{j} (gap {j-i}; "
          f"words {words[i]:.0f} & {words[j]:.0f}; "
          f"dialogue {dr[i]:.2f} & {dr[j]:.2f})")
        w(f"      {i}: {prev(texts[i], 260)}")
        w(f"      {j}: {prev(texts[j], 260)}")
    w()

    # ---- longest shared rare content words, for the distant pair ---------
    w("E. WHAT SHARED *CONTENT* WORDS DRIVE THE DISTANT PAIR (2, 73)?")
    from collections import Counter
    sw = set(C.__dict__.get("_STOP", []) or [])
    stop = set("""a an the and or but if of to in on at for with by from as is was were
    be been being he she it they them his her their its i you we me my your that this
    these those not no so do did does had has have will would could should there here
    then than when where which who whom what how all any both each few more most other
    some such only own same too very can just now up down out off over under again
    further once about into through during before after above below between""".split())
    for (i, j) in [(2, 73)]:
        a = [t for t in C.tokens(texts[i]) if t not in stop and len(t) > 3]
        b = [t for t in C.tokens(texts[j]) if t not in stop and len(t) > 3]
        ca, cb = Counter(a), Counter(b)
        shared = sorted(set(ca) & set(cb))
        w(f"   units {i}&{j} shared content words: {shared}")
    w()

    txt = "\n".join(OUT)
    (DERIVED / "alternative_explanations.txt").write_text(txt, encoding="utf-8")
    print("wrote", DERIVED / "alternative_explanations.txt")


if __name__ == "__main__":
    main()
