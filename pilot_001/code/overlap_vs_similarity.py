"""How much of the unit-level similarity is simply shared content vocabulary?

The scan found that none of the top 40 non-adjacent pairs has zero shared
content words.  That raises the general question: at the 150-word scale, is
similarity largely a function of lexical overlap?

If it is, then the whole "cluster" language of the pilot describes something
much more mundane than it sounds, and the intervention results (which show that
substituting the shared words damages the link) are simply the same fact seen
from the other side.

Reports, over all non-adjacent pairs (gap >= 5):
  * correlation of cosine with the number of shared content word types, and
    with Jaccard overlap, for both encoders;
  * the same after removing the effect of unit length;
  * the overlap distribution of the top 40 pairs against random pairs.

Writes ``data/derived/overlap_vs_similarity.txt``.
"""

from __future__ import annotations

import random

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []
GAP = 5

STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while yes o oh well""".split())


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def ctypes(text):
    return {t for t in C.tokens(text) if t not in STOP and len(t) > 1}


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    n = len(units)
    types = [ctypes(u.text) for u in units]
    lens = np.array([u.n_words for u in units], dtype=float)

    import torch
    from sentence_transformers import SentenceTransformer

    mats = {}
    for name, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                      ("bge", "BAAI/bge-small-en-v1.5")):
        m = SentenceTransformer(mid)
        m.eval()
        with torch.no_grad():
            e = m.encode([u.text for u in units], batch_size=64,
                         convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        mats[name] = e @ e.T

    pairs = [(i, j) for i in range(n) for j in range(i + GAP, n)]
    shared = np.array([len(types[i] & types[j]) for i, j in pairs], dtype=float)
    union = np.array([len(types[i] | types[j]) for i, j in pairs], dtype=float)
    jac = shared / np.maximum(union, 1)
    size_diff = np.array([abs(lens[i] - lens[j]) for i, j in pairs])

    w("=" * 78)
    w(f"IS SIMILARITY JUST SHARED VOCABULARY?   (all pairs with gap >= {GAP}, "
      f"n = {len(pairs)})")
    w("=" * 78)
    w()
    w(f"   shared content word types per pair: min {shared.min():.0f}  "
      f"median {np.median(shared):.0f}  max {shared.max():.0f}")
    w(f"   Jaccard overlap                   : median {np.median(jac):.3f}")
    w()
    w(f"   {'model':<8}{'r(cos, #shared)':>18}{'r(cos, Jaccard)':>18}"
      f"{'partial: cos~Jacc | len':>26}")
    for name, M in mats.items():
        c = np.array([M[i, j] for i, j in pairs])
        r1 = float(np.corrcoef(c, shared)[0, 1])
        r2 = float(np.corrcoef(c, jac)[0, 1])
        # partial correlation of cos and Jaccard controlling for size difference
        rab, rac, rbc = (r2, float(np.corrcoef(c, size_diff)[0, 1]),
                         float(np.corrcoef(jac, size_diff)[0, 1]))
        den = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
        pr = (rab - rac * rbc) / den if den else float("nan")
        w(f"   {name:<8}{r1:>18.3f}{r2:>18.3f}{pr:>26.3f}")
    w()
    w("   (a high r means the encoder's similarity is largely a restatement")
    w("    of 'these two units use the same words')")
    w()

    # top pairs vs random
    iu, ju = np.triu_indices(n, k=1)
    keep = np.array([(j - i) >= GAP for i, j in zip(iu, ju)])
    rng = random.Random(42)
    for name, M in mats.items():
        vals = np.array([M[i, j] for i, j in zip(iu, ju)])[keep]
        sh = np.array([len(types[i] & types[j]) for i, j in zip(iu, ju)])[keep]
        order = np.argsort(-vals)[:40]
        w(f"   {name}: top-40 pairs share {sh[order].mean():.1f} content types "
          f"on average; all non-adjacent pairs share {sh.mean():.1f}")
    w()

    # how much of the top-40 similarity is above the overlap-matched baseline?
    w("-" * 78)
    w("DOES THE ENCODER SEE ANYTHING BEYOND OVERLAP?")
    w("-" * 78)
    w("   For each top-40 pair, compare its cosine with the mean cosine of")
    w("   pairs having the SAME number of shared content types.")
    w()
    w(f"   {'model':<8}{'top-40 excess over overlap-matched mean':>45}")
    for name, M in mats.items():
        vals = np.array([M[i, j] for i, j in zip(iu, ju)])[keep]
        sh = np.array([len(types[i] & types[j]) for i, j in zip(iu, ju)])[keep]
        order = np.argsort(-vals)[:40]
        exc = []
        for k in order:
            m = sh == sh[k]
            if m.sum() >= 5:
                exc.append(vals[k] - vals[m].mean())
        exc = np.array(exc)
        w(f"   {name:<8}{exc.mean():>+45.3f}")
    w()
    w("   A positive mean means the top pairs are more similar than other")
    w("   pairs with the same amount of shared vocabulary - i.e. there is")
    w("   something beyond overlap.")

    (DERIVED / "overlap_vs_similarity.txt").write_text("\n".join(OUT),
                                                       encoding="utf-8")
    print("wrote", DERIVED / "overlap_vs_similarity.txt")


if __name__ == "__main__":
    main()
