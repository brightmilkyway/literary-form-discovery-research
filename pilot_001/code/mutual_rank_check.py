"""Correction-free test for every candidate pair: mutual nearest-neighbour rank.

Hub correction has a symmetric flaw: it removes hub inflation but *inflates*
pairs between two **peripheral** units (low row mean -> large positive
residual).  The pair (30, 39) is exactly that case, so its "rank 1" may be an
artifact of the correction rather than a property of the text.

The correction-free alternative: for a pair (i, j), ask
  * where does j rank inside i's row (how similar to i is j, out of all units)?
  * where does i rank inside j's row?
A genuine link should be a **mutual near-neighbour** in both rows.

Writes ``data/derived/mutual_rank_check.txt``.
"""

from __future__ import annotations

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

PAIRS = [
    (2, 73, "'snow frame'"),
    (30, 39, "religion / Church thread"),
    (55, 58, "song lyric vs. song name"),
    (12, 16, "Mr Browne & Freddy"),
    (71, 74, "cold interrogation vs. Gretta sobbing"),
    (37, 58, "singers past vs. song name"),
    (69, 72, "Gabriel asks / Gretta's revelation"),
]


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def row_rank(S, i, j):
    """Rank of j within row i (1 = j is i's most similar unit)."""
    row = S[i].copy()
    row[i] = -np.inf
    return int((row > row[j]).sum()) + 1


def main():
    C.set_seed(42)
    models = {
        "MiniLM-pooled": np.load(DERIVED / "sbert_group150_pooled.npy"),
        "bge-small": np.load(DERIVED / "sbert_group150_bge.npy"),
    }
    n = list(models.values())[0].shape[0]

    w("=" * 78)
    w("MUTUAL-RANK CHECK (correction-free)")
    w("=" * 78)
    w()
    w(f"   {n} units.  A pair is a genuine link if each unit ranks the other")
    w("   high inside its own row — this needs no hub correction at all.")
    w()
    for nm, S in models.items():
        w(f"--- {nm} ---")
        w(f"   {'pair':<9}{'i->j':>7}{'j->i':>7}{'worse of two':>14}   note")
        for (i, j, note) in PAIRS:
            r1 = row_rank(S, i, j)
            r2 = row_rank(S, j, i)
            worst = max(r1, r2)
            w(f"   {f'{i}&{j}':<9}{r1:>7}{r2:>7}{worst:>14}   {note}")
        w()

    # how extreme is a "worse of two" rank of k, by chance?
    w("REFERENCE — distribution of max(row-rank_i(j), row-rank_j(i)) over")
    w("            1000 random pairs, per model")
    rng = np.random.default_rng(42)
    for nm, S in models.items():
        vals = []
        for _ in range(1000):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            vals.append(max(row_rank(S, i, j), row_rank(S, j, i)))
        vals = np.array(vals)
        w(f"   {nm}: median {np.median(vals):.0f}, "
          f"10th pct {np.percentile(vals, 10):.0f}, "
          f"5th pct {np.percentile(vals, 5):.0f}, "
          f"1st pct {np.percentile(vals, 1):.0f}")
    w()
    w("   A pair is convincing only if its 'worse of two' rank is near the")
    w("   1st percentile of this null (roughly <= 5 of 78).")
    w()

    # and the same for the peripheral-unit concern
    w("DIAGNOSTIC — are the two units of each pair peripheral?")
    for nm, S in models.items():
        nn = S.shape[0]
        off = ~np.eye(nn, dtype=bool)
        rm = np.array([S[i, off[i]].mean() for i in range(nn)])
        w(f"--- {nm} (global mean {rm.mean():.3f}) ---")
        for (i, j, note) in PAIRS:
            ri = int((rm < rm[i]).sum()) + 1
            rj = int((rm < rm[j]).sum()) + 1
            w(f"   ({i},{j}) centrality ranks {ri:>2} and {rj:>2} of {nn} "
              f"(1 = most peripheral)   {note}")

    (DERIVED / "mutual_rank_check.txt").write_text("\n".join(OUT),
                                                   encoding="utf-8")
    print("wrote", DERIVED / "mutual_rank_check.txt")


if __name__ == "__main__":
    main()
