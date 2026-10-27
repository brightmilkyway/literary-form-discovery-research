"""Reviewer checks on NCD: concatenation order, and the 'template' claim.

Two criticisms addressed here.

1.  **Concatenation order.**  ``ncd.py`` computes ``C(xy)`` once in reading
    order and mirrors it into a symmetric matrix.  Practical compressors do
    not satisfy ``C(xy) = C(yx)``, so the symmetry is an assumption, not a
    measurement.  This script computes both directions and reports how much
    the matrix changes, whether the asymmetry is length-related, and how much
    the top pairs change.

2.  **Is NCD a 'formulaic-template detector', or just a length effect?**  The
    claim came from top pairs that share a sentence frame.  But sentence-scale
    NCD is strongly length-confounded, so the frame effect may be a by-product.
    Test: residualise NCD against the size difference and inspect the top pairs
    again; and check whether the raw top pairs are unusually length-matched.

Writes ``data/derived/ncd_reviewer_checks.txt``.
"""

from __future__ import annotations

import re

import numpy as np

import common as C
import ncd as NCD
import sentences as S

DERIVED = C.DATA_DIR / "derived"
OUT = []


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def both_orders(units, compressor="zlib"):
    """Return (forward, reverse) NCD matrices computed independently."""
    comp = NCD.COMPRESSORS[compressor]
    blobs = [u.text.encode("utf-8") for u in units]
    sizes = np.array([comp(b) for b in blobs], dtype=float)
    n = len(units)
    F = np.zeros((n, n))
    Rv = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cxy = comp(blobs[i] + blobs[j])
            cyx = comp(blobs[j] + blobs[i])
            m = max(sizes[i], sizes[j])
            F[i, j] = F[j, i] = (cxy - min(sizes[i], sizes[j])) / m
            Rv[i, j] = Rv[j, i] = (cyx - min(sizes[i], sizes[j])) / m
    return F, Rv, sizes


def top_pairs(M, units, k=10, min_gap=3):
    n = len(units)
    p = [(float(M[i, j]), i, j) for i in range(n) for j in range(i + min_gap, n)]
    p.sort(key=lambda t: t[0])
    return p[:k]


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units150 = C.build_units(body, "group", target_words=150)

    w("=" * 78)
    w("NCD REVIEWER CHECKS")
    w("=" * 78)
    w()

    # ==================================================================
    w("-" * 78)
    w("1. CONCATENATION ORDER: is the symmetric matrix an assumption?")
    w("-" * 78)
    for label, units in (("group150 (n=78)", units150),):
        F, Rv, sizes = both_orders(units)
        n = len(units)
        iu, ju = np.triu_indices(n, k=1)
        a, b = F[iu, ju], Rv[iu, ju]
        diff = np.abs(a - b)
        w(f"   {label}")
        w(f"      r(forward, reverse)          = {np.corrcoef(a, b)[0,1]:+.4f}")
        w(f"      |C(xy)-C(yx)| as NCD units   : mean {diff.mean():.4f}, "
          f"max {diff.max():.4f}")
        w(f"      pairs whose order changes the NCD by > 0.01: "
          f"{int((diff > 0.01).sum())} of {len(diff)} "
          f"({(diff > 0.01).mean()*100:.1f}%)")
        size_diff = np.abs(sizes[iu] - sizes[ju])
        w(f"      r(|NCD(xy)-NCD(yx)|, size difference) = "
          f"{np.corrcoef(diff, size_diff)[0,1]:+.3f}")
        w(f"      r(|NCD(xy)-NCD(yx)|, NCD forward)      = "
          f"{np.corrcoef(diff, a)[0,1]:+.3f}")
        tf = {(i, j) for _, i, j in top_pairs(F, units)}
        tr = {(i, j) for _, i, j in top_pairs(Rv, units)}
        w(f"      top-10 closest pairs: overlap between the two orders = "
          f"{len(tf & tr)}/10")
        for (i, j) in sorted(tf - tr):
            w(f"         only in forward: ({i},{j})")
        for (i, j) in sorted(tr - tf):
            w(f"         only in reverse: ({i},{j})")
    w()

    # ==================================================================
    w("-" * 78)
    w("2. IS THE 'TEMPLATE' EFFECT JUST LENGTH?")
    w("-" * 78)
    sents = S.split_sentences(body)
    keep = [s for s in sents if s.n_words >= 5]
    w(f"   sentences with >= 5 words: {len(keep)}")
    res = NCD.ncd_matrix(keep, "zlib")
    M = res.matrix
    sizes = res.sizes
    n = len(keep)
    iu, ju = np.triu_indices(n, k=1)
    d = M[iu, ju]
    sd = np.abs(sizes[iu] - sizes[ju])
    b1, a1 = np.polyfit(sd, d, 1)
    resid = d - (a1 + b1 * sd)
    R = np.full((n, n), np.nan)
    R[iu, ju] = resid
    R[ju, iu] = resid
    np.fill_diagonal(R, 0.0)
    w(f"   raw NCD: r(size difference) = {np.corrcoef(d, sd)[0,1]:+.3f}; "
      f"variance explained = {np.corrcoef(d, sd)[0,1]**2:.3f}")
    w()

    def show(pairs, title):
        w(f"   {title}")
        for v, i, j in pairs:
            wc_i, wc_j = keep[i].n_words, keep[j].n_words
            w(f"      d={v:+.3f}  |len {wc_i} vs {wc_j}|  "
              f"sents {keep[i].index}&{keep[j].index}")
            w(f"         {re.sub(r'\\s+', ' ', keep[i].text)[:88]}")
            w(f"         {re.sub(r'\\s+', ' ', keep[j].text)[:88]}")
        w()

    show(top_pairs(M, keep, k=10), "RAW NCD, top-10 closest non-adjacent:")
    show(top_pairs(R, keep, k=10),
         "RESIDUAL NCD (length regressed out), top-10:")

    # are the raw top pairs unusually length-matched?
    raw_top = top_pairs(M, keep, k=50)
    ld = np.array([abs(keep[i].n_words - keep[j].n_words) for _, i, j in raw_top])
    all_ld = np.array([abs(keep[i].n_words - keep[j].n_words)
                       for i in range(n) for j in range(i + 3, n)])
    w(f"   length difference of the 50 closest raw-NCD pairs: "
      f"mean {ld.mean():.1f} words")
    w(f"   length difference of all non-adjacent pairs          : "
      f"mean {all_ld.mean():.1f} words")
    w(f"   -> the closest pairs are "
      f"{'length-matched' if ld.mean() < all_ld.mean() * 0.7 else 'NOT especially length-matched'}")
    w()
    w("   Same statistic for the residual matrix's 50 closest pairs:")
    res_top = top_pairs(R, keep, k=50)
    ld2 = np.array([abs(keep[i].n_words - keep[j].n_words) for _, i, j in res_top])
    w(f"      mean {ld2.mean():.1f} words  "
      f"({'length-matched' if ld2.mean() < all_ld.mean() * 0.7 else 'not especially length-matched'})")
    w()

    # how many of the residual top-10 share a sentence frame?
    FRAME = re.compile(r"(said|asked|cried|answered|repeated|exclaimed)\s")
    shared = 0
    for _, i, j in top_pairs(R, keep, k=10):
        fi = set(FRAME.findall(keep[i].text.lower()))
        fj = set(FRAME.findall(keep[j].text.lower()))
        ci = set(C.tokens(keep[i].text)) - {"said", "asked"}
        cj = set(C.tokens(keep[j].text)) - {"said", "asked"}
        if (fi & fj) or len(ci & cj) >= 2:
            shared += 1
    w(f"   of the residual top-10 pairs, {shared}/10 share a dialogue-tag "
      f"frame or >= 2 content words")
    w()

    # ---- the equal-length test the reviewer asked for -------------------
    w("-" * 78)
    w("3. EQUAL-LENGTH TEST: within sentences of the SAME word count, do")
    w("   frame-sharing pairs still have lower NCD?")
    w("-" * 78)
    FRAME_WORDS = {"said", "asked", "cried", "answered", "repeated",
                   "exclaimed", "murmured", "added", "continued"}
    by_len = {}
    for idx, s in enumerate(keep):
        by_len.setdefault(s.n_words, []).append(idx)
    rows = []
    for L, idxs in sorted(by_len.items()):
        if len(idxs) < 25:
            continue
        frame = [k for k in idxs
                 if set(C.tokens(keep[k].text)) & FRAME_WORDS]
        plain = [k for k in idxs
                 if not (set(C.tokens(keep[k].text)) & FRAME_WORDS)]
        if len(frame) < 3 or len(plain) < 3:
            continue
        ff = [M[a, b] for a in frame for b in frame if a < b]
        pp = [M[a, b] for a in plain for b in plain if a < b]
        fp = [M[a, b] for a in frame for b in plain]
        rows.append((L, len(idxs), len(frame), len(plain),
                     float(np.mean(ff)), float(np.mean(pp)), float(np.mean(fp))))
    if rows:
        w(f"   {'words':>5}{'n':>6}{'frame':>7}{'plain':>7}"
          f"{'frame-frame':>13}{'plain-plain':>13}{'frame-plain':>13}")
        for L, n_, nf, npl, ff, pp, fp in rows:
            w(f"   {L:>5}{n_:>6}{nf:>7}{npl:>7}{ff:>13.3f}{pp:>13.3f}{fp:>13.3f}")
        diffs = [pp - ff for _, _, _, _, ff, pp, _ in rows]
        w()
        w(f"   within every equal-length bin, mean(plain-plain) - "
          f"mean(frame-frame):")
        w(f"      {[round(x, 3) for x in diffs]}")
        w(f"      positive in {sum(1 for x in diffs if x > 0)}/{len(diffs)} bins")
        w()
        w("   A positive value means frame-sharing sentences are MORE similar")
        w("   to each other than non-frame sentences OF THE SAME LENGTH, i.e.")
        w("   the frame effect is not a length effect.")
    else:
        w("   not enough equal-length bins with >= 25 sentences")

    (DERIVED / "ncd_reviewer_checks.txt").write_text("\n".join(OUT),
                                                     encoding="utf-8")
    print("wrote", DERIVED / "ncd_reviewer_checks.txt")


if __name__ == "__main__":
    main()
