"""Does removing the anisotropy restore content sensitivity?

Why this exists
  v16 measured the baseline: 384 dimensions but only ~31 effectively used
  (participation ratio), cosine to the centroid 0.71 / 0.85, cross-story
  baseline 0.40 / 0.69 against ~0.11 for an isotropic cloud.  And v13 measured
  that shared vocabulary explains only 18% / 27% of the similarity variance.

  If the content signal is being buried by a few dominant directions, then
  whitening the space should bring it back.  If it is not, whitening changes
  the numbers but not the structure.

Design
  Fit the whitening on the 256 units of the OTHER 14 stories, apply it to the
  78 units of The Dead.  Fitting and evaluating on the same units would be
  circular; this keeps the two apart.

  Then re-measure, before and after whitening:
    * mean pairwise cosine, within The Dead vs cross-story  (v16's test)
    * r(cosine, number of shared content types)             (v13's test)
    * top-40 excess over the overlap-matched mean           (v13's test)

  Whitening is ZCA: centre, then rotate by V diag(1/sqrt(lambda)) V^T with a
  small ridge.  Embeddings are re-normalised after the transform so the numbers
  stay comparable in kind.

Writes ``data/derived/whitening_test_<tag>.txt``.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np

import common as C
from ladder_principled import build_pooler
from baseline_decomposition import load_other_stories

DERIVED = C.DATA_DIR / "derived"
MIN_GAP = 5


def zca_fit(X: np.ndarray, ridge: float = 1e-3):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    lam, V = np.linalg.eigh(cov)
    lam = np.maximum(lam, 0)
    scale = 1.0 / np.sqrt(lam + ridge * lam.mean())
    W = (V * scale) @ V.T
    return mu, W

def apply_white(X, mu, W):
    Z = (X - mu) @ W
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="minilm")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--sweep", action="store_true",
                    help="report the key numbers across ridge values")
    args = ap.parse_args()

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    dead_units = C.build_units(body, "group", target_words=150)
    stories = load_other_stories()
    other = []
    for name, ps in stories:
        for u in C.units_by_group(ps, 150):
            if u.n_words >= 100:
                other.append((name, u))
    names = [s for s, _ in other]

    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(args.model)
    m.eval()
    enc = build_pooler(m)
    D = enc([u.text for u in dead_units])
    O = enc([u[1].text for u in other])

    mu, W = zca_fit(O, args.ridge)         # fitted on the OTHER stories only
    Dw = apply_white(D, mu, W)
    Ow = apply_white(O, mu, W)

    def ctypes_of(u):
        return {t for t in C.tokens(u.text)}

    def within(E, gap=MIN_GAP):
        S = E @ E.T
        iu = np.triu_indices(len(E), k=1)
        k = (iu[1] - iu[0]) >= gap
        return float(S[iu[0][k], iu[1][k]].mean())

    def across(A, B):
        return float((A @ B.T).mean())

    def rho(E, units):
        """correlation of cosine with the number of shared content types."""
        sets = [ctypes_of(u) for u in units]
        S = E @ E.T
        iu = np.triu_indices(len(E), k=1)
        k = (iu[1] - iu[0]) >= MIN_GAP
        a, b = iu[0][k], iu[1][k]
        sh = np.array([len(sets[i] & sets[j]) for i, j in zip(a, b)],
                      dtype=float)
        cs = S[a, b]
        return float(np.corrcoef(sh, cs)[0, 1]), sh, cs

    def excess(E, units, top):
        _, sh, cs = rho(E, units)
        order = np.argsort(-cs)[:top]
        vals = []
        for i in order:
            same = np.isclose(sh, sh[i])
            vals.append(cs[i] - cs[same].mean())
        return float(np.mean(vals))

    def pr(E):
        X = E - E.mean(0, keepdims=True)
        cov = (X.T @ X) / max(1, len(X) - 1)
        lam = np.linalg.eigvalsh(cov)
        lam = lam[lam > 1e-12]
        return float(lam.sum() ** 2 / (lam ** 2).sum())

    OUT = []
    def w(*a):
        OUT.append(" ".join(str(x) for x in a))

    w("=" * 78)
    w("DOES WHITENING RESTORE CONTENT SENSITIVITY?")
    w("=" * 78)
    w(f"  model {args.model}")
    w(f"  whitening fitted on {len(O)} units of the other 14 stories,")
    w(f"  applied to the {len(D)} units of The Dead")
    w("")
    w(f"  {'':<34}{'raw':>10}{'whitened':>12}")
    w(f"  {'participation ratio (dead)':<34}{pr(D):>10.1f}{pr(Dw):>12.1f}")
    w(f"  {'participation ratio (other)':<34}{pr(O):>10.1f}{pr(Ow):>12.1f}")
    w("")
    w("  v16's test — same-story premium")
    w(f"  {'within The Dead (gap>=5)':<34}{within(D):>10.3f}{within(Dw):>12.3f}")
    w(f"  {'The Dead x other stories':<34}{across(D, O):>10.3f}"
      f"{across(Dw, Ow):>12.3f}")
    w(f"  {'premium':<34}{within(D) - across(D, O):>+10.3f}"
      f"{within(Dw) - across(Dw, Ow):>+12.3f}")
    w("")
    w("  v13's test — is similarity a restatement of shared vocabulary?")
    r_raw, _, _ = rho(D, dead_units)
    r_whi, _, _ = rho(Dw, dead_units)
    w(f"  {'r(cos, #shared content types)':<34}{r_raw:>+10.3f}{r_whi:>+12.3f}")
    e_raw = excess(D, dead_units, args.top)
    e_whi = excess(Dw, dead_units, args.top)
    w(f"  {f'top-{args.top} excess over overlap-matched':<34}"
      f"{e_raw:>+10.3f}{e_whi:>+12.3f}")
    w("")
    w("  READING")
    w("    If the content signal was buried under a few dominant directions,")
    w("    whitening should raise r(cos, #shared) and shrink the excess.")
    w("    If it does neither, anisotropy is not what is hiding the content.")
    w("")

    if args.sweep:
        w("  ROBUSTNESS — the same numbers across ridge values")
        w("    (a bigger ridge whitens less; the raw row is no whitening)")
        w(f"    {'ridge':>9}{'r(cos,#shared)':>17}{'top-40 excess':>16}"
          f"{'within-Dead':>14}{'premium':>11}")
        w(f"    {'raw':>9}{r_raw:>+17.3f}{e_raw:>+16.3f}"
          f"{within(D):>14.3f}{within(D) - across(D, O):>+11.3f}")
        for rg in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
            mu2, W2 = zca_fit(O, rg)
            D2 = apply_white(D, mu2, W2)
            O2 = apply_white(O, mu2, W2)
            r2, _, _ = rho(D2, dead_units)
            e2 = excess(D2, dead_units, args.top)
            w(f"    {rg:>9.0e}{r2:>+17.3f}{e2:>+16.3f}"
              f"{within(D2):>14.3f}"
              f"{within(D2) - across(D2, O2):>+11.3f}")
        w("")

    out = DERIVED / f"whitening_test_{args.tag}.txt"
    out.write_text("\n".join(OUT), encoding="utf-8")
    print("\n".join(OUT))
    print("wrote", out)


if __name__ == "__main__":
    main()
