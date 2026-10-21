"""Cross-check every claim against all four representations after the fixes.

Representations compared
------------------------
* ``NCD group150``            original (length-confounded)
* ``NCD residual``            group150 with the size-difference term regressed out
* ``NCD equal150``            equal-word-count units (structural control)
* ``SBERT truncated``         original (32% of units truncated)
* ``SBERT pooled``            paragraph-aware mean pooling (0% truncated)
* ``SBERT group80``           80-word units

Writes ``data/derived/robustness_check.txt``.
"""

from __future__ import annotations

import json
import re

import numpy as np

import common as C
import baselines as BL

DERIVED = C.DATA_DIR / "derived"
OUT = []


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def prev(t, n=110):
    s = re.sub(r"\s+", " ", t).strip()
    return s[:n] + ("…" if len(s) > n else "")


def hub_resid(S):
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    rm = np.array([S[i, off[i]].mean() for i in range(n)])
    R = S - rm[:, None] - rm[None, :] + S[off].mean()
    np.fill_diagonal(R, -np.inf)
    return rm, R


def rank_of(M, i, j):
    """1 = most similar, among all upper-triangle pairs."""
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    v = M[i, j]
    return int((M[iu, ju] > v).sum()) + 1


def main():
    body = C.strip_title(C.load_raw())[0]
    U150 = json.loads((DERIVED / "units.json").read_text(encoding="utf-8"))["group150"]
    units150 = C.build_units(body, "group", target_words=150)
    U80 = C.build_units(body, "group", target_words=80)
    Ueq = C.build_units(body, "window", window=150, stride=150)

    NCD_g = np.load(DERIVED / "ncd_group150_zlib.npy")
    NCD_res = np.load(DERIVED / "ncd_group150_zlib_residual.npy")
    NCD_eq = np.load(DERIVED / "ncd_equal150_zlib.npy")
    SB_tr = np.load(DERIVED / "sbert_group150.npy")
    SB_po = np.load(DERIVED / "sbert_group150_pooled.npy")
    SB_80 = np.load(DERIVED / "sbert_group80.npy")
    TF = np.load(DERIVED / "tfidf_group150.npy")
    LN = np.load(DERIVED / "length_group150.npy")
    tt = json.loads((DERIVED / "texttiling_full.json").read_text(encoding="utf-8"))

    w("=" * 78)
    w("ROBUSTNESS CHECK ACROSS ALL REPRESENTATIONS")
    w("=" * 78)
    w()

    # ---- 1. length confound, before and after ---------------------------
    w("1. NCD LENGTH CONFOUND, BEFORE AND AFTER THE FIX")
    w("   (Pearson r of the NCD matrix with |compressed-size difference|)")
    for lab, M in (("NCD group150 (original)", NCD_g),
                   ("NCD residual (statistical fix)", None),
                   ("NCD equal150 (structural fix)", NCD_eq)):
        if M is None:
            continue
        n = M.shape[0]
        iu, ju = np.triu_indices(n, k=1)
        d = M[iu, ju]
        # size difference needs the sizes; recompute for equal-length units
        if lab.startswith("NCD equal150"):
            sizes = np.array([len(u.text.encode("utf-8")) for u in Ueq], float)
        else:
            import ncd as NCDmod
            sizes = NCDmod.ncd_matrix(units150, "zlib").sizes
        sd = np.abs(sizes[iu] - sizes[ju])
        w(f"   {lab:<34} r = {np.corrcoef(d, sd)[0,1]:+.3f}")
    n = NCD_g.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    import ncd as NCDmod
    sizes = NCDmod.ncd_matrix(units150, "zlib").sizes
    sd = np.abs(sizes[iu] - sizes[ju])
    w(f"   NCD residual (by construction)   r = "
      f"{np.corrcoef(NCD_res[iu, ju], sd)[0,1]:+.3f}")
    w()

    # ---- 2. method agreement, fixed representations ---------------------
    w("2. METHOD AGREEMENT AFTER THE FIXES (upper-triangle Pearson)")
    names = ["NCD-eq(1-d)", "SBERT-pooled", "SBERT-g80", "TF-IDF-eq", "length-eq"]
    TFeq = BL.tfidf_matrix(Ueq)
    LNeq = BL.length_similarity(Ueq)
    mats = {"NCD-eq(1-d)": 1 - NCD_eq, "SBERT-pooled": SB_po,
            "SBERT-g80": SB_80, "TF-IDF-eq": TFeq, "length-eq": LNeq}
    w("   " + "".join(f"{k:>14}" for k in names))
    for a in names:
        row = ""
        for b in names:
            if a == b:
                row += f"{1.0:>14.3f}"
            elif mats[a].shape == mats[b].shape:
                row += f"{BL.compare_matrices(mats[a], mats[b])['pearson']:>14.3f}"
            else:
                row += f"{'n/a':>14}"
        w(f"   {a:<11}{row}")
    w()
    w("   (NCD-eq and SBERT-pooled use different unit sets: n=105 vs n=78,")
    w("    so their cross-correlation is n/a; each is compared with its own")
    w("    lexical baseline on the same units.)")
    w()

    # ---- 3. which specific pairs survive everywhere? --------------------
    w("3. PAIR-LEVEL ROBUSTNESS (150-word units)")
    w("   For each candidate pair: raw rank and hub-corrected rank under each")
    w("   representation.  rank 1 = most similar of all pairs.")
    pairs = [(2, 73, "'snow' frame — the eye-catching but fragile one"),
             (30, 39, "religion / Church thread (pope vs. monks)"),
             (55, 58, "song lyric vs. song name"),
             (69, 72, "Gabriel asks / Gretta's revelation"),
             (37, 58, "singers past vs. song name"),
             (12, 16, "Mr Browne & Freddy (punch / lemonade)"),
             (71, 74, "cold interrogation vs. Gretta sobbing")]
    for i, j, desc in pairs:
        w(f"   units {i}&{j}  — {desc}")
        for lab, S in (("SBERT truncated", SB_tr), ("SBERT pooled", SB_po)):
            rm, R = hub_resid(S)
            w(f"      {lab:<16} raw rank {rank_of(S, i, j):>4}/3003   "
              f"hub-corrected rank {rank_of(R, i, j):>4}/3003   "
              f"cos={S[i, j]:.3f}")
        for lab, M in (("NCD group150", NCD_g), ("NCD residual", NCD_res)):
            w(f"      {lab:<16} raw rank {rank_of(M, i, j):>4}/3003   "
              f"(lower = more similar)   NCD={M[i, j]:.3f}")
        w()

    # ---- 4. TextTiling vs. the FIXED similarity matrices ----------------
    w("4. TextTiling STRONGEST BOUNDARIES vs. THE FIXED MATRICES")
    def unit_at(pos):
        for u in units150:
            if u.start <= pos <= u.end:
                return u.index
        return None
    topk = sorted(tt["main"]["boundaries"], key=lambda b: -b["depth"])[:12]
    gaps = []
    for b in topk:
        k = unit_at(b["char_pos"])
        if k is None:
            continue
        if k - 1 >= 0:
            gaps.append((k - 1, k))
        if k + 1 < len(units150):
            gaps.append((k, k + 1))
    w(f"   {len(topk)} strongest boundaries -> {len(gaps)} straddled gaps")
    for lab, M, lower in (("NCD group150", NCD_g, True),
                          ("NCD residual", NCD_res, True),
                          ("SBERT truncated", SB_tr, False),
                          ("SBERT pooled", SB_po, False),
                          ("TF-IDF", TF, False)):
        alladj = np.array([M[i, i + 1] for i in range(len(units150) - 1)])
        vals = np.array([M[i, j] for i, j in gaps])
        pct = float(np.mean([(alladj < v).mean() for v in vals]))
        w(f"   {lab:<18} mean at boundaries {vals.mean():+.3f} | "
          f"all adjacent {alladj.mean():+.3f} | mean percentile {pct*100:.0f}%")
    w("   (50% = no relation to the boundaries; the fixes do not change this.)")
    w()

    # ---- 5. which units are peripheral, across all settings -------------
    w("5. PERIPHERAL UNITS ACROSS SETTINGS (1 = least similar to the rest)")
    for lab, S, units in (("SBERT truncated g150", SB_tr, units150),
                          ("SBERT pooled g150", SB_po, units150),
                          ("SBERT g80", SB_80, U80)):
        rm, _ = hub_resid(S)
        order = np.argsort(rm)[:5]
        w(f"   {lab:<22}: " + ", ".join(
            f"u{i}(paras {units[i].first_para}-{units[i].last_para})"
            f"[{rm[i]:.3f}]" for i in order))
    w()
    w("   monastery units (contain 'monk'/'monastery'):")
    for lab, S, units in (("SBERT truncated g150", SB_tr, units150),
                          ("SBERT pooled g150", SB_po, units150),
                          ("SBERT g80", SB_80, U80)):
        rm, _ = hub_resid(S)
        rk = {i: int((rm < rm[i]).sum()) + 1 for i in range(len(units))}
        monks = [u.index for u in units
                 if any(k in u.text.lower() for k in ("monk", "monastery"))]
        w(f"      {lab:<22}: " + ", ".join(
            f"u{i} rank {rk[i]}/{len(units)}" for i in monks))
    w()

    (DERIVED / "robustness_check.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "robustness_check.txt")


if __name__ == "__main__":
    main()
