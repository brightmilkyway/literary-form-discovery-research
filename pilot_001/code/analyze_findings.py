"""Turn the derived matrices into a readable digest + candidate-finding tables.

Reads only the artifacts in ``data/derived`` (written by ``run_all.py``) and
writes:

* ``data/derived/findings_digest.txt`` — human-readable summary
* ``results/tables/*.csv``            — the pairs/boundaries behind it

Run from code/:  python analyze_findings.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def load(name):
    p = DERIVED / name
    return np.load(p) if p.exists() else None


def prev(text, width=150):
    import re
    s = re.sub(r"\s+", " ", text).strip()
    return s[:width] + ("…" if len(s) > width else "")


def top_sim_pairs(S, units, top=12, min_gap=3):
    n = len(units)
    pairs = [(float(S[i, j]), i, j) for i in range(n)
             for j in range(i + min_gap, n)]
    pairs.sort(key=lambda t: -t[0])
    return pairs[:top]


def top_dist_pairs(D, units, top=12, min_gap=3):
    n = len(units)
    pairs = [(float(D[i, j]), i, j) for i in range(n)
             for j in range(i + min_gap, n)]
    pairs.sort(key=lambda t: t[0])
    return pairs[:top]


def agreement(A, B):
    n = A.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    x, y = A[iu, ju], B[iu, ju]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    units = json.loads((DERIVED / "units.json").read_text(encoding="utf-8"))["group150"]
    body = C.strip_title(C.load_raw())[0]

    NCD = load("ncd_group150_zlib.npy")
    SB = load("sbert_group150.npy")
    TF = load("tfidf_group150.npy")
    JA = load("jaccard_group150.npy")
    LN = load("length_group150.npy")

    w("=" * 78)
    w("PILOT 001 — FINDINGS DIGEST  (seed=42, 150-word paragraph groups, n=78)")
    w("=" * 78)
    w()

    # ---- 1. method agreement --------------------------------------------
    w("1. HOW MUCH DO THE METHODS AGREE?  (Pearson r on upper triangles)")
    mats = {"NCD(1-d)": 1 - NCD, "SBERT": SB, "TF-IDF": TF,
            "Jaccard": JA, "length": LN}
    names = list(mats)
    w("        " + "".join(f"{k:>10}" for k in names))
    for a in names:
        row = "".join(f"{agreement(mats[a], mats[b]):>10.3f}" for b in names)
        w(f"{a:>8}{row}")
    w()

    # ---- 2. adjacency / length confound ---------------------------------
    w("2. CONFOUNDS: similarity vs. index distance, and vs. length")
    def prof(M, invert=False):
        n = M.shape[0]
        out = {}
        for i in range(n):
            for j in range(i + 1, n):
                out.setdefault(j - i, []).append(M[i, j])
        return {g: float(np.mean(v)) for g, v in sorted(out.items())}
    pn = prof(NCD); ps = prof(SB); pt = prof(TF)
    w("  gap :   mean NCD   mean SBERT   mean TF-IDF")
    for g in [1, 2, 3, 5, 8, 13, 21, 34, 55, 77]:
        if g in pn:
            w(f"  {g:>3} :   {pn[g]:>8.3f}   {ps[g]:>9.3f}   {pt[g]:>9.3f}")
    w("  (NCD: distance, lower = more similar.  SBERT/TF-IDF: cosine, higher = more similar.)")
    w()

    # ---- 3. NCD top non-adjacent pairs ----------------------------------
    w("3. NCD — most similar NON-ADJACENT pairs (zlib, gap>=3)")
    rows = []
    for d, i, j in top_dist_pairs(NCD, units, top=12):
        w(f"  d={d:.3f}  units {i:>2}&{j:>2} (gap {j-i:>2}, "
          f"paras {units[i]['first_para']}-{units[i]['last_para']} / "
          f"{units[j]['first_para']}-{units[j]['last_para']})")
        w(f"      i: {prev(body[units[i]['start']:units[i]['end']], 130)}")
        w(f"      j: {prev(body[units[j]['start']:units[j]['end']], 130)}")
        rows.append({"distance": round(d, 4), "unit_i": i, "unit_j": j,
                     "gap": j - i, "i_text": prev(body[units[i]['start']:units[i]['end']], 200),
                     "j_text": prev(body[units[j]['start']:units[j]['end']], 200)})
    _csv(rows, C.TABLE_DIR / "ncd_top_nonadjacent_pairs.csv")
    w()

    # ---- 4. SBERT top non-adjacent pairs --------------------------------
    w("4. SBERT — most similar NON-ADJACENT pairs (all-MiniLM-L6-v2, gap>=3)")
    rows = []
    for s, i, j in top_sim_pairs(SB, units, top=12):
        w(f"  cos={s:.3f}  units {i:>2}&{j:>2} (gap {j-i:>2}, "
          f"paras {units[i]['first_para']}-{units[i]['last_para']} / "
          f"{units[j]['first_para']}-{units[j]['last_para']})")
        w(f"      i: {prev(body[units[i]['start']:units[i]['end']], 130)}")
        w(f"      j: {prev(body[units[j]['start']:units[j]['end']], 130)}")
        rows.append({"similarity": round(s, 4), "unit_i": i, "unit_j": j,
                     "gap": j - i, "i_text": prev(body[units[i]['start']:units[i]['end']], 200),
                     "j_text": prev(body[units[j]['start']:units[j]['end']], 200)})
    _csv(rows, C.TABLE_DIR / "sbert_top_nonadjacent_pairs.csv")
    w()

    # ---- 5. agreement on the SAME pair set ------------------------------
    w("5. DO NCD AND SBERT AGREE ON *WHICH* DISTANT PAIRS STAND OUT?")
    a = set((i, j) for _, i, j in top_sim_pairs(SB, units, top=20))
    b = set((i, j) for _, i, j in top_dist_pairs(NCD, units, top=20))
    c = set((i, j) for _, i, j in top_sim_pairs(TF, units, top=20))
    w(f"  top-20 SBERT ∩ top-20 NCD   = {len(a & b)}  {sorted(a & b)}")
    w(f"  top-20 SBERT ∩ top-20 TFIDF = {len(a & c)}  {sorted(a & c)}")
    w(f"  top-20 NCD   ∩ top-20 TFIDF = {len(b & c)}  {sorted(b & c)}")
    w()

    # ---- 6. TextTiling strongest boundaries -----------------------------
    tt = json.loads((DERIVED / "texttiling_full.json").read_text(encoding="utf-8"))
    w("6. TextTiling (w=20,k=6) — the 15 STRONGEST candidate boundaries")
    topb = sorted(tt["main"]["boundaries"], key=lambda b: -b["depth"])[:15]
    rows = []
    for b in topb:
        cp = b["char_pos"]
        w(f"  depth={b['depth']:.3f}  word~{b['word_pos']:>5}  unit "
          f"{b['left_unit']}->{b['right_unit']}")
        w(f"      {prev(body[cp:cp + 300], 150)}")
        rows.append({"depth": round(b["depth"], 4), "word_pos": b["word_pos"],
                     "left_unit": b["left_unit"], "right_unit": b["right_unit"],
                     "quote": prev(body[cp:cp + 300], 200)})
    _csv(rows, C.TABLE_DIR / "texttiling_strongest_boundaries.csv")
    w()

    # ---- 7. do TextTiling boundaries sit at low-similarity gaps? --------
    w("7. RELATION: the STRONGEST TextTiling boundaries vs. NCD/SBERT")
    n = len(units)

    def unit_at(pos):
        for u in units:
            if u["start"] <= pos <= u["end"]:
                return u["index"]
        return None

    topk = sorted(tt["main"]["boundaries"], key=lambda b: -b["depth"])[:12]
    gaps = []          # adjacent unit pairs straddling a strong boundary
    for b in topk:
        k = unit_at(b["char_pos"])
        if k is None:
            continue
        if k - 1 >= 0:
            gaps.append((k - 1, k))
        if k + 1 < n:
            gaps.append((k, k + 1))
    w(f"  {len(topk)} strongest boundaries -> {len(gaps)} straddled adjacent gaps")
    for lab, M, lower_is_similar in (("NCD", NCD, True), ("SBERT", SB, False),
                                     ("TF-IDF", TF, False)):
        alladj = np.array([M[i, i + 1] for i in range(n - 1)])
        vals = np.array([M[i, j] for i, j in gaps])
        pct = [float((alladj < v).mean()) for v in vals]
        w(f"  {lab}: at strong boundaries mean={vals.mean():.3f}"
          f"  |  all adjacent pairs mean={alladj.mean():.3f}"
          f"  |  mean percentile={np.mean(pct) * 100:.0f}%")
    w()
    w("  (For NCD lower = more similar; for SBERT/TF-IDF higher = more similar.")
    w("   If TextTiling tracks the same cohesion as the other methods, its")
    w("   boundaries should sit at LOW SBERT/TF-IDF percentile and HIGH NCD")
    w("   percentile.  50% = no relation.)")
    w()

    # ---- 8. which units are hubs / islands? ------------------------------
    w("8. UNIT CENTRALITY (mean similarity to all other units)")
    for lab, M, invert in (("SBERT", SB, False), ("NCD", NCD, True)):
        n = len(units)
        rowmean = (M.sum(axis=1) - np.diag(M)) / (n - 1)
        order = np.argsort(rowmean) if invert else np.argsort(-rowmean)
        w(f"  {lab} — most similar to the rest (top 5): "
          f"{[(int(i), round(float(rowmean[i]), 3)) for i in order[:5]]}")
        w(f"  {lab} — least similar to the rest (bottom 5): "
          f"{[(int(i), round(float(rowmean[i]), 3)) for i in order[-5:]]}")
    w()
    w("  unit 39 (paras 227-233, Mr Browne and the monks) is the least")
    w("  semantically central unit:  "
      f"{prev(body[units[39]['start']:units[39]['end']], 120)}")
    w()

    # ---- 9. hub-corrected residuals --------------------------------------
    w("9. HUB-CORRECTED PAIRS: similarity minus what the two units' general")
    w("   'centrality' predicts (two-way residual).  Controls for the fact")
    w("   that some units are similar to everything.")
    for lab, M, invert in (("SBERT", SB, False), ("NCD", NCD, True)):
        S = -M if invert else M
        n = len(units)
        off = ~np.eye(n, dtype=bool)
        rm = np.array([S[i, off[i]].mean() for i in range(n)])
        gm = S[off].mean()
        RES = S - rm[:, None] - rm[None, :] + gm
        np.fill_diagonal(RES, -np.inf)
        cand = [(float(RES[i, j]), i, j) for i in range(n)
                for j in range(i + 3, n)]
        cand.sort(key=lambda t: -t[0])
        w(f"  {lab} — top 8 hub-corrected pairs (gap>=3):")
        for v, i, j in cand[:8]:
            w(f"    resid={v:+.3f}  units {i}&{j} (gap {j-i})"
              f"  [raw {'NCD-dist' if invert else 'cos'}="
              f"{(NCD[i, j] if invert else SB[i, j]):.3f}]")
            w(f"        {i}: {prev(body[units[i]['start']:units[i]['end']], 110)}")
            w(f"        {j}: {prev(body[units[j]['start']:units[j]['end']], 110)}")
        if lab == "SBERT":
            w(f"  centrality of unit 72 = {rm[72]:.3f}, unit 73 = {rm[73]:.3f} "
              f"(mean {gm:.3f})")
        w()

    txt = "\n".join(OUT)
    (DERIVED / "findings_digest.txt").write_text(txt, encoding="utf-8")
    print("wrote", DERIVED / "findings_digest.txt")


def _csv(rows, path: Path):
    import csv
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
