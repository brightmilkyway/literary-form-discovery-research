"""Fix the two representation-level defects found in the main run.

Defect 1 (NCD): the distance matrix correlates ~0.66-0.69 with unit *length*,
so "similarity" is partly just "these two chunks are the same size".
Fix A1 — structural: use **equal-word-count** units (non-overlapping 150-word
         windows), so every unit has the same word count and the length
         baseline degenerates to a constant.
Fix A2 — statistical: regress the existing NCD matrix on the compressed-size
         difference and re-rank pairs by the **residual** NCD.

Defect 2 (SBERT): 25/78 units (32%) exceeded the encoder's 256-token limit and
were silently truncated.
Fix B1 — pool: split each unit at paragraph boundaries into sub-chunks of
         <=130 words (~175 tokens), encode each, mean-pool the normalised
         embeddings.  Truncation should become 0.
Fix B2 — independent check: rerun at 80-word units.

Writes ``data/derived/representation_fixes.txt`` plus new matrices.
Run from code/:  python fix_representations.py
"""

from __future__ import annotations

import json
import re

import numpy as np

import common as C
import ncd as NCD
import baselines as BL

DERIVED = C.DATA_DIR / "derived"
OUT = []
SUBCHUNK_WORDS = 130


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def prev(t, n=120):
    s = re.sub(r"\s+", " ", t).strip()
    return s[:n] + ("…" if len(s) > n else "")


def unit_at(units, pos):
    for u in units:
        if u.start <= pos <= u.end:
            return u.index
    return None


# --------------------------------------------------------------------------
# NCD fixes
# --------------------------------------------------------------------------

def ncd_fixes(body):
    w("=" * 78)
    w("FIX A — NCD LENGTH CONFOUND")
    w("=" * 78)
    w()

    U150 = C.build_units(body, "group", target_words=150)
    Ueq = C.build_units(body, "window", window=150, stride=150)  # equal word count
    wl = [u.n_words for u in Ueq]
    cl = [len(u.text.encode('utf-8')) for u in Ueq]
    w(f"A1. equal-word-count units: n={len(Ueq)}, word counts "
      f"{min(wl)}-{max(wl)} (constant), compressed byte sizes "
      f"{min(cl)}-{max(cl)} (still varies with word length)")
    w()

    res_eq = NCD.ncd_matrix(Ueq, "zlib")
    np.save(DERIVED / "ncd_equal150_zlib.npy", res_eq.matrix)
    conf_eq = NCD.length_confound(res_eq)
    w("A1 result — length confound of the equal-length NCD:")
    for k, v in conf_eq.items():
        w(f"    {k} = {v:+.3f}")

    # baseline: original group150 NCD for comparison
    res_g = NCD.ncd_matrix(U150, "zlib")
    conf_g = NCD.length_confound(res_g)
    w("    (for comparison, the original group150 NCD:)")
    for k, v in conf_g.items():
        w(f"    {k} = {v:+.3f}")
    w()
    w("A1 top-12 closest NON-ADJACENT pairs on the equal-length NCD:")
    for p in NCD.non_adjacent_pairs(res_eq, top=12, min_gap=3):
        w(f"    d={p['distance']:.3f}  units {p['i']}&{p['j']} "
          f"(gap {p['gap']}, paras {Ueq[p['i']].first_para}-{Ueq[p['i']].last_para}"
          f" / {Ueq[p['j']].first_para}-{Ueq[p['j']].last_para})")
        w(f"        {p['i']}: {p['i_preview']}")
        w(f"        {p['j']}: {p['j_preview']}")
    w()

    # ---- A2: statistical residualisation --------------------------------
    M = res_g.matrix
    n = len(U150)
    iu, ju = np.triu_indices(n, k=1)
    d = M[iu, ju]
    size_diff = np.abs(res_g.sizes[iu] - res_g.sizes[ju])
    b, a = np.polyfit(size_diff, d, 1)
    pred = a + b * size_diff
    resid = d - pred
    w(f"A2. regress NCD on compressed-size difference: "
      f"D = {a:.4f} + {b:.6f} * |size_i - size_j|   "
      f"(slope b = {b:.6f})")
    w(f"    variance explained by size difference alone: "
      f"{np.corrcoef(size_diff, d)[0, 1] ** 2:.3f}")
    w()
    R = np.full((n, n), np.nan)
    R[iu, ju] = resid
    R[ju, iu] = resid
    np.fill_diagonal(R, 0.0)
    np.save(DERIVED / "ncd_group150_zlib_residual.npy", R)

    cand = [(float(R[i, j]), i, j) for i in range(n) for j in range(i + 3, n)]
    cand.sort(key=lambda t: t[0])
    w("A2 top-12 pairs by RESIDUAL NCD (most similar after removing length):")
    rows = []
    for v, i, j in cand[:12]:
        sd = abs(res_g.sizes[i] - res_g.sizes[j])
        w(f"    resid={v:+.4f}  rawNCD={M[i, j]:.3f}  size_diff={sd:>5.0f}  "
          f"units {i}&{j} (gap {j-i})")
        w(f"        {i}: {prev(U150[i].text)}")
        w(f"        {j}: {prev(U150[j].text)}")
        rows.append({"residual": round(v, 4), "raw_ncd": round(float(M[i, j]), 4),
                     "size_diff": int(sd), "unit_i": i, "unit_j": j,
                     "gap": j - i})
    _csv(rows, C.TABLE_DIR / "ncd_residual_top_pairs.csv")

    # ---- did the 'song' cluster survive? --------------------------------
    w()
    w("A3. DID THE 'SONG' CLUSTER SURVIVE THE LENGTH FIX?")
    song_units = {37, 55, 58, 69, 72, 73}
    # group150 pairs
    g_pairs = [(37, 55), (37, 58), (55, 58), (69, 72), (69, 73)]
    for (i, j) in g_pairs:
        rank_raw = _rank(M, i, j, n)
        rank_res = _rank(R, i, j, n)
        w(f"    group150 units {i}&{j}: raw NCD rank {rank_raw}, "
          f"residual NCD rank {rank_res} (of {n*(n-1)//2} pairs)")
    w()
    # equal-length: map song text to the equal-length units that contain it
    w("    equal-length (150w) units whose text overlaps the song passages:")
    for target in (37, 55, 58, 69, 72, 73):
        lo, hi = U150[target].start, U150[target].end
        hits = [u.index for u in Ueq if u.start < hi and u.end > lo]
        w(f"      group150 unit {target} (paras {U150[target].first_para}-"
          f"{U150[target].last_para}) -> equal-length units {hits}")
    w()
    eq_pairs = set()
    for a_ in (37, 55, 58, 69, 72, 73):
        lo, hi = U150[a_].start, U150[a_].end
        hits = [u.index for u in Ueq if u.start < hi and u.end > lo]
        for x in hits:
            for y in hits:
                if x < y:
                    eq_pairs.add((x, y))
    ranks = sorted(_rank(res_eq.matrix, i, j, len(Ueq)) for (i, j) in eq_pairs)
    w(f"    ranks of the equal-length units that carry the song passages: "
      f"{ranks[:12]}{' …' if len(ranks) > 12 else ''}")
    w(f"    (median rank {int(np.median(ranks))}; lower = more similar)")
    w()


def _rank(M, i, j, n):
    """Rank of pair (i,j) among all pairs, 1 = most similar (smallest value)."""
    iu, ju = np.triu_indices(n, k=1)
    vals = M[iu, ju]
    v = M[i, j]
    return int((vals < v).sum()) + 1


# --------------------------------------------------------------------------
# SBERT fix
# --------------------------------------------------------------------------

def split_paragraphs(paras, first, last, max_words):
    """Group the unit's paragraphs into sub-chunks of <= max_words words."""
    texts = []
    buf, cnt = [], 0
    for p in paras[first:last + 1]:
        if p.n_words > max_words:
            # a single long paragraph: split on word count
            toks = p.text.split()
            for k in range(0, len(toks), max_words):
                texts.append(" ".join(toks[k:k + max_words]))
            continue
        if cnt + p.n_words > max_words and buf:
            texts.append("\n\n".join(buf))
            buf, cnt = [], 0
        buf.append(p.text)
        cnt += p.n_words
    if buf:
        texts.append("\n\n".join(buf))
    return texts


def sbert_fixes(body):
    w("=" * 78)
    w("FIX B — SBERT TRUNCATION")
    w("=" * 78)
    w()

    import torch
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    C.set_seed(42)
    name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(name)
    model.eval()
    tk = AutoTokenizer.from_pretrained(f"sentence-transformers/{name}")
    limit = int(model.max_seq_length)

    U150 = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)

    # ---- B1: paragraph-aware pooling, no truncation ----------------------
    subchunks, owner = [], []
    for u in U150:
        for t in split_paragraphs(paras, u.first_para, u.last_para, SUBCHUNK_WORDS):
            subchunks.append(t)
            owner.append(u.index)
    lens = np.array([len(tk(t, add_special_tokens=True)["input_ids"])
                     for t in subchunks])
    w(f"B1. pooling: {len(U150)} units -> {len(subchunks)} sub-chunks "
      f"(<= {SUBCHUNK_WORDS} words each)")
    w(f"    sub-chunk token lengths: min {lens.min()} median {int(np.median(lens))} "
      f"max {lens.max()};  over {limit} tokens: {int((lens > limit).sum())}")

    with torch.no_grad():
        sub = model.encode(subchunks, batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    emb = np.zeros((len(U150), sub.shape[1]), dtype=float)
    for k, o in enumerate(owner):
        emb[o] += sub[k]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    S_pool = emb @ emb.T
    np.fill_diagonal(S_pool, 1.0)
    np.save(DERIVED / "sbert_group150_pooled.npy", S_pool)

    S_old = np.load(DERIVED / "sbert_group150.npy")
    w(f"    agreement of pooled vs. truncated matrix (upper triangle Pearson): "
      f"{BL.compare_matrices(S_pool, S_old)['pearson']:+.3f}")
    w()

    # ---- B2: 80-word units ----------------------------------------------
    U80 = C.build_units(body, "group", target_words=80)
    l80 = np.array([len(tk(u.text, add_special_tokens=True)["input_ids"])
                    for u in U80])
    w(f"B2. 80-word units: n={len(U80)}; token lengths min {l80.min()} "
      f"median {int(np.median(l80))} max {l80.max()};  over {limit}: "
      f"{int((l80 > limit).sum())} ({(l80 > limit).mean()*100:.0f}%)")
    with torch.no_grad():
        e80 = model.encode([u.text for u in U80], batch_size=32,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
    S80 = e80 @ e80.T
    np.fill_diagonal(S80, 1.0)
    np.save(DERIVED / "sbert_group80.npy", S80)
    w()

    # ---- centrality + hub-corrected pairs for all three ------------------
    w("B3. CENTRALITY AND HUB-CORRECTED DISTANT PAIRS, BEFORE vs AFTER")
    w()
    for lab, S, units in (("TRUNCATED group150 (old)", S_old, U150),
                          ("POOLED group150 (fixed)", S_pool, U150),
                          ("group80 (independent)", S80, U80)):
        n = len(units)
        off = ~np.eye(n, dtype=bool)
        rm = np.array([S[i, off[i]].mean() for i in range(n)])
        gm = S[off].mean()
        RES = S - rm[:, None] - rm[None, :] + gm
        np.fill_diagonal(RES, -np.inf)
        w(f"  --- {lab} ---")
        w("    least central units:")
        for i in np.argsort(rm)[:4]:
            w(f"      unit {i:>3} central={rm[i]:.3f} paras "
              f"{units[i].first_para}-{units[i].last_para} | {prev(units[i].text, 100)}")
        w("    top-6 hub-corrected distant pairs (gap>=3):")
        cand = [(float(RES[i, j]), i, j) for i in range(n) for j in range(i + 3, n)]
        cand.sort(key=lambda t: -t[0])
        for v, i, j in cand[:6]:
            w(f"      resid={v:+.3f} cos={S[i, j]:.3f} units {i}&{j} "
              f"(gap {j-i}) paras {units[i].first_para}-{units[i].last_para}"
              f" / {units[j].first_para}-{units[j].last_para}")
            w(f"          {i}: {prev(units[i].text, 95)}")
            w(f"          {j}: {prev(units[j].text, 95)}")
        w()

    # ---- does the (2,73) 'snow' pair survive? ---------------------------
    w("B4. DOES THE 'SNOW' PAIR (2,73) SURVIVE THE TRUNCATION FIX?")
    n = len(U150)
    off = ~np.eye(n, dtype=bool)
    for lab, S in (("truncated", S_old), ("pooled", S_pool)):
        rm = np.array([S[i, off[i]].mean() for i in range(n)])
        gm = S[off].mean()
        RES = S - rm[:, None] - rm[None, :] + gm
        r = _rank(S, 2, 73, n)
        rres = _rank(RES, 2, 73, n)
        w(f"    {lab}: cos(2,73)={S[2,73]:.3f}  rank by raw cos {r}, "
          f"rank by hub-corrected residual {rres}")
    w("    (rank is out of all 3003 pairs; lower = more similar / more special)")
    w()

    # ---- song-cluster / monastery checks on the fixed representation ----
    w("B5. ARE THE MAIN SBERT CLAIMS STABLE AFTER THE FIX?")
    w("    (a) monastery passage peripherality")
    for lab, S, units in (("truncated g150", S_old, U150),
                          ("pooled g150", S_pool, U150),
                          ("group80", S80, U80)):
        n = len(units)
        off = ~np.eye(n, dtype=bool)
        rm = np.array([S[i, off[i]].mean() for i in range(n)])
        rank = {i: int((rm < rm[i]).sum()) + 1 for i in range(n)}
        monks = [u.index for u in units
                 if any(k in u.text.lower() for k in ("monk", "monastery"))]
        snow = [u.index for u in units if "the dead" in u.text.lower()]
        w(f"      {lab}: monastery units {monks} centrality ranks "
          f"{[rank[i] for i in monks]} of {n} (1 = most peripheral)")
    w()
    w("    (b) is the final snow passage peripheral too?")
    for lab, S, units in (("truncated g150", S_old, U150),
                          ("pooled g150", S_pool, U150),
                          ("group80", S80, U80)):
        n = len(units)
        off = ~np.eye(n, dtype=bool)
        rm = np.array([S[i, off[i]].mean() for i in range(n)])
        last = n - 1
        w(f"      {lab}: last unit {last} central={rm[last]:.3f}, rank "
          f"{int((rm < rm[last]).sum()) + 1} of {n}")


def _csv(rows, path):
    import csv
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    ncd_fixes(body)
    sbert_fixes(body)
    (DERIVED / "representation_fixes.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "representation_fixes.txt")


if __name__ == "__main__":
    main()
