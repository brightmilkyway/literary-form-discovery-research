"""Adjudicate the (2, 73) 'snow frame' pair.

The pair unit 2 (paras 5-9: Gabriel at the door, snow on his goloshes, cold air
from out-of-doors) and unit 73 (paras 442-447: Gretta's memory - the wet window,
Michael Furey shivering in the garden) has the highest raw cosine among
long-range pairs, but its hub-corrected rank collapses.  Which is it?

  (i)   a real echo that hub correction over-corrects
  (ii)  a pure hub artifact ("everything resembles Gretta's revelation")
  (iii) invisible to the encoder (the similarity is not semantic at all)

Four independent tests
----------------------
A. Long-range-restricted percentile.  The fair baseline for a claim about
   *distant* echoes is other *distant* pairs, not all pairs.
B. Leave-one-out hub correction.  Recompute the row means with the pair itself
   excluded, so the pair cannot inflate its own hub estimate.
C. Cross-chunking replication.  Re-cut the text into 80- and 100-word units and
   ask whether the *passage block* paras 5-9 still matches paras 442-447.
D. Cross-model replication.  Re-encode with a second encoder from a different
   family (BAAI/bge-small-en-v1.5, 512-token context) and repeat A and B.
E. Lexical ablation.  Delete the weather/outdoor lexicon from both passages and
   re-encode: if the link is the weather semantics, it should collapse; if it
   survives, the link is not what it appears to be.

Writes ``data/derived/snow_adjudication.txt``.
"""

from __future__ import annotations

import json
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []
PAIR = (2, 73)                      # 150-word units
EARLY_PARAS = (5, 9)                # what unit 2 covers
LATE_PARAS = (442, 447)             # what unit 73 covers
WEATHER = {
    "snow", "snowing", "snow-stiffened", "snowy", "cold", "wet", "rain",
    "window", "garden", "shivering", "shiver", "outside", "out-of-doors",
    "air", "dew", "damp", "fragrant", "chilled", "cool",
}


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def prev(t, n=140):
    s = re.sub(r"\s+", " ", t).strip()
    return s[:n] + ("…" if len(s) > n else "")


def hub(S, leave_out=None):
    """Row/column-centred residual.  ``leave_out`` excludes a pair from the
    row means so a pair cannot inflate its own hub estimate."""
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    if leave_out is not None:
        i, j = leave_out
        off[i, j] = off[j, i] = False
    rm = np.array([S[i, off[i]].mean() for i in range(n)])
    R = S - rm[:, None] - rm[None, :] + S[off].mean()
    np.fill_diagonal(R, -np.inf)
    return rm, R


def pct_of(M, i, j, restrict=None):
    """Percentile of M[i,j] within `restrict` (a boolean mask) or the whole
    upper triangle.  Higher = more extreme in the 'more similar' direction."""
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = M[iu, ju]
    if restrict is not None:
        vals = vals[restrict]
    v = M[i, j]
    return float((vals < v).mean()), len(vals)


def rank_of(M, i, j, restrict=None):
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = M[iu, ju]
    if restrict is not None:
        vals = vals[restrict]
    return int((vals > M[i, j]).sum()) + 1, len(vals)


def unit_block(units, p0, p1):
    """Units whose paragraph span overlaps [p0, p1]."""
    return [u.index for u in units if u.last_para >= p0 and u.first_para <= p1]


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units150 = C.build_units(body, "group", target_words=150)
    SB = np.load(DERIVED / "sbert_group150_pooled.npy")   # the fixed matrix
    SB_tr = np.load(DERIVED / "sbert_group150.npy")       # the truncated one
    n = SB.shape[0]
    i, j = PAIR

    iu, ju = np.triu_indices(n, k=1)
    gap = ju - iu
    LONG = gap >= 40            # "distant" pairs
    VLONG = gap >= 60

    w("=" * 78)
    w(f"ADJUDICATING THE ({i}, {j}) 'SNOW FRAME' PAIR")
    w("=" * 78)
    w()
    w(f"unit {i}  (paras {units150[i].first_para}-{units150[i].last_para}, "
      f"{units150[i].n_words} words):")
    w(f"   {prev(units150[i].text, 300)}")
    w(f"unit {j}  (paras {units150[j].first_para}-{units150[j].last_para}, "
      f"{units150[j].n_words} words):")
    w(f"   {prev(units150[j].text, 300)}")
    w()

    # ------------------------------------------------------------------
    w("-" * 78)
    w("TEST A — IS IT EXTREME AMONG *DISTANT* PAIRS, BEFORE AND AFTER")
    w("         HUB CORRECTION?")
    w("-" * 78)
    _, R = hub(SB)
    for lab, M in (("raw cos", SB), ("hub-corrected", R)):
        for name, mask in (("all pairs", None),
                           ("gap>=40", LONG), ("gap>=60", VLONG)):
            r, tot = rank_of(M, i, j, mask)
            p, _ = pct_of(M, i, j, mask)
            w(f"   {lab:<14} {name:<10} rank {r:>4}/{tot:<5} "
              f"percentile {p*100:5.1f}%")
    w()
    w("   Interpretation: a *high* percentile restricted to distant pairs is")
    w("   the fair test for a claim about a distant echo.")
    w()

    # ------------------------------------------------------------------
    w("-" * 78)
    w("TEST B — LEAVE-ONE-OUT HUB CORRECTION")
    w("         (row means recomputed with the pair itself excluded)")
    w("-" * 78)
    rm_lo, R_lo = hub(SB, leave_out=PAIR)
    rm_full, R_full = hub(SB)
    w(f"   centrality of unit {i}: {rm_lo[i]:.4f} (leave-one-out) vs "
      f"{rm_full[i]:.4f} (full)")
    w(f"   centrality of unit {j}: {rm_lo[j]:.4f} (leave-one-out) vs "
      f"{rm_full[j]:.4f} (full)")
    w(f"   residual of the pair, leave-one-out = {R_lo[i, j]:+.4f}; "
      f"full = {R_full[i, j]:+.4f}")
    for lab, M in (("leave-one-out residual", R_lo), ("full residual", R_full)):
        r, tot = rank_of(M, i, j, LONG)
        w(f"   {lab:<24} rank among gap>=40: {r}/{tot}")
    w()

    # ------------------------------------------------------------------
    w("-" * 78)
    w("TEST C — CROSS-CHUNKING REPLICATION (does the *passage block* recur?)")
    w("-" * 78)
    for scheme, kw in (("group100", dict(target_words=100)),
                       ("group80", dict(target_words=80))):
        units = C.build_units(body, "group", **kw)
        S = np.load(DERIVED / f"sbert_{scheme}.npy")
        _, R2 = hub(S)
        E = unit_block(units, *EARLY_PARAS)
        L = unit_block(units, *LATE_PARAS)
        nn = S.shape[0]
        e_all = [u.index for u in units if u.last_para <= 60]
        l_all = [u.index for u in units if u.first_para >= 400]
        w(f"   [{scheme}] n={nn}")
        w(f"      units covering paras {EARLY_PARAS[0]}-{EARLY_PARAS[1]}: {E}")
        w(f"      units covering paras {LATE_PARAS[0]}-{LATE_PARAS[1]}: {L}")
        if not E or not L:
            w("      (no units) ")
            continue
        # distribution of all early x late pairs
        raw_vals, res_vals = [], []
        for a in e_all:
            for b in l_all:
                raw_vals.append(S[a, b])
                res_vals.append(R2[a, b])
        raw_vals = np.array(raw_vals)
        res_vals = np.array(res_vals)
        for a in E:
            for b in L:
                pr = float((raw_vals < S[a, b]).mean()) * 100
                pres = float((res_vals < R2[a, b]).mean()) * 100
                w(f"      pair ({a},{b}) [paras {units[a].first_para}-"
                  f"{units[a].last_para} x {units[b].first_para}-"
                  f"{units[b].last_para}]  cos={S[a,b]:.3f} "
                  f"(pct among earlyxlate {pr:.0f}%)   resid={R2[a,b]:+.3f} "
                  f"(pct {pres:.0f}%)")
        w(f"      baseline earlyxlate pairs: {len(raw_vals)}, mean cos "
          f"{raw_vals.mean():.3f}, mean resid {res_vals.mean():+.3f}")
        w()

    # ------------------------------------------------------------------
    w("-" * 78)
    w("TEST D — CROSS-MODEL REPLICATION")
    w("         (a second encoder from a different family)")
    w("-" * 78)
    model2 = "BAAI/bge-small-en-v1.5"
    S2 = None
    try:
        import torch
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer

        m2 = SentenceTransformer(model2)
        m2.eval()
        tk2 = AutoTokenizer.from_pretrained(model2)
        paras = C.paragraphs(body)
        sub, owner = [], []
        for u in units150:
            buf, cnt = [], 0
            for p in paras[u.first_para:u.last_para + 1]:
                if cnt + p.n_words > 130 and buf:
                    sub.append("\n\n".join(buf))
                    owner.append(u.index)
                    buf, cnt = [], 0
                buf.append(p.text)
                cnt += p.n_words
            if buf:
                sub.append("\n\n".join(buf))
                owner.append(u.index)
        lens = [len(tk2(t, add_special_tokens=True)["input_ids"]) for t in sub]
        w(f"   {model2}: max_seq_length={m2.max_seq_length}, "
          f"{len(sub)} sub-chunks, max {max(lens)} tokens, "
          f"over limit {sum(1 for x in lens if x > m2.max_seq_length)}")
        with torch.no_grad():
            e2 = m2.encode(sub, batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        emb2 = np.zeros((len(units150), e2.shape[1]))
        for k, o in enumerate(owner):
            emb2[o] += e2[k]
        emb2 /= np.linalg.norm(emb2, axis=1, keepdims=True)
        S2 = emb2 @ emb2.T
        np.fill_diagonal(S2, 1.0)
        np.save(DERIVED / "sbert_group150_bge.npy", S2)
        _, R2m = hub(S2)
        w(f"   cos({i},{j}) = {S2[i,j]:.3f}")
        for lab, M in (("raw cos", S2), ("hub-corrected", R2m)):
            for name, mask in (("all pairs", None), ("gap>=40", LONG)):
                r, tot = rank_of(M, i, j, mask)
                p, _ = pct_of(M, i, j, mask)
                w(f"   {lab:<14} {name:<10} rank {r:>4}/{tot:<5} "
                  f"percentile {p*100:5.1f}%")
        w(f"   agreement of the two models' matrices (upper triangle Pearson): "
          f"{np.corrcoef(SB[iu, ju], S2[iu, ju])[0,1]:+.3f}")
    except Exception as e:  # noqa: BLE001
        w(f"   FAILED to run the second model: {e!r}")
    w()

    # ------------------------------------------------------------------
    w("-" * 78)
    w("TEST E — LEXICAL ABLATION (is the link the weather semantics?)")
    w("-" * 78)
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        m1 = SentenceTransformer("all-MiniLM-L6-v2")
        m1.eval()

        def ablate(t):
            return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                          lambda m: "thing" if m.group(0).lower().strip("'")
                          in WEATHER else m.group(0), t)

        base = [units150[i].text, units150[j].text]
        abl = [ablate(t) for t in base]
        with torch.no_grad():
            eb = m1.encode(base, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
            ea = m1.encode(abl, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        cos_b = float(eb[0] @ eb[1])
        cos_a = float(ea[0] @ ea[1])
        w(f"   cos({i},{j}) before ablation = {cos_b:.3f}")
        w(f"   cos({i},{j}) after  ablation = {cos_a:.3f}   "
          f"(drop {cos_b - cos_a:+.3f})")
        w(f"   words replaced in unit {i}: "
          f"{sorted({t for t in re.findall(r'[A-Za-z]+', units150[i].text) if t.lower() in WEATHER})}")
        w(f"   words replaced in unit {j}: "
          f"{sorted({t for t in re.findall(r'[A-Za-z]+', units150[j].text) if t.lower() in WEATHER})}")
        # control: the same ablation applied to the other top distant pairs
        ctrl = [(30, 39), (55, 58), (12, 16), (71, 74), (37, 58)]
        w("   control pairs (same ablation):")
        for (a, b) in ctrl:
            with torch.no_grad():
                x = m1.encode([units150[a].text, units150[b].text],
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)
                y = m1.encode([ablate(units150[a].text), ablate(units150[b].text)],
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)
            w(f"      ({a},{b}) gap {b-a:>2}: cos {float(x[0]@x[1]):.3f} -> "
              f"{float(y[0]@y[1]):.3f}  (drop {float(x[0]@x[1]) - float(y[0]@y[1]):+.3f})")
    except Exception as e:  # noqa: BLE001
        w(f"   FAILED: {e!r}")
    w()

    (DERIVED / "snow_adjudication.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "snow_adjudication.txt")


if __name__ == "__main__":
    main()
