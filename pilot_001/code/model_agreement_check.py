"""Follow-up controls for the (2,73) adjudication.

Two things the first pass could not settle:

1.  **Is bge's failure to replicate (2,73) specific, or is bge just a worse
    model?**  If bge also fails on the pairs we consider robust ((30,39),
    (55,58), (12,16), (71,74), (69,72)), then its verdict on (2,73) carries no
    weight.  If bge *replicates those but not (2,73)*, that is real evidence.
2.  **Was the lexical ablation underpowered?**  Replacing ~5 words in a
    180-word passage may not move the embedding at all.  Control: replace the
    same number of *random* content words and see whether the cosine moves.

Also reports, for both models, which findings survive.

Writes ``data/derived/model_agreement_check.txt``.
"""

from __future__ import annotations

import json
import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

CANDIDATES = [
    (2, 73, "'snow frame' — under adjudication"),
    (30, 39, "religion / Church thread"),
    (55, 58, "song lyric vs. song name"),
    (12, 16, "Mr Browne & Freddy"),
    (71, 74, "cold interrogation vs. Gretta sobbing"),
    (37, 58, "singers past vs. song name"),
    (69, 72, "Gabriel asks / Gretta's revelation"),
]


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def hub(S):
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    rm = np.array([S[i, off[i]].mean() for i in range(n)])
    R = S - rm[:, None] - rm[None, :] + S[off].mean()
    np.fill_diagonal(R, -np.inf)
    return rm, R


def stats(M, i, j, mask):
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = M[iu, ju]
    if mask is not None:
        vals = vals[mask]
    r = int((vals > M[i, j]).sum()) + 1
    return r, len(vals), float((vals < M[i, j]).mean()) * 100


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    n = len(units)
    iu, ju = np.triu_indices(n, k=1)
    gap = ju - iu
    LONG = gap >= 40

    models = {}
    models["MiniLM-L6 (384d, 256tok, pooled)"] = np.load(
        DERIVED / "sbert_group150_pooled.npy")
    models["bge-small-en-v1.5 (384d, 512tok)"] = np.load(
        DERIVED / "sbert_group150_bge.npy")

    w("=" * 78)
    w("MODEL-AGREEMENT CONTROL — is bge's verdict on (2,73) specific?")
    w("=" * 78)
    w()
    names = list(models)
    w(f"matrix agreement between the two models (upper triangle Pearson): "
      f"{np.corrcoef(models[names[0]][iu, ju], models[names[1]][iu, ju])[0,1]:+.3f}")
    w()

    w("A. EVERY CANDIDATE PAIR UNDER BOTH MODELS")
    w("   (rank among gap>=40 pairs; lower = more similar; 741 pairs)")
    w(f"   {'pair':<10}{'MiniLM raw':>12}{'MiniLM hub':>12}"
      f"{'bge raw':>10}{'bge hub':>10}   note")
    rows = []
    for (i, j, note) in CANDIDATES:
        cells = []
        for nm in names:
            S = models[nm]
            _, R = hub(S)
            r1, tot, _ = stats(S, i, j, LONG)
            r2, _, _ = stats(R, i, j, LONG)
            cells.append((r1, r2))
        w(f"   {f'{i}&{j}':<10}{cells[0][0]:>12}{cells[0][1]:>12}"
          f"{cells[1][0]:>10}{cells[1][1]:>10}   {note}")
        rows.append({"pair": f"{i}&{j}", "minilm_raw": cells[0][0],
                     "minilm_hub": cells[0][1], "bge_raw": cells[1][0],
                     "bge_hub": cells[1][1], "note": note})
    w()

    w("B. DOES bge REPLICATE THE PAIRS WE CALL ROBUST?")
    robust = [(30, 39), (55, 58), (12, 16), (71, 74)]
    w("   (criterion: hub-corrected rank within the top 10% of distant pairs,")
    w("    i.e. rank <= 74 of 741)")
    for (i, j) in robust:
        S = models[names[1]]
        _, R = hub(S)
        r_raw, _, _ = stats(S, i, j, LONG)
        r_hub, _, _ = stats(R, i, j, LONG)
        verdict = "replicated" if r_hub <= 74 else "NOT replicated"
        w(f"   ({i},{j}): bge raw {r_raw}/741, hub {r_hub}/741  -> {verdict}")
    S = models[names[1]]
    _, R = hub(S)
    r_raw, _, _ = stats(S, 2, 73, LONG)
    r_hub, _, _ = stats(R, 2, 73, LONG)
    w(f"   (2,73):  bge raw {r_raw}/741, hub {r_hub}/741  -> "
      f"{'replicated' if r_hub <= 74 else 'NOT replicated'}")
    w()

    w("C. TOP-20 HUB-CORRECTED DISTANT PAIRS, PER MODEL — HOW MUCH OVERLAP?")
    tops = {}
    for nm in names:
        S = models[nm]
        _, R = hub(S)
        cand = [(float(R[a, b]), a, b) for a in range(n) for b in range(a + 3, n)]
        cand.sort(key=lambda t: -t[0])
        tops[nm] = [(a, b) for _, a, b in cand[:20]]
        w(f"   {nm}: {tops[nm]}")
    a, b = set(tops[names[0]]), set(tops[names[1]])
    w(f"   overlap = {len(a & b)}/20  {sorted(a & b)}")
    w()

    w("D. MONASTERY AND FINAL-SNOW CENTRALITY UNDER BOTH MODELS")
    for nm in names:
        S = models[nm]
        rm, _ = hub(S)
        rk = {k: int((rm < rm[k]).sum()) + 1 for k in range(n)}
        monks = [u.index for u in units
                 if any(x in u.text.lower() for x in ("monk", "monastery"))]
        w(f"   {nm}")
        w(f"      monastery units {monks} -> ranks "
          f"{[rk[k] for k in monks]} of {n} (1 = most peripheral)")
        w(f"      last unit {n-1} (final snow) -> rank {rk[n-1]} of {n}, "
          f"centrality {rm[n-1]:.3f}")
    w()

    # ------------------------------------------------------------------
    w("=" * 78)
    w("E. WAS THE LEXICAL ABLATION UNDERPOWERED?")
    w("=" * 78)
    w()
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        m1 = SentenceTransformer("all-MiniLM-L6-v2")
        m1.eval()
        WEATHER = {"snow", "snowing", "snow-stiffened", "snowy", "cold", "wet",
                   "rain", "window", "garden", "shivering", "shiver",
                   "outside", "out-of-doors", "air", "dew", "damp",
                   "fragrant", "chilled", "cool"}
        STOP = set("""a an the and or but if of to in on at for with by from as is
        was were be been being he she it they them his her their its i you we me my
        your that this these those not no so do did does had has have will would
        could should there here then than when where which who whom what how all any
        both each few more most other some such only own same too very can just now
        up down out off over under again further once about into through during
        before after above below between said asked cried""".split())

        def enc(pairs):
            with torch.no_grad():
                e = m1.encode(list(pairs), convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
            return e

        def cos_of(ti, tj):
            e = enc([ti, tj])
            return float(e[0] @ e[1])

        def ablate_words(text, words):
            ws = set(x.lower() for x in words)
            return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                          lambda m: "thing"
                          if m.group(0).lower().strip("'") in ws else m.group(0),
                          text)

        t2, t73 = units[2].text, units[73].text
        base = cos_of(t2, t73)
        w(f"   baseline cos(2,73) = {base:.4f}")

        wl2 = sorted({x for x in re.findall(r"[A-Za-z]+", t2)
                      if x.lower() in WEATHER})
        wl73 = sorted({x for x in re.findall(r"[A-Za-z]+", t73)
                       if x.lower() in WEATHER})
        a_weather = cos_of(ablate_words(t2, wl2), ablate_words(t73, wl73))
        w(f"   weather ablation ({len(wl2)}+{len(wl73)} word types) "
          f"-> {a_weather:.4f}  (drop {base - a_weather:+.4f})")

        # random-content-word ablation of the SAME SIZE, 5 repetitions
        rng = random.Random(42)
        drops = []
        for rep in range(5):
            c2 = [x for x in set(re.findall(r"[A-Za-z]+", t2))
                  if x.lower() not in STOP and len(x) > 3]
            c73 = [x for x in set(re.findall(r"[A-Za-z]+", t73))
                   if x.lower() not in STOP and len(x) > 3]
            rng.shuffle(c2)
            rng.shuffle(c73)
            pick2 = c2[:len(wl2)]
            pick73 = c73[:len(wl73)]
            v = cos_of(ablate_words(t2, pick2), ablate_words(t73, pick73))
            drops.append(base - v)
        w(f"   random-content ablation, same size, 5 reps: drops "
          f"{[round(d, 4) for d in drops]}")
        w(f"      mean drop {np.mean(drops):+.4f}, sd {np.std(drops):.4f}")
        w()
        # larger ablation: remove ALL content words except punctuation-carrying
        big2 = [x for x in set(re.findall(r"[A-Za-z]+", t2))
                if x.lower() not in STOP]
        big73 = [x for x in set(re.findall(r"[A-Za-z]+", t73))
                 if x.lower() not in STOP]
        v = cos_of(ablate_words(t2, big2), ablate_words(t73, big73))
        w(f"   extreme ablation (ALL content word types: {len(big2)}+{len(big73)})"
          f" -> {v:.4f}  (drop {base - v:+.4f})")
        w("   (A large drop here means the link lives in the content words; a")
        w("    small drop means it lives in function words / syntax / length.)")
    except Exception as e:  # noqa: BLE001
        w(f"   FAILED: {e!r}")

    (DERIVED / "model_agreement_check.txt").write_text("\n".join(OUT),
                                                       encoding="utf-8")
    import csv
    with (C.TABLE_DIR / "snow_adjudication_pairs.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", DERIVED / "model_agreement_check.txt")


if __name__ == "__main__":
    main()
