"""Full-text scan: run the granularity ladder over many pairs, by rule.

The v13 ladder was demonstrated on six hand-picked cases.  This script turns it
into a DISTRIBUTION over a rule-defined population, so that the conclusion no
longer rests on the author's choice of examples.

Rule-defined population
  all unit pairs (150-word groups) with index distance >= MIN_GAP, ranked by
  cosine, taking the top N.  No case is chosen by hand.

Per pair, by rule
  * target paragraph in A  = the one sharing most content word types with B
  * span                   = the 18-word window of that paragraph sharing most
                             with B
  * shared words           = content word types of the paragraph that also
                             occur in B; split into THEMATIC and NAMED using
                             spaCy's NER + PROPN tags
  * replacement words      = the encoder's own embedding matrix, furthest from
                             the original at a comparable token length

Levels
  G1  thematic shared words substituted   G1b named shared words substituted
  G2  shared words deleted, length kept   G3 the most shared sentence replaced
  G5  the whole paragraph replaced

Efficiency: all variants for all pairs are built first, encoded in one batched
pass, and the z-scores are then computed with a single matrix product.  Calling
the encoder once per pair would take hours.

Writes ``data/derived/scan_full_ladder.txt`` plus
``data/derived/scan_full_ladder.csv`` (per-pair rows).
"""

from __future__ import annotations

import argparse
import csv
import random
import re

import numpy as np

import common as C
from ladder_principled import Replacer, build_names, build_pooler, ctypes

DERIVED = C.DATA_DIR / "derived"
OUT = []
MIN_GAP = 5
SPAN = 18
Z_THRESHOLD = -1.5          # reporting device only; continuous z is also given


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="minilm")
    args = ap.parse_args()

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    n = len(units)
    rng = random.Random(42)

    named, ner_str, _ = build_names(body)

    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(args.model)
    m.eval()
    enc = build_pooler(m)
    base = enc([u.text for u in units])          # (n, d), normalised
    rep = Replacer(args.model)
    w(f"model {args.model}; units {n}; names from {ner_str}")
    w()

    # ---- rule-defined pair population ----------------------------------
    iu, ju = np.triu_indices(n, k=1)
    S0 = base @ base.T
    cand = [(float(S0[i, j]), int(i), int(j))
            for i, j in zip(iu, ju) if j - i >= MIN_GAP]
    cand.sort(key=lambda t: -t[0])
    pairs = cand[:args.pairs]
    w(f"population: all pairs with gap >= {MIN_GAP} ({len(cand)}), "
      f"taking the top {len(pairs)} by cosine")
    w(f"  cosine range of the selected pairs: {pairs[-1][0]:.3f} .. "
      f"{pairs[0][0]:.3f}")
    w()

    # ---- build every variant, by rule ----------------------------------
    jobs = []            # (pair_idx, level, text)
    meta = []            # per pair: A, B, sh, th, nm, n_sw, cos0, head, para
    for pi, (cos0, A, B) in enumerate(pairs):
        u = units[A]
        btypes = ctypes(units[B].text)
        # rule: the paragraph of A sharing most content with B
        best_p, best_s = None, -1
        for q in paras[u.first_para:u.last_para + 1]:
            s = len(ctypes(q.text) & btypes)
            if s > best_s:
                best_p, best_s = q, s
        p = best_p
        pw = p.text.split()

        shared_idx = [i for i, t in enumerate(pw)
                      if t.lower().strip("'\u2019") in btypes]
        shared_types = {pw[i].lower().strip("'\u2019") for i in shared_idx}
        name_types = shared_types & named
        theme_idx = [i for i in shared_idx
                     if pw[i].lower().strip("'\u2019") not in name_types]
        name_idx = [i for i in shared_idx
                    if pw[i].lower().strip("'\u2019") in name_types]

        def rebuild(new_para):
            parts = [paras[k].text
                     for k in range(u.first_para, u.last_para + 1)]
            parts[p.index - u.first_para] = new_para
            return "\n\n".join(parts)

        # G1 / G1b
        for lvl, idx_list in (("G1", theme_idx), ("G1b", name_idx)):
            if not idx_list:
                continue
            for _ in range(args.K):
                new = list(pw)
                for i in idx_list:
                    new[i] = rep.pick(pw[i], btypes, rng)
                jobs.append((pi, lvl, rebuild(" ".join(new))))

        # G2
        if shared_idx:
            for _ in range(args.K):
                new = [t for i, t in enumerate(pw) if i not in shared_idx]
                pad = [rep.pick(pw[i], btypes, rng) for i in shared_idx]
                jobs.append((pi, "G2", rebuild(" ".join(new + pad))))

        # G3
        sents = [s for s in re.split(r"(?<=[.!?\u2026])\s+", p.text)
                 if s.strip()]
        if sents:
            best_s_txt = max(sents, key=lambda s: len(ctypes(s) & btypes))
            bw = len(best_s_txt.split())
            pool = [q for q in paras if q.index != p.index and q.n_words >= bw]
            if pool:
                for _ in range(args.K):
                    d = rng.choice(pool).text.split()
                    rep_txt = " ".join((d * (bw // len(d) + 2))[:bw])
                    jobs.append((pi, "G3",
                                 rebuild(p.text.replace(best_s_txt, rep_txt, 1))))

        # G5
        pool = [q for q in paras if q.n_words >= p.n_words]
        for _ in range(args.K):
            d = rng.choice(pool).text.split()
            jobs.append((pi, "G5", rebuild(
                " ".join((d * (p.n_words // len(d) + 2))[:p.n_words]))))

        meta.append(dict(pair=pi, A=A, B=B, cos0=cos0, para=p.index,
                         sh=len(shared_types), th=len(theme_idx),
                         nm=len(name_idx), n_sw=len(shared_idx)))

    w(f"variants built: {len(jobs)}  (avg {len(jobs)/max(1,len(pairs)):.1f} "
      f"per pair)")
    w()

    # ---- encode everything in one batched pass -------------------------
    texts = [t for _, _, t in jobs]
    embs = []
    BS = 256
    with torch.no_grad():
        for a in range(0, len(texts), BS):
            e = m.encode(texts[a:a + BS], batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
            embs.append(e)
            if (a // BS) % 20 == 0:
                print(f"  encoded {a + len(e)}/{len(texts)}", flush=True)
    V = np.vstack(embs)
    print(f"  encoded {len(V)}/{len(texts)}", flush=True)

    # ---- z-scores in one pass -----------------------------------------
    A_of = np.array([meta[j[0]]["A"] for j in jobs])
    B_of = np.array([meta[j[0]]["B"] for j in jobs])
    S = V @ base.T                              # (nvar, n)
    row0 = np.array([base[a] @ base.T for a in A_of])   # (nvar, n)
    ok = np.ones((len(jobs), n), dtype=bool)
    ok[np.arange(len(jobs)), A_of] = False
    delta = S - row0
    dm = np.array([delta[k][ok[k]].mean() for k in range(len(jobs))])
    ds = np.array([delta[k][ok[k]].std() for k in range(len(jobs))])
    zt = np.array([delta[k, B_of[k]] for k in range(len(jobs))])
    z = (zt - dm) / np.where(ds > 0, ds, np.nan)

    # ---- aggregate per pair -------------------------------------------
    levels = ("G1", "G1b", "G2", "G3", "G5")
    per = {pi: {lv: [] for lv in levels} for pi in range(len(pairs))}
    for k, (pi, lv, _) in enumerate(jobs):
        if not np.isnan(z[k]):
            per[pi][lv].append(float(z[k]))
    for pi in per:
        for lv in levels:
            v = per[pi][lv]
            per[pi][lv] = float(np.mean(v)) if v else float("nan")

    def depth(d):
        for lv, name in (("G1", "word"), ("G1b", "name"),
                         ("G3", "sentence"), ("G5", "paragraph")):
            if not np.isnan(d[lv]) and d[lv] <= Z_THRESHOLD:
                return name
        return "none"

    counts = {}
    rows = []
    for pi, d in per.items():
        dep = depth(d)
        counts[dep] = counts.get(dep, 0) + 1
        rows.append(dict(meta[pi], **{lv: d[lv] for lv in levels},
                         depth=dep))

    # ---- report --------------------------------------------------------
    w("=" * 78)
    w(f"FULL-TEXT SCAN — top {len(pairs)} non-adjacent pairs (gap >= {MIN_GAP}), "
      f"K = {args.K} draws per level")
    w("=" * 78)
    w()
    w(f"   classification rule: finest level with mean z <= {Z_THRESHOLD}")
    w(f"   order tried: G1 word -> G1b name -> G3 sentence -> G5 paragraph")
    w()
    tot = len(pairs)
    for k in ("word", "name", "sentence", "paragraph", "none"):
        c = counts.get(k, 0)
        w(f"   {k:<10} {c:>4}  ({100*c/tot:5.1f}%)")
    w()
    sh0 = sum(1 for r in rows if r["sh"] == 0)
    w(f"   pairs whose target paragraph shares NO content word (sh = 0): "
      f"{sh0} ({100*sh0/tot:.1f}%)  -- the word-level rungs do not apply")
    w()

    for lv in levels:
        v = np.array([r[lv] for r in rows if not np.isnan(r[lv])])
        if len(v):
            w(f"   {lv:<5} n={len(v):>4}  mean z {v.mean():+.2f}  "
              f"median {np.median(v):+.2f}  z<-1.5 in {int((v < -1.5).sum())}")
    w()

    # how depth relates to observable properties
    w("   depth vs. observable pair properties (means):")
    w(f"   {'depth':<11}{'n':>5}{'cos0':>8}{'gap':>6}{'sh':>5}{'th':>5}{'nm':>5}")
    for k in ("word", "name", "sentence", "paragraph", "none"):
        sel = [r for r in rows if r["depth"] == k]
        if not sel:
            continue
        w(f"   {k:<11}{len(sel):>5}"
          f"{np.mean([r['cos0'] for r in sel]):>8.3f}"
          f"{np.mean([r['B'] - r['A'] for r in sel]):>6.0f}"
          f"{np.mean([r['sh'] for r in sel]):>5.1f}"
          f"{np.mean([r['th'] for r in sel]):>5.1f}"
          f"{np.mean([r['nm'] for r in sel]):>5.1f}")

    (DERIVED / f"scan_full_ladder_{args.tag}.txt").write_text(
        "\n".join(OUT), encoding="utf-8")
    with (DERIVED / f"scan_full_ladder_{args.tag}.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", DERIVED / f"scan_full_ladder_{args.tag}.txt")


if __name__ == "__main__":
    main()
