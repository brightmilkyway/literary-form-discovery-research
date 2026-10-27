"""Probe the "none" pairs from the full scan.

The scan localises only ~22-31% of the top 500 pairs: for the rest, replacing
any ONE component (shared thematic words, shared names, the most shared
sentence, or the single most-shared paragraph) does not collapse the link.

Two readings:
  (a) the similarity is DISTRIBUTED over several paragraphs, so no single
      intervention removes enough of it;
  (b) the similarity is not carried by shared vocabulary at all.

Test: for the "none" pairs, replace EVERY paragraph of A that shares any
content word with B (the maximal lexical removal available), keeping each
paragraph's length, and measure z again.

  * if the maximal removal collapses them -> reading (a), distributed
  * if it does not -> reading (b), not lexical

Writes ``data/derived/scan_none_probe_<tag>.txt``.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="minilm")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--K", type=int, default=2)
    args = ap.parse_args()

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    n = len(units)
    rng = random.Random(42)

    rows = list(csv.DictReader(
        (DERIVED / f"scan_full_ladder_{args.tag}.csv").open(
            encoding="utf-8-sig")))
    none_rows = [r for r in rows if r["depth"] == "none"]
    print(f"pairs total {len(rows)}, depth='none' {len(none_rows)}")

    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(args.model)
    m.eval()
    enc = build_pooler(m)
    base = enc([u.text for u in units])
    rep = Replacer(args.model)

    jobs, meta = [], []
    for r in none_rows:
        A, B = int(r["A"]), int(r["B"])
        u = units[A]
        btypes = ctypes(units[B].text)
        hits = [q for q in paras[u.first_para:u.last_para + 1]
                if ctypes(q.text) & btypes]
        if not hits:
            continue
        for _ in range(args.K):
            parts = [paras[k].text
                     for k in range(u.first_para, u.last_para + 1)]
            for q in hits:
                words = q.text.split()
                new = [rep.pick(w, btypes, rng) for w in words]
                parts[q.index - u.first_para] = " ".join(new)
            jobs.append((len(meta), "\n\n".join(parts)))
        meta.append(dict(A=A, B=B, n_paras_replaced=len(hits),
                         frac_replaced=sum(q.n_words for q in hits)
                         / max(1, u.n_words)))

    print(f"variants: {len(jobs)} over {len(meta)} pairs")
    texts = [t for _, t in jobs]
    embs = []
    with torch.no_grad():
        for a in range(0, len(texts), 256):
            embs.append(m.encode(texts[a:a + 256], batch_size=64,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False))
    V = np.vstack(embs)

    A_of = np.array([meta[j[0]]["A"] for j in jobs])
    B_of = np.array([meta[j[0]]["B"] for j in jobs])
    S = V @ base.T
    row0 = np.array([base[a] @ base.T for a in A_of])
    ok = np.ones((len(jobs), n), dtype=bool)
    ok[np.arange(len(jobs)), A_of] = False
    delta = S - row0
    dm = np.array([delta[k][ok[k]].mean() for k in range(len(jobs))])
    ds = np.array([delta[k][ok[k]].std() for k in range(len(jobs))])
    zt = np.array([delta[k, B_of[k]] for k in range(len(jobs))])
    z = (zt - dm) / np.where(ds > 0, ds, np.nan)

    per = {}
    for k, (pi, _) in enumerate(jobs):
        if not np.isnan(z[k]):
            per.setdefault(pi, []).append(float(z[k]))
    zz = np.array([np.mean(v) for v in per.values()])

    out = []
    out.append("=" * 78)
    out.append(f"PROBE OF THE 'none' PAIRS — maximal lexical removal "
               f"({args.model})")
    out.append("=" * 78)
    out.append("")
    out.append(f"  pairs probed: {len(zz)} of {len(none_rows)}")
    out.append(f"  replaced, per pair: mean {np.mean([m_['n_paras_replaced'] for m_ in meta]):.1f} "
               f"paragraphs = {100*np.mean([m_['frac_replaced'] for m_ in meta]):.0f}% "
               f"of the unit's words")
    out.append("")
    out.append(f"  mean z after maximal removal: {zz.mean():+.2f}")
    out.append(f"  median z                    : {np.median(zz):+.2f}")
    out.append(f"  z <= -1.5 in {int((zz <= -1.5).sum())} of {len(zz)} "
               f"({100*(zz <= -1.5).mean():.0f}%)")
    out.append(f"  z <= -1.0 in {int((zz <= -1.0).sum())} of {len(zz)} "
               f"({100*(zz <= -1.0).mean():.0f}%)")
    out.append("")
    out.append("  Reading: if the maximal removal still leaves z near zero, the")
    out.append("  similarity of these pairs is not carried by shared content")
    out.append("  words at all (reading b); if it collapses them, the similarity")
    out.append("  is distributed over several paragraphs (reading a).")
    out.append("")
    # how many paragraphs were actually available to remove
    counts = [m_["n_paras_replaced"] for m_ in meta]
    out.append(f"  paragraphs removed per pair: min {min(counts)} "
               f"median {int(np.median(counts))} max {max(counts)}")

    (DERIVED / f"scan_none_probe_{args.tag}.txt").write_text(
        "\n".join(out), encoding="utf-8")
    print("wrote", DERIVED / f"scan_none_probe_{args.tag}.txt")


if __name__ == "__main__":
    main()
