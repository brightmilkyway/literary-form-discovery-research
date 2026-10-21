"""SBERT sensitivity to unit size (the truncation problem).

At 150-word groups, 25 of 78 units (32%) exceed the encoder's 256-token limit
and are silently truncated, so a large part of each long unit is never seen.
This script re-runs the semantic analysis at 100-word groups, where truncation
should largely disappear, and asks whether the qualitative findings survive:

* is the monastery / monks unit still the most semantically peripheral?
* does the "religion" link (Aunt Kate on the pope ↔ Mr Browne and the monks)
  still appear among the hub-corrected distant pairs?

Writes ``data/derived/sbert_sensitivity.txt`` and
``data/derived/sbert_group100.npy``.
"""

from __future__ import annotations

import json
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def prev(t, n=130):
    s = re.sub(r"\s+", " ", t).strip()
    return s[:n] + ("…" if len(s) > n else "")


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=100)
    w(f"units at 100-word groups: {len(units)}")

    import torch
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    name = "all-MiniLM-L6-v2"
    tk = AutoTokenizer.from_pretrained(f"sentence-transformers/{name}")
    lens = np.array([len(tk(u.text, add_special_tokens=True)["input_ids"])
                     for u in units])
    model = SentenceTransformer(name)
    model.eval()
    with torch.no_grad():
        emb = model.encode([u.text for u in units], batch_size=32,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
    S = emb @ emb.T
    np.fill_diagonal(S, 1.0)
    np.save(DERIVED / "sbert_group100.npy", S)

    w(f"token lengths: min {lens.min()} median {int(np.median(lens))} "
      f"max {lens.max()};  units over 256 tokens: "
      f"{int((lens > 256).sum())} ({(lens > 256).mean() * 100:.0f}%)")
    w()

    n = len(units)
    off = ~np.eye(n, dtype=bool)
    rm = np.array([S[i, off[i]].mean() for i in range(n)])
    gm = S[off].mean()
    RES = S - rm[:, None] - rm[None, :] + gm
    np.fill_diagonal(RES, -np.inf)

    w("A. MOST PERIPHERAL UNITS (lowest mean similarity to the rest)")
    for i in np.argsort(rm)[:6]:
        w(f"   unit {i:>3}  central={rm[i]:.3f}  paras "
          f"{units[i].first_para}-{units[i].last_para}  |  {prev(units[i].text, 130)}")
    w()
    w("B. MOST CENTRAL UNITS")
    for i in np.argsort(-rm)[:6]:
        w(f"   unit {i:>3}  central={rm[i]:.3f}  paras "
          f"{units[i].first_para}-{units[i].last_para}  |  {prev(units[i].text, 130)}")
    w()
    w("C. TOP HUB-CORRECTED DISTANT PAIRS (gap >= 3)")
    cand = [(float(RES[i, j]), i, j) for i in range(n) for j in range(i + 3, n)]
    cand.sort(key=lambda t: -t[0])
    for v, i, j in cand[:12]:
        w(f"   resid={v:+.3f}  cos={S[i, j]:.3f}  units {i}&{j} (gap {j-i})  "
          f"paras {units[i].first_para}-{units[i].last_para} / "
          f"{units[j].first_para}-{units[j].last_para}")
        w(f"        {i}: {prev(units[i].text, 120)}")
        w(f"        {j}: {prev(units[j].text, 120)}")
    w()
    w("D. DOES THE 'RELIGION' THREAD SURVIVE? (units mentioning pope/monks/Church)")
    for i, u in enumerate(units):
        t = u.text.lower()
        if any(k in t for k in ("monk", "monastery", "pope", "church")):
            w(f"   unit {i:>3}  paras {u.first_para}-{u.last_para}  "
              f"central={rm[i]:.3f}  |  {prev(u.text, 110)}")

    (DERIVED / "sbert_sensitivity.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "sbert_sensitivity.txt")


if __name__ == "__main__":
    main()
