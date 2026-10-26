"""Is the 150-word unit sensitive to content AT ALL?

The lyric swap showed that replacing an 18-word span in unit 55 moves
cos(55,58) by at most 0.02.  Two readings:

  (a) the pair (55,58) does not depend on that span -> a finding about the pair
  (b) the 150-word unit is simply insensitive to an ~11% content substitution
      -> a finding about the method, and (a) is uninformative

To tell them apart, run the SAME swap on pairs whose similarity we have reason
to think IS content-driven, and compare.

Design
------
For a unit A and a paragraph inside it that carries a motif, replace that
paragraph with unrelated prose of the SAME word count.  Then measure, for each
other unit C:

    delta(C) = cos(A', C) - cos(A, C)

Any swap moves A's embedding, so every delta is non-zero.  The question is
whether the delta for the PARTNER that shares the motif is larger than the
deltas for unrelated units.  Report:

    delta_partner           the motif-sharing partner
    mean |delta| over 20 unrelated units   (the global shift from swapping)
    excess = |delta_partner| - mean|delta_unrelated|

If excess is near zero for a pair, the pair's similarity is not carried by the
swapped content; if it is large, it is.

Writes ``data/derived/swap_sensitivity.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

# (unit, keyword identifying the motif-bearing paragraph, partner, label)
CASES = [
    (55, "rain falls", 58, "the lyric  <->  the song name"),
    (72, "Michael Furey", 73, "Michael Furey  <->  the night before"),
    (72, "Michael Furey", 69, "Michael Furey  <->  Gabriel asks"),
    (39, "monks", 30, "the monks  <->  the pope"),
    (16, "lemonade", 12, "the lemonade  <->  the punch"),
    (74, "sobbing", 71, "sobbing  <->  cold interrogation"),
    (58, "Lass of Aughrim", 55, "the song name  <->  the lyric"),
]


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def build_pooler(model):
    import torch

    def enc(texts):
        subs, owner = [], []
        for k, t in enumerate(texts):
            paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
            buf, cnt = [], 0
            for p in paras:
                nw = len(C.word_span(p))
                if nw > 130:
                    toks = p.split()
                    for i in range(0, len(toks), 130):
                        subs.append(" ".join(toks[i:i + 130]))
                        owner.append(k)
                    continue
                if cnt + nw > 130 and buf:
                    subs.append("\n\n".join(buf))
                    owner.append(k)
                    buf, cnt = [], 0
                buf.append(p)
                cnt += nw
            if buf:
                subs.append("\n\n".join(buf))
                owner.append(k)
        with torch.no_grad():
            e = model.encode(subs, batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        out = np.zeros((len(texts), e.shape[1]))
        for k, o in enumerate(owner):
            out[o] += e[k]
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out
    return enc


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    n = len(units)

    def para_of(unit_idx, keyword):
        u = units[unit_idx]
        for p in paras[u.first_para:u.last_para + 1]:
            if keyword.lower() in p.text.lower():
                return p
        return None

    rng = random.Random(42)
    # a pool of "unrelated prose" to draw filler from: distant paragraphs
    donor_pool = [p for p in paras if p.n_words >= 12]

    import torch
    from sentence_transformers import SentenceTransformer

    base_texts = [u.text for u in units]

    for mname, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                       ("bge", "BAAI/bge-small-en-v1.5")):
        m = SentenceTransformer(mid)
        m.eval()
        enc = build_pooler(m)
        base = enc(base_texts)
        w("=" * 78)
        w(f"MODEL: {mname}")
        w("=" * 78)
        w()
        w(f"   {'swap':<44}{'partner':>8}{'delta_p':>10}"
          f"{'|d|_unrel':>11}{'excess':>9}")
        for (ua, kw, partner, label) in CASES:
            p = para_of(ua, kw)
            if p is None:
                w(f"   {label:<44}  (paragraph not found: {kw!r})")
                continue
            nwords = p.n_words
            # build replacement text: unrelated words, same count
            donor = max(donor_pool,
                        key=lambda q: abs(q.start - p.start))
            words = donor.text.split()
            while len(words) < nwords:
                words += donor.text.split()
            repl = " ".join(words[:nwords])

            u = units[ua]
            parts = [paras[i].text for i in range(u.first_para, u.last_para + 1)]
            parts[p.index - u.first_para] = repl
            mod = "\n\n".join(parts)

            texts = list(base_texts)
            texts[ua] = mod
            v = enc([mod])[0]

            delta_p = float(v @ base[partner]) - float(base[ua] @ base[partner])
            others = [c for c in range(n) if c != ua and c != partner]
            rng2 = random.Random(7)
            sample = rng2.sample(others, 20)
            ds = [abs(float(v @ base[c]) - float(base[ua] @ base[c]))
                  for c in sample]
            mean_d = float(np.mean(ds))
            excess = abs(delta_p) - mean_d
            w(f"   {label:<44}{partner:>8}{delta_p:>+10.3f}"
              f"{mean_d:>11.3f}{excess:>+9.3f}")
        w()
        w("   delta_p     = change in cos(A,B) after swapping a paragraph in A")
        w("   |d|_unrel   = mean |change| over 20 unrelated partners (the")
        w("                 global shift that any swap causes)")
        w("   excess      = |delta_p| - |d|_unrel ; large positive means the")
        w("                 partner is specifically damaged by the swap")
        w()

    (DERIVED / "swap_sensitivity.txt").write_text("\n".join(OUT),
                                                  encoding="utf-8")
    print("wrote", DERIVED / "swap_sensitivity.txt")


if __name__ == "__main__":
    main()
