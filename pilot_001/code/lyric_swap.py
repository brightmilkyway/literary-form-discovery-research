"""Swap the lyric in unit 55 for other songs, change nothing else.

Motivation.  Removing the lyric from unit 55 changed cos(55,58) by 0.000, which
rejected "literal repetition of the song text" as the explanation.  Removal is
a blunt control though: it also removes 18 words and changes the unit's length.
A sharper control is a **content swap of matched length**: keep the unit
otherwise byte-identical and replace only the lyric paragraph (para 320, 18
words) with a different song's lyric, a genre-matched Irish ballad, a neutral
non-lyric filler, the same words shuffled, and finally nothing.

If the metrics do not move under these swaps, the pair's similarity does not
depend on the song's content at all.

Writes ``data/derived/lyric_swap.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C
import ncd as NCD

DERIVED = C.DATA_DIR / "derived"
OUT = []
TARGET = 55
PARTNER = 58
SUPPORT = [54, 56, 57, 37, 69, 72, 73]

ORIGINAL = ("O, the rain falls on my heavy locks\n"
            "And the dew wets my skin,\n"
            "My babe lies cold....")

VARIANTS = [
    ("original", ORIGINAL),
    ("Bieber 'Baby'",
     "And I was like baby baby baby oh\n"
     "Like baby baby baby no\n"
     "Like baby baby baby oh"),
    ("Green Day 'Wake Me Up When September Ends'",
     "Like my father's come to pass\n"
     "Seven years has gone so fast\n"
     "Wake me up when September ends"),
    ("Irish ballad 'The Minstrel Boy'",
     "The minstrel boy to the war is gone\n"
     "In the ranks of death you will find him there"),
    ("neutral filler (non-lyric)",
     "The wind moved through the empty street\n"
     "and the grey door stood open\n"
     "in the cold morning air"),
    ("original words, shuffled", None),      # filled in below
    ("removed", ""),
]

# Calibration variants, added after the first run: they answer "how much does
# ANY 18-word change to this unit move the metric?" so that the lyric's effect
# can be judged against a baseline of unrelated perturbations.
CALIBRATION = [
    ("calibration: 18 words of prose from elsewhere",
     "PARA_FROM_ELSEWHERE"),
    ("calibration: same swap in a different paragraph",
     "FILLER_IN_PARA_323"),
]


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def prev(t, n=90):
    return re.sub(r"\s+", " ", t).strip()[:n]


def build_pooler(model):
    """Paragraph-aware pooling, identical to fix_representations.py."""
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
    n = len(units)
    u = units[TARGET]
    paras = C.paragraphs(body)
    head = paras[u.first_para].text          # para 320 == the lyric
    tail = [p.text for p in paras[u.first_para + 1:u.last_para + 1]]
    assert head.strip() == ORIGINAL.strip(), "lyric paragraph changed"

    # the shuffled variant: same 18 words, different order
    toks = ORIGINAL.split()
    rng = random.Random(42)
    rng.shuffle(toks)
    variants = []
    for label, lyric in VARIANTS:
        if lyric is None:
            lyric = " ".join(toks)
        variants.append((label, lyric))

    # ---- calibration: how much does ANY 18-word change move the metric? ---
    # (a) 18 words of ordinary prose taken from a distant paragraph
    far = next(p for p in paras if p.index > 120 and p.n_words >= 18)
    far_words = " ".join(far.text.split()[:18])
    variants.append((CALIBRATION[0][0], far_words))
    # (b) the same kind of swap applied to a DIFFERENT paragraph of unit 55
    other_para = paras[u.first_para + 3]          # para 323, 25 words
    filler = ("The room was quiet and nobody spoke for a while and then the "
              "door opened slowly again in the dark night")
    fw = filler.split()
    if len(fw) > 18:
        fw = fw[:18]
    while len(fw) < 18:
        fw.append("again")
    filler18 = " ".join(fw)
    variants.append((CALIBRATION[1][0], filler18))
    # pad to the target paragraph's own length so only the content changes
    other_repl = " ".join(
        (filler18.split() + other_para.text.split())[:other_para.n_words])

    w("=" * 78)
    w("LYRIC SWAP — replace only the song text in unit 55, nothing else")
    w("=" * 78)
    w()
    w(f"unit {TARGET}: paras {u.first_para}-{u.last_para}, {u.n_words} words")
    w(f"the lyric is exactly paragraph {paras[u.first_para].index} "
      f"({paras[u.first_para].n_words} words)")
    w()
    w(f"   {'variant':<44}{'words':>6}{'chars':>7}")
    for label, lyric in variants:
        w(f"   {label:<44}{len(lyric.split()):>6}{len(lyric):>7}")
    w("   (all variants are held at the original's word count where possible,")
    w("    so that length cannot explain a difference)")
    w()

    # ---- build the modified unit texts --------------------------------
    texts = []
    for label, lyric in variants:
        if label == CALIBRATION[0][0]:
            t = "\n\n".join([lyric] + tail)      # prose from elsewhere
        elif label == CALIBRATION[1][0]:
            # rebuild the unit with para 323's text replaced instead
            body_paras = [paras[u.first_para].text] + tail
            idx = 3                              # para 323 within the unit
            body_paras[idx] = other_repl
            t = "\n\n".join(body_paras)
        elif lyric.strip() == "":
            t = "\n\n".join(tail)                # lyric removed entirely
        else:
            t = "\n\n".join([lyric] + tail)
        texts.append(t)

    import torch
    from sentence_transformers import SentenceTransformer

    base_texts = [x.text for x in units]

    results = {}
    for mname, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                       ("bge", "BAAI/bge-small-en-v1.5")):
        m = SentenceTransformer(mid)
        m.eval()
        enc = build_pooler(m)
        base = enc(base_texts)                   # all units, unmodified
        w("-" * 78)
        w(f"MODEL: {mname}")
        w("-" * 78)
        w(f"   {'variant':<44}{'cos(55,58)':>11}{'mutual':>8}"
          f"{'centr.':>8}{'rank':>6}")
        for (label, _), t in zip(variants, texts):
            v = enc([t])[0]
            c = float(v @ base[PARTNER])
            row = base @ v
            row[TARGET] = -np.inf
            rank_of_partner = int((row > c).sum()) + 1
            cen = float(row[np.isfinite(row)].mean())
            order = np.argsort(-(base @ base.T).diagonal() * 0)  # placeholder
            w(f"   {label:<44}{c:>11.3f}{rank_of_partner:>8}{cen:>8.3f}"
              f"{'':>6}")
            results.setdefault(label, {})[mname] = (c, rank_of_partner, cen)
        w()
        w("   top-5 partners of unit 55 under each variant:")
        for (label, _), t in zip(variants, texts):
            v = enc([t])[0]
            row = base @ v
            row[TARGET] = -np.inf
            top = np.argsort(-row)[:5]
            w(f"      {label:<44} "
              f"{[(int(i), round(float(row[i]), 3)) for i in top]}")
        w()
        w("   similarity to the other cluster members:")
        for (label, _), t in zip(variants, texts):
            v = enc([t])[0]
            vals = " ".join(f"{s}:{float(v @ base[s]):.3f}" for s in SUPPORT)
            w(f"      {label:<44} {vals}")
        w()

    # ---- NCD ------------------------------------------------------------
    w("-" * 78)
    w("NCD (zlib) — distance of (55,58) and its rank among all pairs")
    w("-" * 78)
    w(f"   {'variant':<44}{'d(55,58)':>10}{'rank':>12}{'words':>7}")
    for (label, _), t in zip(variants, texts):
        mod = list(base_texts)
        mod[TARGET] = t
        sub = [C.Unit(i, 0, 0, x, len(C.word_span(x)), 0, 0, "g")
               for i, x in enumerate(mod)]
        res = NCD.ncd_matrix(sub, "zlib")
        M = res.matrix
        iu, ju = np.triu_indices(n, k=1)
        d = M[TARGET, PARTNER]
        rank = int((M[iu, ju] < d).sum()) + 1
        w(f"   {label:<44}{d:>10.3f}{rank:>12}{len(t.split()):>7}")
    w()
    w("   (the NCD column is length-dominated by construction, so a")
    w("    length-matched swap should barely move it — that is the point)")

    (DERIVED / "lyric_swap.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "lyric_swap.txt")


if __name__ == "__main__":
    main()
