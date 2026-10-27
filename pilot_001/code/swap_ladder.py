"""Donor-source ladder: is the content-carriedness verdict invariant to the
material used for the replacement?

Motivation
----------
The earlier runs drew replacements from *The Dead* itself.  That keeps author,
register, period and length constant, but it is not "unrelated": same-story
donors carried the story's characters, places and themes into the replacement.
Measured on the six cases, same-story donors shared 3.6 content word types with
the partner on average, while the lyric paragraph being removed shared 0 - so
for that case the swap INJECTED related material instead of removing it.

An examiner would ask: is the classification an artefact of the donor source?
Answer it by running the same operation on a LADDER of sources, from "same
story" to "definitely unrelated", and reporting whether the verdict moves.

Rungs (each replacement is length-matched and constrained to share no more
content word types with the partner than the removed span does, so that every
rung removes rather than injects):

  L1  another paragraph of The Dead                     (style, world, cast)
  L2  as L1, and sharing no proper noun with the partner (style, world)
  L3  a paragraph of another Dubliners story            (author, book, period)
  L4  The Dead's own content words, shuffled            (vocabulary, no syntax)
  L5  the corpus's content words, shuffled              (broad vocabulary only)

L3 is the informative rung: same author, same book, same period, different
content.  It is fetched by ``fetch_external_donors.py``.

Writes ``data/derived/swap_ladder.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
EXTERNAL = C.DATA_DIR / "source" / "dubliners_other_stories.txt"
OUT = []
K = 10
SPAN = 18
MIN_DONOR_GAP = 100

CASES = [
    (55, "rain falls", 58, "the lyric"),
    (72, "Michael Furey", 69, "Michael Furey"),
    (39, "monks", 30, "the monks"),
    (16, "lemonade", 12, "the lemonade"),
    (74, "sobbing", 71, "sobbing"),
    (58, "Lass of Aughrim", 55, "the song name"),
]

STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while yes o oh well""".split())


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def ctypes(text):
    return {t for t in C.tokens(text) if t not in STOP and len(t) > 1}


def proper_nouns(text):
    """Capitalised tokens that are not sentence-initial (crude but explicit)."""
    out = set()
    for sent in re.split(r"(?<=[.!?\u2026])\s+", text):
        toks = re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", sent)
        for t in toks[1:]:
            if t[:1].isupper() and t.lower() not in STOP:
                out.add(t.lower())
    return out


def build_pooler(model):
    import torch

    def enc(texts):
        subs, owner = [], []
        for k, t in enumerate(texts):
            ps = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
            buf, cnt = [], 0
            for p in ps:
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
            e = model.encode(subs, batch_size=64, convert_to_numpy=True,
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
    rng = random.Random(42)

    if not EXTERNAL.exists():
        raise SystemExit("run fetch_external_donors.py first")
    ext_text = EXTERNAL.read_text(encoding="utf-8")
    ext_paras = [p.strip() for p in re.split(r"\n\s*\n", ext_text) if p.strip()]
    ext_pool = [p for p in ext_paras if len(C.word_span(p)) >= SPAN]
    w(f"external donor source: {len(ext_pool)} paragraphs of other Dubliners "
      f"stories")

    dead_vocab = sorted({t for p in paras for t in ctypes(p.text)})
    corpus_vocab = sorted(set(dead_vocab) | {t for p in ext_paras
                                             for t in ctypes(p)})

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
        w(f"MODEL: {mname}   (K = {K} donors per rung)")
        w("=" * 78)
        w()
        w(f"   {'case':<18}{'sh':>4}" +
          "".join(f"{lab.split()[0]:>9}" for lab in
                  ["L1", "L2", "L3", "L4", "L5"]))
        w(f"   {'':<18}{'':>4}" +
          "".join(f"{'mean z':>9}" for _ in range(5)))
        for (ua, kw, partner, label) in CASES:
            u = units[ua]
            p = next(q for q in paras[u.first_para:u.last_para + 1]
                     if kw.lower() in q.text.lower())
            btypes = ctypes(units[partner].text)
            bprops = proper_nouns(units[partner].text)

            pw = p.text.split()
            width = min(SPAN, len(pw))
            # rule-based span: the window sharing most content types with B
            best, head = -1, 0
            for i in range(0, max(1, len(pw) - width + 1)):
                sc = len(ctypes(" ".join(pw[i:i + width])) & btypes)
                if sc > best:
                    best, head = sc, i
            sh = best
            tail = pw[head + width:]

            def make(new_words):
                return "\n\n".join(
                    [" ".join(pw[:head] + new_words + tail)]
                    + [paras[i].text
                       for i in range(u.first_para + 1, u.last_para + 1)])

            rung_stats = []
            for rung in ("L1", "L2", "L3", "L4", "L5"):
                made = []
                if rung in ("L1", "L2"):
                    pool = [q for q in paras
                            if q.n_words >= width
                            and abs(q.index - p.index) > MIN_DONOR_GAP
                            and len(ctypes(q.text) & btypes) <= sh
                            and (rung == "L1"
                                 or not (proper_nouns(q.text) & bprops))]
                    rng.shuffle(pool)
                    for q in pool[:K]:
                        dw = q.text.split()
                        made.append(" ".join(
                            (dw * (width // len(dw) + 2))[:width]))
                elif rung == "L3":
                    pool = [q for q in ext_pool
                            if len(ctypes(q) & btypes) <= sh
                            and not (proper_nouns(q) & bprops)]
                    rng.shuffle(pool)
                    for q in pool[:K]:
                        dw = q.split()
                        made.append(" ".join(
                            (dw * (width // len(dw) + 2))[:width]))
                else:
                    vocab = dead_vocab if rung == "L4" else corpus_vocab
                    tries = 0
                    while len(made) < K and tries < 400:
                        tries += 1
                        pick = [rng.choice(vocab) for _ in range(width)]
                        if len(set(pick) & btypes) <= sh:
                            made.append(" ".join(pick))
                rung_stats.append((rung, made))
            zs_by_rung = {}
            for rung, made in rung_stats:
                if not made:
                    zs_by_rung[rung] = np.array([np.nan])
                    continue
                mods = [make(m.split()) for m in made]
                emb = enc(mods)
                cos0 = float(base[ua] @ base[partner])
                zz = []
                for k in range(len(mods)):
                    v = emb[k]
                    deltas = np.array([float(v @ base[c])
                                       - float(base[ua] @ base[c])
                                       for c in range(n) if c != ua])
                    dp = float(v @ base[partner]) - cos0
                    mu, sd = float(deltas.mean()), float(deltas.std())
                    zz.append((dp - mu) / sd if sd > 0 else np.nan)
                zz = np.array(zz)
                zs_by_rung[rung] = zz[~np.isnan(zz)]
            w(f"   {label:<18}{sh:>4}" +
              "".join(f"{zs_by_rung[r].mean():>9.2f}"
                      for r in ("L1", "L2", "L3", "L4", "L5")))
        w()
        w("   sh = content word types the REMOVED span shares with the partner")
        w("   L1 another paragraph of The Dead")
        w("   L2 as L1, no shared proper noun with the partner")
        w("   L3 a paragraph of another Dubliners story (same author/book)")
        w("   L4 The Dead's content words, shuffled")
        w("   L5 the corpus's content words, shuffled")
        w()
        w("   All rungs are length-matched and share <= sh content types with")
        w("   the partner, so each removes rather than injects related")
        w("   material.  The verdict is invariant if the column pattern holds")
        w("   across the five rungs.")
        w()

    (DERIVED / "swap_ladder.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "swap_ladder.txt")


if __name__ == "__main__":
    main()
