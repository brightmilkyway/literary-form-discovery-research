"""Granularity ladder: how much of the passage has to be replaced before the
link moves?

Motivation
----------
"Replace the whole paragraph" is a blunt intervention: it removes the shared
material, but it also removes a lot of unrelated text, so the measured damage
mixes "the shared content is gone" with "a large chunk of text is gone".
The right intervention is the one that removes *exactly* the shared material.
Whether that is a word, a sentence or a paragraph is an empirical question, and
it is also a researcher degree of freedom unless it is fixed by a rule.

This script runs the same case at five granularities:

  G1  minimal-pair substitution: every content word of the target paragraph
      that also occurs in the partner is replaced by a different content word
      of the same token length, drawn from a donor vocabulary that excludes the
      partner's own content words.  Syntax and length are preserved exactly;
      only the shared lexical items change.
  G2  the same shared words deleted, with the same number of neutral filler
      words appended so the length is preserved (isolates "the lexical items"
      from "the sentence frame").
  G3  the sentence carrying the most shared content, replaced by a
      length-matched unrelated sentence.
  G4  the 18-word window with the most shared content (the rule-based span
      used elsewhere in the pilot).
  G5  the whole paragraph.

If the effect already appears at G1, the shared words carry the link and the
coarser levels were never needed.  If it only appears at G4/G5, then the
conclusion "the content does not participate" was an artefact of the blunt
instrument.

Cases whose target paragraph shares NO content word with the partner have no
G1/G2 (there is nothing to substitute or delete); that is reported as such and
is itself the classification "no lexical bridge".

Writes ``data/derived/swap_granularity.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []
K = 10

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


def build_proper_noun_set(corpus_texts):
    """Corpus-level proper nouns: a token capitalised in >= 80% of its
    occurrences and seen at least twice.

    The per-paragraph heuristic above misses a name whenever it happens to
    start a sentence (it did exactly that for 'Gabriel' in the sobbing case),
    so the classification is made corpus-wide instead.
    """
    cap: dict[str, int] = {}
    tot: dict[str, int] = {}
    for text in corpus_texts:
        for tok in re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", text):
            low = tok.lower()
            if low in STOP or len(low) < 2:
                continue
            tot[low] = tot.get(low, 0) + 1
            if tok[:1].isupper():
                cap[low] = cap.get(low, 0) + 1
    return {t for t in tot if tot[t] >= 2 and cap.get(t, 0) / tot[t] >= 0.8}


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

    # vocabulary for minimal-pair substitution: The Dead's content words
    vocab_all = sorted({t for p in paras for t in ctypes(p.text)})
    by_len: dict[int, list[str]] = {}
    for t in vocab_all:
        by_len.setdefault(len(t), []).append(t)

    import torch
    from sentence_transformers import SentenceTransformer

    base_texts = [u.text for u in units]
    PROPER = build_proper_noun_set(base_texts + [p.text for p in paras])
    print(f"corpus proper nouns detected: {len(PROPER)} e.g. "
          f"{sorted(PROPER)[:12]}")

    for mname, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                       ("bge", "BAAI/bge-small-en-v1.5")):
        m = SentenceTransformer(mid)
        m.eval()
        enc = build_pooler(m)
        base = enc(base_texts)
        w("=" * 78)
        w(f"MODEL: {mname}   (K = {K} draws per granularity where applicable)")
        w("=" * 78)
        w()
        w(f"   {'case':<18}{'sh':>4}{'n_sw':>6}" +
          "".join(f"{g:>10}" for g in ("G1", "G1b", "G2", "G3", "G4", "G5")))
        w(f"   {'':<18}{'':>4}{'':>6}" +
          "".join(f"{'mean z':>10}" for _ in range(6)))
        for (ua, kw, partner, label) in CASES:
            u = units[ua]
            p = next(q for q in paras[u.first_para:u.last_para + 1]
                     if kw.lower() in q.text.lower())
            btypes = ctypes(units[partner].text)
            pw = p.text.split()

            # words of the paragraph that are shared with the partner
            shared_idx = [i for i, t in enumerate(pw)
                          if t.lower().strip("'\u2019") in btypes]
            sh = len({pw[i].lower().strip("'\u2019") for i in shared_idx})

            def rebuild(new_para):
                parts = [paras[i].text
                         for i in range(u.first_para, u.last_para + 1)]
                parts[p.index - u.first_para] = new_para
                return "\n\n".join(parts)

            variants: dict[str, list[str]] = {g: [] for g in
                                              ("G1", "G1b", "G2", "G3",
                                               "G4", "G5")}

            # ---- G1: split into thematic words vs names ------------------
            # Replacing a character's NAME damages any pair that mentions that
            # character, which is a name effect, not a content effect.  So the
            # word-level rung is split: G1a acts on the shared non-name content
            # words, G1b on the shared proper nouns.
            NAMEISH = {t for t in ctypes(p.text) & btypes
                       if t in PROPER}
            THEME = [i for i in shared_idx
                     if pw[i].lower().strip("'\u2019") not in NAMEISH]
            NAMES = [i for i in shared_idx
                     if pw[i].lower().strip("'\u2019") in NAMEISH]

            def substitute(idx_list):
                out = []
                for _ in range(K):
                    new = list(pw)
                    for i in idx_list:
                        tgt = new[i]
                        cand = by_len.get(len(tgt), [])
                        pick = None
                        for _try in range(60):
                            c = rng.choice(cand) if cand else "something"
                            if c not in btypes:
                                pick = c
                                break
                        new[i] = pick or tgt
                    out.append(" ".join(new))
                return out

            variants["G1"] = substitute(THEME) if THEME else []
            variants["G1b"] = substitute(NAMES) if NAMES else []

            # ---- G2: the same shared words deleted -----------------------
            if shared_idx:
                for _ in range(K):
                    new = [t for i, t in enumerate(pw) if i not in shared_idx]
                    pad = [rng.choice(["then", "again", "there", "once",
                                       "still", "now"]) for _ in shared_idx]
                    variants["G2"].append(" ".join(new + pad))

            # ---- G3: the sentence with the most shared content -----------
            sents = [s for s in re.split(r"(?<=[.!?\u2026])\s+", p.text)
                     if s.strip()]
            if sents:
                def sc(s):
                    return len(ctypes(s) & btypes)
                best_s = max(sents, key=sc)
                bwords = len(best_s.split())
                pool = [q for q in paras
                        if q.index != p.index and q.n_words >= bwords]
                for _ in range(K):
                    d = rng.choice(pool).text.split()
                    rep = " ".join((d * (bwords // len(d) + 2))[:bwords])
                    variants["G3"].append(
                        " ".join(x for x in
                                 [p.text.replace(best_s, rep, 1)] if x))

            # ---- G4: the maximally-overlapping 18-word window ------------
            width = min(18, len(pw))
            best, head = -1, 0
            for i in range(0, max(1, len(pw) - width + 1)):
                s = len(ctypes(" ".join(pw[i:i + width])) & btypes)
                if s > best:
                    best, head = s, i
            max_share = best
            tail = pw[head + width:]
            pool = [q for q in paras
                    if q.n_words >= width
                    and len(ctypes(q.text) & btypes) <= max_share]
            for _ in range(K):
                d = rng.choice(pool).text.split()
                rep = " ".join((d * (width // len(d) + 2))[:width])
                variants["G4"].append(
                    " ".join(pw[:head] + rep.split() + tail))

            # ---- G5: the whole paragraph ---------------------------------
            pool = [q for q in paras
                    if q.n_words >= p.n_words
                    and len(ctypes(q.text) & btypes) <= max_share]
            for _ in range(K):
                d = rng.choice(pool).text.split()
                rep = " ".join((d * (p.n_words // len(d) + 2))[:p.n_words])
                variants["G5"].append(rep)

            cos0 = float(base[ua] @ base[partner])
            means = []
            for g in ("G1", "G1b", "G2", "G3", "G4", "G5"):
                if not variants[g]:
                    means.append(float("nan"))
                    continue
                mods = [rebuild(v) for v in variants[g]]
                emb = enc(mods)
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
                zz = zz[~np.isnan(zz)]
                means.append(float(zz.mean()) if len(zz) else float("nan"))
            w(f"   {label:<18}{sh:>4}{len(shared_idx):>6}" +
              "".join(("      n/a" if np.isnan(x) else f"{x:>10.2f}")
                      for x in means))
        w()
        w("   sh   = content word TYPES the paragraph shares with the partner")
        w("   n_sw = number of word OCCURRENCES acted on by G1/G2")
        w("   G1 minimal-pair substitution of the shared words (syntax and")
        w("      length preserved)")
        w("   G2 the same words deleted, length preserved with filler")
        w("   G3 the most shared sentence replaced")
        w("   G4 the maximally-overlapping 18-word window replaced")
        w("   G5 the whole paragraph replaced")
        w()
        w("   n/a means the paragraph shares no content word with the partner,")
        w("   so there is nothing to substitute or delete at word level.")
        w()

    (DERIVED / "swap_granularity.txt").write_text("\n".join(OUT),
                                                  encoding="utf-8")
    print("wrote", DERIVED / "swap_granularity.txt")


if __name__ == "__main__":
    main()
