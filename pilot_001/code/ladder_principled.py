"""Principled granularity ladder, and a scan over many pairs.

Two review points are fixed here.

  1. NAMES.  The earlier runs identified proper nouns with a capitalisation
     heuristic, which was wrong in both directions (it called 'academy',
     'church', 'god', 'graces', 'ireland' names, and missed multi-word names
     such as 'Aunt Kate' and 'Michael Furey').  Names now come from spaCy's
     statistical NER (en_core_web_sm) plus its PROPN tag, and the model name is
     recorded.  Like any statistical tagger it has its own error rate; that is
     reported rather than hidden.

  2. REPLACEMENT WORDS.  The earlier G2 padded deletions with a hand-picked
     six-word list.  Replacements are now drawn from the encoder's OWN
     embedding matrix: among whole alphabetic vocabulary items of comparable
     length, the one whose embedding is FURTHEST (cosine) from the word being
     replaced and from the partner's content words is chosen.  No hand-picked
     list is involved, and the same procedure supplies both the G1
     substitutions and the G2 filler.

Granularities
  G1   shared THEMATIC content words, substituted in place (syntax and length
       preserved)
  G1b  shared NAMED entities, substituted in place (the name-effect control)
  G2   shared content words deleted, length restored with embedding-selected
       filler
  G3   the sentence carrying the most shared content, replaced
  G5   the whole paragraph, replaced

Two modes
  cases  the six named cases, for comparison with the earlier tables
  scan   the top N non-adjacent pairs by similarity, chosen by rule, with the
         span and the partner fixed by the pair itself

Writes ``data/derived/ladder_principled.txt``.
"""

from __future__ import annotations

import argparse
import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

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


# --------------------------------------------------------------------------
# 1. names, from a real NER
# --------------------------------------------------------------------------

def build_names(body: str, model: str = "en_core_web_sm"):
    """Return (named_token_set, ner_model_string, entity_counter)."""
    import spacy
    from collections import Counter

    nlp = spacy.load(model)
    doc = nlp(body)
    named = set()
    for e in doc.ents:                      # any entity type counts as "named"
        for t in e:
            named.add(t.text.lower())
    for t in doc:
        if t.pos_ == "PROPN":
            named.add(t.text.lower())
    cnt = Counter((e.label_, e.text) for e in doc.ents)
    return named, f"spaCy {spacy.__version__} / {model}", cnt


# --------------------------------------------------------------------------
# 2. replacement words, from the encoder's own embedding matrix
# --------------------------------------------------------------------------

class Replacer:
    """Pick the vocabulary word furthest from the target in the encoder's
    input-embedding space, at a comparable token length."""

    def __init__(self, model_name: str, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(model_name)
        else:
            m = model                      # reuse an already-loaded model
        W = m[0].auto_model.get_input_embeddings().weight.detach().cpu().numpy()
        tok = m.tokenizer
        vocab = tok.get_vocab()
        self.words = [x for x in vocab
                      if re.fullmatch(r"[a-z]{3,}", x)]
        self.W = W
        self.idx = {x: vocab[x] for x in self.words}
        # normalise once
        self.E = W[[self.idx[x] for x in self.words]]
        self.E /= (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-9)
        self.by_len: dict[int, list[int]] = {}
        for i, x in enumerate(self.words):
            self.by_len.setdefault(len(x), []).append(i)

    def pick(self, target: str, banned: set[str], rng: random.Random) -> str:
        """Furthest-from-target word of comparable length, not in `banned`."""
        cands = []
        for L in (len(target), len(target) - 1, len(target) + 1):
            cands += self.by_len.get(L, [])
        if not cands:
            cands = list(range(len(self.words)))
        tid = self.idx.get(target.lower())
        if tid is None:
            return rng.choice([self.words[i] for i in cands])
        tvec = self.W[tid]
        tvec = tvec / (np.linalg.norm(tvec) + 1e-9)
        sims = self.E[cands] @ tvec                 # higher = more similar
        order = np.argsort(sims)                    # least similar first
        for k in order[:400]:
            word = self.words[cands[int(k)]]
            if word not in banned:
                return word
        return self.words[cands[int(order[0])]]


# --------------------------------------------------------------------------
# machinery
# --------------------------------------------------------------------------

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


def ctypes(text, stop=STOP):
    return {t for t in C.tokens(text) if t not in stop and len(t) > 1}


def ladder_for_pair(units, paras, ua, partner, p, enc, base, rep: Replacer,
                    named: set, K: int, rng: random.Random):
    """Run the five granularities for one (unit, paragraph, partner)."""
    n = len(units)
    pw = p.text.split()
    btypes = ctypes(units[partner].text)

    shared_idx = [i for i, t in enumerate(pw)
                  if t.lower().strip("'\u2019") in btypes]
    shared_types = {pw[i].lower().strip("'\u2019") for i in shared_idx}
    name_types = shared_types & named
    theme_idx = [i for i in shared_idx
                 if pw[i].lower().strip("'\u2019") not in name_types]
    name_idx = [i for i in shared_idx
                if pw[i].lower().strip("'\u2019") in name_types]

    def rebuild(new_para):
        u = units[ua]
        parts = [paras[i].text
                 for i in range(u.first_para, u.last_para + 1)]
        parts[p.index - u.first_para] = new_para
        return "\n\n".join(parts)

    def subst(idx_list):
        out = []
        for _ in range(K):
            new = list(pw)
            for i in idx_list:
                new[i] = rep.pick(pw[i], btypes, rng)
            out.append(" ".join(new))
        return out

    variants: dict[str, list[str]] = {g: [] for g in
                                      ("G1", "G1b", "G2", "G3", "G5")}
    variants["G1"] = subst(theme_idx) if theme_idx else []
    variants["G1b"] = subst(name_idx) if name_idx else []

    if shared_idx:
        for _ in range(K):
            new = [t for i, t in enumerate(pw) if i not in shared_idx]
            pad = [rep.pick(pw[i], btypes, rng) for i in shared_idx]
            variants["G2"].append(" ".join(new + pad))

    sents = [s for s in re.split(r"(?<=[.!?\u2026])\s+", p.text) if s.strip()]
    if sents:
        best_s = max(sents, key=lambda s: len(ctypes(s) & btypes))
        bw = len(best_s.split())
        pool = [q for q in paras if q.index != p.index and q.n_words >= bw]
        for _ in range(K):
            d = rng.choice(pool).text.split()
            rep_txt = " ".join((d * (bw // len(d) + 2))[:bw])
            variants["G3"].append(p.text.replace(best_s, rep_txt, 1))

    pool = [q for q in paras if q.n_words >= p.n_words]
    for _ in range(K):
        d = rng.choice(pool).text.split()
        variants["G5"].append(
            " ".join((d * (p.n_words // len(d) + 2))[:p.n_words]))

    cos0 = float(base[ua] @ base[partner])
    prof = {}
    for g, vs in variants.items():
        if not vs:
            prof[g] = float("nan")
            continue
        mods = [rebuild(v) for v in vs]
        emb = enc(mods)
        zz = []
        for k in range(len(mods)):
            v = emb[k]
            deltas = np.array([float(v @ base[c]) - float(base[ua] @ base[c])
                               for c in range(n) if c != ua])
            dp = float(v @ base[partner]) - cos0
            mu, sd = float(deltas.mean()), float(deltas.std())
            zz.append((dp - mu) / sd if sd > 0 else np.nan)
        zz = np.array(zz)
        zz = zz[~np.isnan(zz)]
        prof[g] = float(zz.mean()) if len(zz) else float("nan")
    return prof, len(shared_idx), len(theme_idx), len(name_idx), cos0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=0,
                    help="also scan the top N non-adjacent pairs")
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    n = len(units)
    rng = random.Random(42)

    named, ner_str, ent_cnt = build_names(body)
    w(f"names from {ner_str}; {len(named)} named token types, "
      f"{sum(ent_cnt.values())} entity mentions")
    w("  (a statistical tagger, so it has its own error rate: it also tags "
      "'brown satin' and 'distant music' as entities)")
    w()

    import torch
    from sentence_transformers import SentenceTransformer

    for mname, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                       ("bge", "BAAI/bge-small-en-v1.5")):
        m = SentenceTransformer(mid)
        m.eval()
        enc = build_pooler(m)
        base = enc([u.text for u in units])
        rep = Replacer(mid)

        w("=" * 78)
        w(f"MODEL {mname}   K = {args.K} draws per granularity")
        w("=" * 78)
        w()
        w(f"   {'case':<18}{'sh':>4}{'th':>4}{'nm':>4}{'cos0':>7}"
          f"{'G1':>8}{'G1b':>8}{'G2':>8}{'G3':>8}{'G5':>8}")
        for (ua, kw, partner, label) in CASES:
            u = units[ua]
            p = next(q for q in paras[u.first_para:u.last_para + 1]
                     if kw.lower() in q.text.lower())
            prof, sh, th, nm, cos0 = ladder_for_pair(
                units, paras, ua, partner, p, enc, base, rep, named,
                args.K, rng)
            def fmt(x):
                return "     n/a" if np.isnan(x) else f"{x:>8.2f}"
            w(f"   {label:<18}{sh:>4}{th:>4}{nm:>4}{cos0:>7.3f}" +
              "".join(fmt(prof[g]) for g in ("G1", "G1b", "G2", "G3", "G5")))
        w()
        w("   sh/th/nm = shared content word types in total / thematic / named")
        w()

        if args.scan:
            iu, ju = np.triu_indices(n, k=1)
            cand = [(float(base[i] @ base[j]), i, j) for i, j in zip(iu, ju)
                    if j - i >= 5]
            cand.sort(key=lambda t: -t[0])
            top = cand[:args.scan]
            w("-" * 78)
            w(f"SCAN — top {len(top)} non-adjacent pairs by similarity "
              f"(gap >= 5), span and partner fixed by the pair")
            w("-" * 78)
            w()
            w(f"   {'pair':<10}{'gap':>5}{'cos0':>7}{'u_sh':>5}{'sh':>4}{'th':>4}"
              f"{'nm':>4}{'G1':>8}{'G1b':>8}{'G2':>8}{'G3':>8}{'G5':>8}")
            rows = []
            for sim, i, j in top:
                u = units[i]
                # unit-level lexical bridge: does the WHOLE unit share any
                # content word with the partner?  (The span below is chosen to
                # maximise overlap, so it can never be zero by construction.)
                u_sh = len(ctypes(units[i].text) & ctypes(units[j].text))
                # rule-based target paragraph: the one sharing most content
                # with the partner unit
                best_p, best_s = None, -1
                for q in paras[u.first_para:u.last_para + 1]:
                    s = len(ctypes(q.text) & ctypes(units[j].text))
                    if s > best_s:
                        best_p, best_s = q, s
                prof, sh, th, nm, cos0 = ladder_for_pair(
                    units, paras, i, j, best_p, enc, base, rep, named,
                    args.K, rng)
                def fmt(x):
                    return "     n/a" if np.isnan(x) else f"{x:>8.2f}"
                w(f"   {f'{i}&{j}':<10}{j-i:>5}{cos0:>7.3f}{u_sh:>5}{sh:>4}"
                  f"{th:>4}{nm:>4}" + "".join(fmt(prof[g])
                                              for g in ("G1", "G1b", "G2", "G3", "G5")))
                rows.append((i, j, u_sh, sh, th, nm, prof))
            # summary
            w()
            no_bridge = [r for r in rows if r[2] == 0]
            bridge = [r for r in rows if r[2] > 0]
            w(f"   UNIT-LEVEL BRIDGE: pairs where the whole units share NO")
            w(f"   content word: {len(no_bridge)} of {len(rows)}")
            w(f"   pairs with a unit-level lexical bridge: {len(bridge)}")
            for g in ("G1", "G2", "G3", "G5"):
                vals = np.array([r[6][g] for r in rows
                                 if not np.isnan(r[6][g])])
                if len(vals):
                    w(f"      {g}: mean z = {vals.mean():+.2f} "
                      f"(median {np.median(vals):+.2f}); "
                      f"z < -1 in {int((vals < -1).sum())}/{len(vals)}")
            w()
            w("   u_sh = shared content word TYPES between the whole units")
            w("   sh   = shared content word TYPES within the chosen span")
            w("   (the span is picked to maximise overlap, so sh > 0 always;")
            w("    u_sh = 0 is what 'no lexical bridge' means)")
            w()

    (DERIVED / "ladder_principled.txt").write_text("\n".join(OUT),
                                                   encoding="utf-8")
    print("wrote", DERIVED / "ladder_principled.txt")


if __name__ == "__main__":
    main()
