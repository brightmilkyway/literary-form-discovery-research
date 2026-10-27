"""Swap robustness: repeat every case over many donors, and target the shared words.

Two weaknesses in the single-draw version:

  * each case used ONE donor paragraph, so a result could be an artefact of
    which unrelated text happened to be drawn;
  * the target span was fixed (the whole paragraph, or its first 18 words),
    which may not be where the shared material actually sits.

This version fixes both:

  * every case is repeated over K = 30 different donors, drawn from paragraphs
    at least 100 paragraphs away, and reported as a distribution;
  * two targeting modes:
      "whole paragraph"    - replace the entire motif paragraph;
      "most-shared 18 words" - within the motif paragraph, find the 18-word
        window that shares the most CONTENT word types with the partner, and
        replace that window.  This removes the material most likely to carry
        the link, so it is the strongest form of the test.

Because the statistic is a z-score computed against all other units within
each draw, a non-damaged partner has E[z] ~ 0 by construction, so the
comparison across cases and across draws is fair.

Writes ``data/derived/swap_robustness.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []
K = 30
MIN_DONOR_WORDS = 25
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


def content_types(text):
    return {t for t in C.tokens(text) if t not in STOP and len(t) > 1}


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


def most_shared_window(words, partner_types, width):
    """Index of the width-word window sharing most content types with partner."""
    if len(words) <= width:
        return 0
    best, best_i = -1, 0
    for i in range(0, len(words) - width + 1):
        win = {t for t in C.tokens(" ".join(words[i:i + width]))
               if t not in STOP and len(t) > 1}
        score = len(win & partner_types)
        if score > best:
            best, best_i = score, i
    return best_i


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    n = len(units)
    rng = random.Random(42)

    def motif_para(ua, kw):
        u = units[ua]
        for p in paras[u.first_para:u.last_para + 1]:
            if kw.lower() in p.text.lower():
                return p
        return None

    import torch
    from sentence_transformers import SentenceTransformer

    base_texts = [u.text for u in units]

    for mode in ("whole paragraph", "most-shared 18 words"):
        for mname, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                           ("bge", "BAAI/bge-small-en-v1.5")):
            m = SentenceTransformer(mid)
            m.eval()
            enc = build_pooler(m)
            base = enc(base_texts)
            w("=" * 78)
            w(f"MODE: {mode}   MODEL: {mname}   (K = {K} donors per case)")
            w("=" * 78)
            w()
            w(f"   {'swap':<18}{'partner':>8}{'cos0':>7}{'sh':>4}{'n':>4}"
              f"{'mean z':>8}{'sd':>7}{'min':>7}{'max':>7}{'< -1':>7}")
            for (ua, kw, partner, label) in CASES:
                p = motif_para(ua, kw)
                u = units[ua]
                pw = p.text.split()
                ptypes = content_types(units[partner].text)

                if mode == "whole paragraph":
                    repl_n = p.n_words
                    keep_head = 0
                else:
                    repl_n = min(18, p.n_words)
                    keep_head = most_shared_window(pw, ptypes, repl_n)

                # donor pool: distant, long enough, and -- crucially --
                # sharing no more content vocabulary with the partner than
                # the paragraph being removed does.  Without this filter the
                # replacement can INJECT related material instead of removing
                # it, which biases the measurement toward "no damage".
                max_share = len(content_types(p.text) & ptypes)
                pool = [q for q in paras
                        if q.n_words >= max(repl_n, MIN_DONOR_WORDS)
                        and abs(q.index - p.index) > MIN_DONOR_GAP
                        and len(content_types(q.text) & ptypes) <= max_share]
                if len(pool) < K:
                    # relax the distance requirement before relaxing overlap
                    pool = [q for q in paras
                            if q.n_words >= repl_n
                            and q.index != p.index
                            and len(content_types(q.text) & ptypes) <= max_share]
                donors = rng.sample(pool, min(K, len(pool)))

                variants = []
                for d in donors:
                    dw = d.text.split()
                    rep = " ".join((dw * (repl_n // len(dw) + 2))[:repl_n])
                    if mode == "whole paragraph":
                        new_para = rep
                    else:
                        new_para = " ".join(pw[:keep_head] + [rep]
                                            + pw[keep_head + repl_n:])
                    parts = [paras[i].text
                             for i in range(u.first_para, u.last_para + 1)]
                    parts[p.index - u.first_para] = new_para
                    variants.append("\n\n".join(parts))

                emb = enc(variants)
                cos0 = float(base[ua] @ base[partner])
                zs = []
                for k in range(len(variants)):
                    v = emb[k]
                    deltas = np.array([float(v @ base[c])
                                       - float(base[ua] @ base[c])
                                       for c in range(n) if c != ua])
                    dp = float(v @ base[partner]) - cos0
                    mu, sd = float(deltas.mean()), float(deltas.std())
                    zs.append((dp - mu) / sd if sd > 0 else np.nan)
                zs = np.array(zs)
                zs = zs[~np.isnan(zs)]
                frac = float((zs < -1).mean())
                w(f"   {label:<18}{partner:>8}{cos0:>7.3f}{max_share:>4}"
                  f"{len(donors):>4}{zs.mean():>8.2f}"
                  f"{zs.std():>7.2f}{zs.min():>7.2f}{zs.max():>7.2f}"
                  f"{frac*100:>6.0f}%")
            w()
            w("   cos0   = original cos(A,B)")
            w("   sh     = content word types the REMOVED paragraph shares with")
            w("            the partner; donors are constrained to share no more")
            w("            than this, so the swap removes rather than injects")
            w("   n      = number of donors actually available after the filter")
            w("   mean z = mean over the donors of the partner's z-score")
            w("   < -1   = share of donors for which the partner is more than")
            w("            1 sd more damaged than the average unit")
            w()
            w("   A content-carried link should be damaged in nearly every")
            w("   draw (large negative mean z, high '< -1' share); a")
            w("   situational link should hover near z = 0.")
            w()

    (DERIVED / "swap_robustness.txt").write_text("\n".join(OUT),
                                                 encoding="utf-8")
    print("wrote", DERIVED / "swap_robustness.txt")


if __name__ == "__main__":
    main()
