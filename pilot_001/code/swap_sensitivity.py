"""Swap sensitivity, scale-corrected.

Problem with the first version
------------------------------
It compared ``excess = |delta_partner| - mean|delta_unrelated|`` across seven
cases.  Two scale inconsistencies make that comparison unfair:

  1. the cases replace different AMOUNTS of text (the lyric is 18 of 164 words,
     i.e. 11%, while some motif paragraphs are much longer), so the global
     perturbation differs by a factor of ~3.5 across cases;
  2. the raw excess is in cosine units, and MiniLM and bge do not share a
     cosine scale (MiniLM spans roughly 0.2-0.8, bge 0.4-0.9), so the two
     models cannot be compared in raw units either.

Corrections
-----------
* **Normalise within each case and each model.**  For a swap in unit A, compute
  ``delta(C) = cos(A', C) - cos(A, C)`` for ALL other units C, then report the
  partner's **z-score** and **percentile** in that distribution.  A z-score is
  in standard deviations, so it is comparable across cases and across models.
* **Report the replaced fraction** so the reader can see whether it confounds.
* **Fraction-matched variant**: also run every case at a fixed replacement of
  exactly 18 words (the lyric's size), so all cases are on the same footing
  regardless of how long their motif paragraph is.
* **Direction matters**: for a content-driven pair the partner's similarity
  should DROP, so a negative delta and a low percentile are the signal.

Writes ``data/derived/swap_sensitivity.txt`` (replacing the uncorrected run).
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

CASES = [
    (55, "rain falls", 58, "the lyric"),
    (72, "Michael Furey", 69, "Michael Furey"),
    (72, "Michael Furey", 73, "Michael Furey"),
    (39, "monks", 30, "the monks"),
    (16, "lemonade", 12, "the lemonade"),
    (74, "sobbing", 71, "sobbing"),
    (58, "Lass of Aughrim", 55, "the song name"),
]


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


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
            e = model.encode(subs, batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        out = np.zeros((len(texts), e.shape[1]))
        for k, o in enumerate(owner):
            out[o] += e[k]
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out
    return enc


def unrelated_words(donor: str, k: int, rng: random.Random) -> str:
    ws = donor.split()
    out = []
    while len(out) < k:
        out += ws
    return " ".join(out[:k])


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

    # donors: paragraphs far from every case, used as unrelated prose
    donors = [p for p in paras if p.n_words >= 12]

    import torch
    from sentence_transformers import SentenceTransformer

    base_texts = [u.text for u in units]

    for mode in ("full paragraph", "fixed 18 words"):
        for mname, mid in (("MiniLM", "all-MiniLM-L6-v2"),
                           ("bge", "BAAI/bge-small-en-v1.5")):
            m = SentenceTransformer(mid)
            m.eval()
            enc = build_pooler(m)
            base = enc(base_texts)
            w("=" * 78)
            w(f"MODE: {mode}    MODEL: {mname}")
            w("=" * 78)
            w()
            w(f"   {'swap':<18}{'partner':>8}{'repl%':>7}{'delta_p':>9}"
              f"{'mean_d':>8}{'sd_d':>7}{'z_p':>7}{'pctile':>8}")
            rows = []
            for (ua, kw, partner, label) in CASES:
                p = motif_para(ua, kw)
                if p is None:
                    w(f"   {label:<18}  (paragraph not found)")
                    continue
                u = units[ua]
                if mode == "full paragraph":
                    repl_n = p.n_words
                    tail_words = []
                else:
                    repl_n = min(18, p.n_words)
                    tail_words = p.text.split()[repl_n:]
                donor = max(donors, key=lambda q: abs(q.start - p.start))
                repl = unrelated_words(donor.text, repl_n, rng)
                new_para = " ".join([repl] + tail_words)

                parts = [paras[i].text
                         for i in range(u.first_para, u.last_para + 1)]
                parts[p.index - u.first_para] = new_para
                mod = "\n\n".join(parts)

                v = enc([mod])[0]
                deltas = np.array([float(v @ base[c]) - float(base[ua] @ base[c])
                                   for c in range(n) if c != ua])
                idx = [c for c in range(n) if c != ua]
                dp = float(v @ base[partner]) - float(base[ua] @ base[partner])
                mu, sd = float(deltas.mean()), float(deltas.std())
                z = (dp - mu) / sd if sd > 0 else float("nan")
                pct = float((deltas < dp).mean()) * 100
                frac = 100.0 * repl_n / u.n_words
                w(f"   {label:<18}{partner:>8}{frac:>6.0f}%{dp:>+9.3f}"
                  f"{mu:>+8.3f}{sd:>7.3f}{z:>7.2f}{pct:>7.1f}%")
                rows.append((label, partner, frac, dp, mu, sd, z, pct))
            w()
            w("   delta_p  = change in cos(A,B) after the swap (negative = the")
            w("              partner was damaged)")
            w("   mean_d/sd_d = mean and sd of the delta over ALL other units")
            w("   z_p      = (delta_p - mean_d) / sd_d   <- scale-free, so it")
            w("              is comparable across cases AND across models")
            w("   pctile   = percentile of delta_p; low = the partner is among")
            w("              the most damaged units")
            w()
            # rank the cases by z, most damaged first
            order = sorted(rows, key=lambda r: r[6])
            w("   ranking by z (most damaged partner first):")
            for r in order:
                w(f"      z={r[6]:+6.2f}  pctile {r[7]:5.1f}%  "
                  f"{r[0]:<18} -> {r[1]}")
            w()

    (DERIVED / "swap_sensitivity.txt").write_text("\n".join(OUT),
                                                  encoding="utf-8")
    print("wrote", DERIVED / "swap_sensitivity.txt")


if __name__ == "__main__":
    main()
