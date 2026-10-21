"""The 'song' cluster: is it just literal repetition?

(55,58) is the only pair that survives every test in this pilot (mutual rank
1 in MiniLM, 2 in bge).  The cheapest explanation is literal repetition: unit
55 contains the sung lyric fragment and unit 58 contains the song title, so a
lexical method will of course match them.

This script separates "the song as textual repetition" from "the song as a
discursive thread" by:

1.  music-vocabulary density per unit — is the cluster simply "the units that
    talk about music"?
2.  ablating the sung lyric from unit 55 and re-measuring the pair, against a
    matched random-ablation control;
3.  ablating the song-name / singing verbs from both units and re-measuring;
4.  listing exactly which content words the pair shares;
5.  tracing every occurrence of the song and of singing vocabulary in the text.

Writes ``data/derived/song_cluster_analysis.txt``.
"""

from __future__ import annotations

import json
import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

CLUSTER = [37, 49, 55, 56, 58, 69, 72, 73]
PAIR = (55, 58)

MUSIC = {"song", "songs", "sing", "sings", "singing", "sang", "sung", "singer",
         "singers", "voice", "voices", "air", "airs", "tune", "tunes", "piano",
         "music", "choir", "concert", "concerts", "soprano", "tenor", "ballad",
         "note", "notes", "melody", "academy", "piece", "pieces", "singers",
         "hummed", "hum", "humming", "quaver", "falsetto"}
STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while yes o oh well said asked cried says say""".split())


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def types_of(text):
    return {m.group(0) for m in re.finditer(r"[A-Za-z][A-Za-z'\u2019-]*", text)}


def content_types(text):
    return {t.lower().strip("'") for t in types_of(text)
            if t.lower().strip("'") not in STOP}


def replace(text, kill, filler="thing"):
    ks = {k.lower() for k in kill}
    return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                  lambda m: filler if m.group(0).lower() in ks else m.group(0),
                  text)


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    n = len(units)

    import torch
    from sentence_transformers import SentenceTransformer

    C.set_seed(42)
    m = SentenceTransformer("all-MiniLM-L6-v2")
    m.eval()
    paras = C.paragraphs(body)

    def split_paras(u):
        buf, cnt, out = [], 0, []
        for p in paras[u.first_para:u.last_para + 1]:
            if p.n_words > 130:
                toks = p.text.split()
                for k in range(0, len(toks), 130):
                    out.append(" ".join(toks[k:k + 130]))
                continue
            if cnt + p.n_words > 130 and buf:
                out.append("\n\n".join(buf))
                buf, cnt = [], 0
            buf.append(p.text)
            cnt += p.n_words
        if buf:
            out.append("\n\n".join(buf))
        return out

    def enc_unit(text):
        pieces = [text] if len(text.split()) <= 130 else [
            " ".join(text.split()[k:k + 130])
            for k in range(0, len(text.split()), 130)]
        with torch.no_grad():
            e = m.encode(pieces, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        v = e.sum(axis=0)
        return v / np.linalg.norm(v)

    base = np.array([enc_unit(u.text) for u in units])
    S = base @ base.T
    np.fill_diagonal(S, np.nan)

    w("=" * 78)
    w("THE 'SONG' CLUSTER — LITERAL REPETITION, OR A DISCURSIVE THREAD?")
    w("=" * 78)
    w()

    # ---- 1. music-vocabulary density -----------------------------------
    w("1. MUSIC-VOCABULARY DENSITY PER UNIT")
    dens = np.zeros(n)
    for i, u in enumerate(units):
        toks = [t.lower().strip("'") for t in re.findall(r"[A-Za-z]+", u.text)]
        dens[i] = sum(1 for t in toks if t in MUSIC) / max(1, len(toks)) * 100
    order = np.argsort(-dens)
    w("   top 12 units by music-word density (per 100 words):")
    for i in order[:12]:
        mark = "  <-- cluster" if int(i) in CLUSTER else ""
        w(f"      unit {i:>2}  {dens[i]:5.2f}  paras "
          f"{units[i].first_para}-{units[i].last_para}{mark}")
    w()
    w("   density of the cluster units:")
    for i in CLUSTER:
        r = int((dens > dens[i]).sum()) + 1
        w(f"      unit {i:>2}  {dens[i]:5.2f}  (rank {r}/{n})")
    w()
    w("   -> If the cluster were just 'the music units', every cluster unit")
    w("      would sit near the top.  Check the ranks above.")
    w()

    # ---- 2. lyric ablation ---------------------------------------------
    w("2. ABLATING THE SUNG LYRIC FROM UNIT 55")
    t55 = units[PAIR[0]].text
    # the verse fragment: three short lines before the prose resumes
    lyric = "O, the rain falls on my heavy locks And the dew wets my skin, My babe lies cold"
    has = lyric.split(",")[0][:20] in t55
    w(f"   lyric fragment present in unit {PAIR[0]}: {has}")
    if has:
        t55_no_lyric = t55.replace(lyric, " ")
    else:
        # fall back: drop the first 20 words
        t55_no_lyric = " ".join(t55.split()[20:])
    v = enc_unit(t55_no_lyric)
    others = np.arange(n) != PAIR[0]
    c_before = float((base[PAIR[0]] @ base[PAIR[1]].T))
    c_after = float((v @ base[PAIR[1]].T))
    w(f"   cos({PAIR[0]},{PAIR[1]}) before lyric ablation: {c_before:.3f}")
    w(f"   cos({PAIR[0]},{PAIR[1]}) after  lyric ablation: {c_after:.3f}")
    w(f"   change {c_after - c_before:+.3f}")
    w(f"   (matched control: deleting the same NUMBER of words from unit "
      f"{PAIR[0]}, 8 reps)")
    nwords = len(lyric.split())
    reps = []
    for r in range(8):
        rng = random.Random(200 + r)
        toks = t55.split()
        start = rng.randrange(0, max(1, len(toks) - nwords))
        t = " ".join(toks[:start] + toks[start + nwords:])
        reps.append(float(enc_unit(t) @ base[PAIR[1]].T))
    reps = np.array(reps)
    w(f"      random-deletion cos: mean {reps.mean():.3f}, sd {reps.std():.3f}, "
      f"range [{reps.min():.3f}, {reps.max():.3f}]")
    pct = float((reps < c_after).mean()) * 100
    w(f"      lyric ablation sits at the {pct:.0f}th percentile "
      f"(low = the lyric was holding the pair together)")
    w()

    # ---- 3. ablating singing vocabulary ---------------------------------
    w("3. ABLATING WORD CLASSES FROM BOTH UNITS (target: cos(55,58) = "
      f"{c_before:.3f})")
    NAMES = {"aunt", "kate", "julia", "mary", "jane", "gabriel", "gretta",
             "d'arcy", "d\u2019arcy", "o'callaghan", "o\u2019callaghan",
             "browne", "malins", "morkan", "miss", "mr", "mrs", "teddy"}
    for lab, kill in (("music words", MUSIC),
                      ("the song title", {"lass", "aughrim"}),
                      ("good-night words", {"good-night", "goodnight"}),
                      ("CHARACTER NAMES", NAMES),
                      ("names + good-night", NAMES | {"good-night"}),
                      ("all content words", content_types(t55)
                       | content_types(units[PAIR[1]].text))):
        a = replace(t55, kill)
        b = replace(units[PAIR[1]].text, kill)
        c = float(enc_unit(a) @ enc_unit(b))
        w(f"   remove {lab:<20} -> cos = {c:.3f}  (change {c - c_before:+.3f})")
    w()
    w("   matched control: replace the same NUMBER of random content-word")
    w("   types in both units, 8 reps")
    nkill = len(NAMES)
    reps2 = []
    for r in range(8):
        rng = random.Random(300 + r)
        ka = rng.sample(sorted(content_types(t55)), min(nkill,
                                                        len(content_types(t55))))
        kb = rng.sample(sorted(content_types(units[PAIR[1]].text)),
                        min(nkill, len(content_types(units[PAIR[1]].text))))
        reps2.append(float(enc_unit(replace(t55, ka))
                           @ enc_unit(replace(units[PAIR[1]].text, kb))))
    reps2 = np.array(reps2)
    w(f"      random-ablation cos: mean {reps2.mean():.3f}, "
      f"sd {reps2.std():.3f}, range [{reps2.min():.3f}, {reps2.max():.3f}]")
    w()

    # ---- 4. shared content words ----------------------------------------
    w("4. WHAT DO UNITS 55 AND 58 ACTUALLY SHARE?")
    for (a_, b_) in [(55, 58), (37, 58), (69, 72)]:
        ca, cb = content_types(units[a_].text), content_types(units[b_].text)
        shared = sorted(ca & cb)
        w(f"   ({a_},{b_}) shared content-word types ({len(shared)}): {shared}")
    w()

    # ---- 5. where does the song / singing vocabulary appear? ------------
    w("5. WHERE 'LASS OF AUGHRIM' AND SINGING VOCABULARY APPEAR")
    w("   (word-boundary match; an earlier substring match produced false")
    w("    positives on 'class' and 'glass')")
    for i, u in enumerate(units):
        t = u.text
        if re.search(r"\baughrim\b", t, re.I) or re.search(r"\blass\b", t, re.I):
            w(f"   [song named] unit {i} (paras {u.first_para}-{u.last_para}): "
              f"{re.sub(r'\\s+', ' ', t)[:220]}")
    w()
    w("   every unit containing 'sing'/'song'/'sang' (with density):")
    for i in range(n):
        if dens[i] > 0:
            w(f"      unit {i:>2}  dens {dens[i]:5.2f}  paras "
              f"{units[i].first_para}-{units[i].last_para}  |  "
              f"{re.sub(r'\\s+', ' ', units[i].text)[:90]}")
    w()

    # ---- 6. mutual rank of the cluster pairs, with and without the lyric
    w("6. MUTUAL RANKS INSIDE THE CLUSTER (MiniLM, correction-free)")
    w(f"   {'pair':<9}{'cos':>7}{'i->j':>7}{'j->i':>7}{'worse':>7}")
    for a_ in CLUSTER:
        for b_ in CLUSTER:
            if b_ <= a_:
                continue
            ra = int((np.delete(S[a_], a_) > S[a_, b_]).sum()) + 1
            rb = int((np.delete(S[b_], b_) > S[b_, a_]).sum()) + 1
            w(f"   {f'{a_}&{b_}':<9}{S[a_, b_]:>7.3f}{ra:>7}{rb:>7}"
              f"{max(ra, rb):>7}")

    (DERIVED / "song_cluster_analysis.txt").write_text("\n".join(OUT),
                                                       encoding="utf-8")
    print("wrote", DERIVED / "song_cluster_analysis.txt")


if __name__ == "__main__":
    main()
