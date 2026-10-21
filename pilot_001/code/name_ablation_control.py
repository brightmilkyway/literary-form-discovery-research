"""Is 'the same cast of characters' a general driver of SBERT similarity?

The song-cluster analysis showed that (55,58) is held together by shared
character names, not by the song.  Before drawing any literary conclusion we
must know whether that is a general property of dialogue-heavy units in this
text — i.e. whether SBERT at this scale is largely a *cast-of-characters*
detector.

Protocol: for a set of pairs, compute cos before and after replacing every
character name (and title) with a placeholder in BOTH units, and compare the
drop against the drop from ablating the same NUMBER of random content-word
types (matched control).

Writes ``data/derived/name_ablation_control.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

NAMES = {"aunt", "kate", "julia", "mary", "jane", "gabriel", "gretta",
         "d'arcy", "d\u2019arcy", "o'callaghan", "o\u2019callaghan", "browne",
         "malins", "morkan", "miss", "mr", "mrs", "teddy", "lily", "ivors",
         "fulham", "conroy", "daly", "power", "furey", "michael", "patrick",
         "morkan's", "julia's"}

CANDIDATES = [(55, 58), (37, 58), (69, 72), (30, 39), (12, 16), (71, 74),
              (72, 73), (55, 56), (56, 58), (49, 72)]

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


def replace(text, kill):
    ks = {k.lower() for k in kill}
    return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                  lambda m: "person" if m.group(0).lower() in ks else m.group(0),
                  text)


def content_types(text):
    return {t.lower().strip("'") for t in
            re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", text)
            if t.lower().strip("'") not in STOP}


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

    cache: dict[str, np.ndarray] = {}

    def e(text: str) -> np.ndarray:
        if text not in cache:
            with torch.no_grad():
                cache[text] = m.encode([text], convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False)[0]
        return cache[text]

    def pair_stats(i, j, reps=5):
        a, b = units[i].text, units[j].text
        c0 = float(e(a) @ e(b))
        ca = float(e(replace(a, NAMES)) @ e(replace(b, NAMES)))
        nk = len(NAMES)
        drops = []
        for r in range(reps):
            rng = random.Random(400 + r)
            ka = rng.sample(sorted(content_types(a)),
                            min(nk, len(content_types(a))))
            kb = rng.sample(sorted(content_types(b)),
                            min(nk, len(content_types(b))))
            drops.append(float(e(replace(a, ka)) @ e(replace(b, kb))))
        drops = np.array(drops)
        return c0, ca, drops

    w("=" * 78)
    w("IS 'THE SAME CAST' A GENERAL DRIVER OF SIMILARITY?")
    w("=" * 78)
    w()
    w("   name ablation replaces every character name/title with 'person'.")
    w("   matched control ablates the same NUMBER of random content types.")
    w()
    w(f"   {'pair':<9}{'cos':>7}{'after names':>13}{'drop':>8}"
      f"{'control mean':>14}{'below ctrl?':>13}")

    rows = []
    for (i, j) in CANDIDATES:
        c0, ca, ctrl = pair_stats(i, j)
        drop = c0 - ca
        below = ca < ctrl.min()
        w(f"   {f'{i}&{j}':<9}{c0:>7.3f}{ca:>13.3f}{drop:>8.3f}"
          f"{ctrl.mean():>14.3f}{('YES' if below else 'no'):>13}")
        rows.append({"pair": f"{i}&{j}", "cos": round(c0, 4),
                     "after_name_ablation": round(ca, 4),
                     "drop": round(drop, 4),
                     "control_mean": round(float(ctrl.mean()), 4),
                     "below_control_min": bool(below)})
    w()

    # ---- null sample: random distant pairs ------------------------------
    w("NULL SAMPLE — 40 random distant pairs (gap>=3)")
    rng = random.Random(7)
    seen, samples = set(), []
    while len(samples) < 40:
        i, j = rng.randrange(n), rng.randrange(n)
        if j - i < 3 or (i, j) in seen:
            continue
        seen.add((i, j))
        samples.append((i, j))
    frac_drop, frac_below = [], []
    for (i, j) in samples:
        c0, ca, ctrl = pair_stats(i, j, reps=3)
        if c0 > 0:
            frac_drop.append((c0 - ca) / c0)
        frac_below.append(1.0 if ca < ctrl.min() else 0.0)
    frac_drop = np.array(frac_drop)
    w(f"   relative drop from name ablation: mean "
      f"{frac_drop.mean() * 100:.1f}%, median "
      f"{np.median(frac_drop) * 100:.1f}%, "
      f"max {frac_drop.max() * 100:.1f}%")
    w(f"   share of pairs where name ablation goes BELOW the matched control "
      f"range: {np.mean(frac_below) * 100:.0f}%")
    w()
    w("   -> If most pairs behave this way, SBERT at 150-word scale is to a")
    w("      large extent a cast-of-characters detector, and 'the same scene'")
    w("      rather than 'the same theme' is what it measures.")

    import csv
    with (C.TABLE_DIR / "name_ablation_control.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    (DERIVED / "name_ablation_control.txt").write_text("\n".join(OUT),
                                                       encoding="utf-8")
    print("wrote", DERIVED / "name_ablation_control.txt")


if __name__ == "__main__":
    main()
