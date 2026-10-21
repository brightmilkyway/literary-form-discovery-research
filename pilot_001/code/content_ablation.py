"""How much of each finding lives in the *content words*?

The ablation of (2,73) showed that replacing every content word in both
passages barely moved the cosine.  If that also holds for the findings we
called robust, then the encoder's similarity is largely a function of
function words / syntax / length rather than of meaning.

Protocol: for each candidate pair, replace every content word (any word type
not in a stop list) with the placeholder "thing" in BOTH passages, re-encode,
and report the cosine drop.  Two reference points:

* ``self``  — ablation should barely move a passage's similarity to *itself*
  (same form), so this calibrates the floor.
* ``null``  — the same ablation applied to randomly chosen distant pairs gives
  the distribution of drops expected if the pair carries no content signal.

Writes ``data/derived/content_ablation.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

PAIRS = [
    (2, 73, "'snow frame' (fails cross-model)"),
    (30, 39, "religion / Church thread"),
    (55, 58, "song lyric vs. song name"),
    (12, 16, "Mr Browne & Freddy"),
    (71, 74, "cold interrogation vs. Gretta sobbing"),
    (37, 58, "singers past vs. song name"),
    (69, 72, "Gabriel asks / Gretta's revelation"),
]

STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while during yes o oh well said asked cried says say""".split())


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def ablate(text: str, mode: str, rng: random.Random | None = None) -> str:
    """mode='all' replaces every content-word token; 'random' replaces a
    random half of the content-word *types*; 'none' returns the text."""
    types = [t for t in {m.group(0) for m in re.finditer(r"[A-Za-z][A-Za-z'\u2019-]*", text)}
             if t.lower().strip("'") not in STOP]
    if mode == "none":
        return text
    if mode == "all":
        kill = set(t.lower() for t in types)
    else:
        types = list(types)
        rng.shuffle(types)
        kill = set(t.lower() for t in types[:max(1, len(types) // 2)])
    return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                  lambda m: "thing" if m.group(0).lower() in kill else m.group(0),
                  text)


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    n = len(units)

    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("all-MiniLM-L6-v2")
    m.eval()

    cache: dict[str, np.ndarray] = {}

    def emb(text: str) -> np.ndarray:
        if text not in cache:
            with torch.no_grad():
                cache[text] = m.encode([text], convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False)[0]
        return cache[text]

    def cos(a: str, b: str) -> float:
        return float(emb(a) @ emb(b))

    w("=" * 78)
    w("CONTENT-WORD ABLATION: how much of each finding is *meaning*?")
    w("=" * 78)
    w()

    # ---- calibration: self-similarity floor ----------------------------
    w("CALIBRATION — ablation applied to a passage paired with *itself*")
    for k in (2, 73, 39, 58):
        t = units[k].text
        w(f"   unit {k}: cos(self) = {cos(t, t):.4f} -> after full ablation "
          f"{cos(ablate(t, 'all'), ablate(t, 'all')):.4f}   "
          f"(self-identity is preserved by construction)")
    w()

    # ---- the candidate pairs -------------------------------------------
    w("CANDIDATE PAIRS — cosine before / after full content-word ablation")
    w(f"   {'pair':<9}{'before':>9}{'after':>9}{'drop':>9}   note")
    rows = []
    for (i, j, note) in PAIRS:
        a, b = units[i].text, units[j].text
        c0 = cos(a, b)
        c1 = cos(ablate(a, "all"), ablate(b, "all"))
        w(f"   {f'{i}&{j}':<9}{c0:>9.3f}{c1:>9.3f}{c0 - c1:>9.3f}   {note}")
        rows.append({"pair": f"{i}&{j}", "before": round(c0, 4),
                     "after": round(c1, 4), "drop": round(c0 - c1, 4),
                     "note": note})
    w()

    # ---- null distribution ---------------------------------------------
    w("NULL DISTRIBUTION — the same ablation on 60 random distant pairs")
    rng = random.Random(42)
    drops, raw_drops = [], []
    tries = 0
    while len(drops) < 60 and tries < 2000:
        tries += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if j - i < 40:
            continue
        a, b = units[i].text, units[j].text
        c0 = cos(a, b)
        c1 = cos(ablate(a, "all"), ablate(b, "all"))
        drops.append(c0 - c1)
        raw_drops.append(c0)
    drops = np.array(drops)
    raw_drops = np.array(raw_drops)
    w(f"   drop: mean {drops.mean():+.3f}, sd {drops.std():.3f}, "
      f"min {drops.min():+.3f}, max {drops.max():+.3f}")
    w(f"   mean raw cos of those pairs: {raw_drops.mean():.3f}")
    w(f"   -> a pair whose drop is *far below* this mean has its similarity")
    w(f"      carried by form, not content.")
    w()
    w("   Where each candidate sits in that null distribution:")
    for (i, j, note) in PAIRS:
        a, b = units[i].text, units[j].text
        d = cos(a, b) - cos(ablate(a, "all"), ablate(b, "all"))
        pct = float((drops < d).mean()) * 100
        w(f"      ({i},{j}) drop {d:+.3f}  -> percentile {pct:5.1f}%  "
          f"{'content-driven' if pct > 66 else ('form-driven' if pct < 33 else 'mixed')}")

    import csv
    with (C.TABLE_DIR / "content_ablation.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    (DERIVED / "content_ablation.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", DERIVED / "content_ablation.txt")


if __name__ == "__main__":
    main()
