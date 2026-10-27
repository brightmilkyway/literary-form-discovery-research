"""What exactly are the 30 donor texts in swap_robustness.py?

Reproduces the donor selection and prints the actual paragraphs used, plus a
check on the assumption that they are "unrelated": how much content vocabulary
does each donor share with the partner unit?
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

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


def content(text):
    return {t for t in C.tokens(text) if t not in STOP and len(t) > 1}


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    rng = random.Random(42)

    print("=" * 78)
    print("DONOR POOL RULES (from swap_robustness.py)")
    print("=" * 78)
    print(f"  K (donors per case)      = {K}")
    print(f"  MIN_DONOR_WORDS          = {MIN_DONOR_WORDS}")
    print(f"  MIN_DONOR_GAP (paras)    = {MIN_DONOR_GAP}")
    print("  the donor is truncated to exactly the replaced span's word count")
    print("  (cycling its own words if the paragraph is shorter than that)")
    print()
    print("  => the donors are PARAGRAPHS OF THE SAME STORY, taken from at least")
    print("     100 paragraphs away.  They are not random words, not text I")
    print("     wrote, and not text from another author.")
    print()

    for (ua, kw, partner, label) in CASES:
        u = units[ua]
        p = None
        for q in paras[u.first_para:u.last_para + 1]:
            if kw.lower() in q.text.lower():
                p = q
                break
        repl_n = min(18, p.n_words)
        pool = [q for q in paras
                if q.n_words >= max(repl_n, MIN_DONOR_WORDS)
                and abs(q.index - p.index) > MIN_DONOR_GAP]
        donors = rng.sample(pool, min(K, len(pool)))
        print("=" * 78)
        print(f"CASE: {label} -> {partner}   (target para {p.index}, "
              f"{p.n_words} words; replacing {repl_n} words)")
        print(f"  pool size after the distance filter: {len(pool)}")
        print("=" * 78)
        partner_types = content(units[partner].text)
        for d in donors:
            shared = len(content(d.text) & partner_types)
            print(f"   para {d.index:>3} ({d.n_words:>3}w) "
                  f"shared content types with partner: {shared:>2}  |  "
                  f"{re.sub(r'\\s+', ' ', d.text)[:62]}")
        print()

    # how much do the donors overlap with the partner, on average?
    print("=" * 78)
    print("IS 'UNRELATED' FAIR?  donor-vs-partner content overlap")
    print("=" * 78)
    rng = random.Random(42)
    for (ua, kw, partner, label) in CASES:
        u = units[ua]
        p = next(q for q in paras[u.first_para:u.last_para + 1]
                 if kw.lower() in q.text.lower())
        repl_n = min(18, p.n_words)
        pool = [q for q in paras
                if q.n_words >= max(repl_n, MIN_DONOR_WORDS)
                and abs(q.index - p.index) > MIN_DONOR_GAP]
        donors = rng.sample(pool, min(K, len(pool)))
        pt = content(units[partner].text)
        mt = content(p.text)          # the paragraph being removed
        ov_d = np.mean([len(content(d.text) & pt) for d in donors])
        ov_m = len(mt & pt)
        print(f"   {label:<18} removed paragraph shares {ov_m:>2} content types "
              f"with the partner; the 30 donors share {ov_d:.1f} on average")
    print()
    print("   If the donors shared as much as the removed paragraph, the")
    print("   replacement would not actually remove the shared material and")
    print("   the damage would be understated.")


if __name__ == "__main__":
    main()
