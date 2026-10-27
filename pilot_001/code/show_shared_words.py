"""Which words does each case's target paragraph actually share with its partner?

Needed to interpret the granularity ladder: the size of the word-level
intervention (n_sw) is what G1/G2 act on.
"""

from __future__ import annotations

import common as C

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


def ctypes(text):
    return {t for t in C.tokens(text) if t not in STOP and len(t) > 1}


def main():
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    for (ua, kw, partner, label) in CASES:
        u = units[ua]
        p = next(q for q in paras[u.first_para:u.last_para + 1]
                 if kw.lower() in q.text.lower())
        b = ctypes(units[partner].text)
        shared = sorted(ctypes(p.text) & b)
        # occurrences in the paragraph
        occ = [t for t in C.tokens(p.text) if t in set(shared)]
        print(f"{label:<18} -> unit {partner}")
        print(f"    target para {p.index} ({p.n_words}w): "
              f"{' '.join(p.text.split())[:100]}")
        print(f"    shared content TYPES ({len(shared)}): {shared}")
        print(f"    occurrences in the paragraph ({len(occ)}): {occ}")
        print()


if __name__ == "__main__":
    main()
