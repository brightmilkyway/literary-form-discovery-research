"""Show the exact text of unit 55 so the lyric boundary is unambiguous."""

from __future__ import annotations

import common as C

body = C.strip_title(C.load_raw())[0]
units = C.build_units(body, "group", target_words=150)
u = units[55]
print(f"unit 55: paras {u.first_para}-{u.last_para}, chars {u.start}-{u.end}, "
      f"{u.n_words} words")
print()
print("--- unit text, repr (first 320 chars) ---")
print(repr(u.text[:320]))
print()
paras = C.paragraphs(body)
print("--- its paragraphs ---")
for p in paras[u.first_para:u.last_para + 1]:
    print(f"  para {p.index} ({p.n_words}w, chars {p.start}-{p.end}):")
    print(f"    {repr(p.text[:200])}")
