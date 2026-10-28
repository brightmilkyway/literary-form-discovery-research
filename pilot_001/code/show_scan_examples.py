"""Pull one illustrative example of each depth class from the scan."""

from __future__ import annotations

import csv

import common as C

DERIVED = C.DATA_DIR / "derived"
body = C.strip_title(C.load_raw())[0]
units = C.build_units(body, "group", target_words=150)


def show(r, label):
    A, B = int(r["A"]), int(r["B"])
    print(f"--- {label} ---")
    print(f"   pair {A}&{B}  gap {B-A}  cos {float(r['cos0']):.3f}  "
          f"sh={r['sh']} th={r['th']} nm={r['nm']}")
    print(f"   G1={float(r['G1']):+.2f}  G1b={float(r['G1b']):+.2f}  "
          f"G2={float(r['G2']):+.2f}  G3={float(r['G3']):+.2f}  "
          f"G5={float(r['G5']):+.2f}")
    for tag, i in (("A", A), ("B", B)):
        t = " ".join(units[i].text.split())
        print(f"   {tag} (paras {units[i].first_para}-{units[i].last_para}): "
              f"{t[:210]}")
    print()


rows = list(csv.DictReader(
    (DERIVED / "scan_full_ladder_minilm.csv").open(encoding="utf-8-sig")))

for dep, n in (("word", 3), ("name", 2), ("sentence", 2), ("none", 3)):
    sel = [r for r in rows if r["depth"] == dep]
    sel.sort(key=lambda r: -float(r["cos0"]))
    for r in sel[:n]:
        show(r, f"depth = {dep}")
