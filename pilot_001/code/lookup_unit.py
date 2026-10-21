"""Print the full text of chosen analysis units, with paragraph ranges.

Used while writing the reports so that every quotation is verified against the
source and can be located by paragraph number.

  python lookup_unit.py 2 3 25 39 55 58 72 73
"""

from __future__ import annotations

import json
import re
import sys

import common as C

DERIVED = C.DATA_DIR / "derived"


def main():
    idxs = [int(a) for a in sys.argv[1:] if a.isdigit()] or [2, 73]
    out_path = C.DATA_DIR / "derived" / "unit_lookup.txt"
    units = json.loads((DERIVED / "units.json").read_text(encoding="utf-8"))["group150"]
    body = C.strip_title(C.load_raw())[0]
    lines = []
    for i in idxs:
        u = units[i]
        t = re.sub(r"\s+", " ", body[u["start"]:u["end"]]).strip()
        lines.append(f"\n===== unit {i}  |  paras {u['first_para']}-{u['last_para']}  "
                     f"|  {u['n_words']} words  |  chars {u['start']}-{u['end']} =====")
        lines.append(t)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
