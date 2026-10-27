"""Fetch and cache the external donor source: Dubliners minus 'The Dead'.

Why an external donor source is needed
--------------------------------------
The swap test replaces a span of unit A with text of the same length and asks
whether the similarity to B survives.  If the replacement is drawn from *The
Dead* itself, it brings the same characters, places and themes with it, so the
"removal" is only partial and the measurement is biased toward "no damage".
(The pilot hit exactly this: same-story donors shared 3.6 content word types
with the partner on average, while the lyric paragraph being removed shared 0.)

Donor sources are therefore treated as a LADDER, from "same story" to
"definitely unrelated", and the question becomes whether the result is
invariant along it.  This script provides the rung that controls for author,
book, period and register while removing the story's own content:

    other stories from the same book, same author, same period

Source: Project Gutenberg eBook #2814 (Dubliners, public domain).

Writes ``data/source/dubliners_other_stories.txt``.
Run from code/:  python fetch_external_donors.py
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import common as C

URL = "https://www.gutenberg.org/cache/epub/2814/pg2814.txt"
OUT_PATH = C.DATA_DIR / "source" / "dubliners_other_stories.txt"
START_MARK = "THE SISTERS"
END_MARK = "THE DEAD"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and "--force" not in sys.argv:
        print(f"cached already: {OUT_PATH} "
              f"({OUT_PATH.stat().st_size} bytes); use --force to refetch")
        return 0

    print(f"fetching {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as r:
        raw = r.read().decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    print(f"  downloaded {len(raw)} chars")

    # the story headings are the first all-caps line matching each title
    def first_heading(text: str) -> int:
        m = re.search(rf"(?m)^{re.escape(text)}\s*$", raw)
        if not m:
            raise SystemExit(f"heading not found: {text}")
        return m.start()

    a = first_heading(START_MARK)
    b = first_heading(END_MARK)
    body = raw[a:b]
    print(f"  keeping chars {a}:{b} = {len(body)} "
          f"(from '{START_MARK}' up to but not including '{END_MARK}')")

    # strip the Gutenberg licence block if it sits inside the range
    body = re.split(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", body)[0]

    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    words = sum(len(C.word_span(p)) for p in paras)
    print(f"  {len(paras)} paragraphs, {words} words")

    OUT_PATH.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
