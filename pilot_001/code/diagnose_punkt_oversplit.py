"""Is punkt over-splitting at '?" said X'?  Test it directly.

The spot-check found 50 units that begin with a lowercase letter, all of them
dialogue tags ("said Aunt Kate.", "asked Gabriel.").  In the source these are
part of the preceding quoted sentence:

    “But you will come, won’t you?” said Miss Ivors, laying her warm hand
    eagerly on his arm.

This script tests punkt on such sentences, in the original (curly) and
normalised (straight) forms, to find out whether the over-split is caused by
the normalisation, by the paragraph-first stage, or by punkt itself.
"""

from __future__ import annotations

import re

import nltk

import common as C
import sentences as S

body = C.strip_title(C.load_raw())[0]

CASES = [
    "“But you will come, won’t you?” said Miss Ivors, laying her warm hand "
    "eagerly on his arm.",
    "“Is it snowing again, Mr Conroy?” asked Lily.",
    "“Here I am as right as the mail, Aunt Kate! Go on up. I’ll follow,” "
    "called out Gabriel from the dark.",
    "“O, Mr D’Arcy,” said Aunt Kate, “now that was a great fib to tell.”",
]

print("=" * 78)
print("A. punkt on individual sentences, original vs normalised")
print("=" * 78)
for c in CASES:
    for label, text in (("curly ", c), ("straight", S.normalise_quotes(c))):
        parts = nltk.sent_tokenize(text)
        print(f"  [{label}] {len(parts)} part(s)")
        for p in parts:
            print(f"        | {p}")
    print()

print("=" * 78)
print("B. where the two-stage splitter cuts, for the same sentences")
print("=" * 78)
for c in CASES:
    # emulate: paragraph-first is a no-op here, then normalise, then punkt
    norm = S.normalise_quotes(c)
    parts = nltk.sent_tokenize(norm)
    print(f"  input : {c}")
    print(f"  cuts  : {len(parts)}")
    for p in parts:
        print(f"        | {p}")
    print()

print("=" * 78)
print("C. how many '?”/'!” + lowercase tag' occurrences are in the text?")
print("=" * 78)
pat = re.compile(r"[?!]\u201d\s+([a-z])")
hits = pat.findall(body)
print(f"  occurrences of [?!]” followed by a lowercase word: {len(hits)}")
pat2 = re.compile(r"\.\u201d\s+([a-z])")
print(f"  occurrences of .” followed by a lowercase word: "
      f"{len(pat2.findall(body))}")

print()
print("=" * 78)
print("D. does punkt join them if the closing quote is REMOVED?")
print("=" * 78)
for c in CASES[:2]:
    stripped = c.replace("\u201d", "")
    print(f"  {stripped}")
    print(f"     -> {len(nltk.sent_tokenize(stripped))} part(s): "
          f"{nltk.sent_tokenize(stripped)}")
