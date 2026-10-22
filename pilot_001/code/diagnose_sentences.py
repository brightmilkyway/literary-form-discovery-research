"""Diagnose the sentence splitter: are the "sentences" really sentences?

Raises: a reviewer noticed that the unit labelled sentence 534 in
``verbatim_echo_test.txt`` actually contains three separate sentences.  This
script measures how often nltk's punkt merges multiple sentences into one unit,
and why.
"""

from __future__ import annotations

import re

import common as C

body = C.strip_title(C.load_raw())[0]

import nltk

units = nltk.sent_tokenize(body)
print(f"nltk.sent_tokenize -> {len(units)} units")

# a unit is suspect if it contains terminal punctuation followed by whitespace
# that includes a newline, or by a closing quote then whitespace+newline
TERM = r"[.!?\u2026]"
CLOSE = r"[\u201d\"\u2019'\)\]]*"
suspect = re.compile(TERM + CLOSE + r"\s*\n")
bad = [i for i, u in enumerate(units) if suspect.search(u)]
print(f"units containing an internal line break after terminal punctuation: {len(bad)}")
print(f"  first 15 indices: {bad[:15]}")

# also count units with an internal '.”' or '."' (the curly-quote case)
curly = re.compile(r"[.!?]\u201d")
n_curly = sum(1 for u in units if curly.search(u))
print(f"units containing '.\\u201d' (period + closing curly quote): {n_curly}")

print()
for i in bad[:3]:
    print(f"--- unit {i} (chars {len(units[i])}) ---")
    print(repr(units[i][:420]))
    print()

print("=" * 70)
print("CONTROL: split on blank lines FIRST, then punkt within each paragraph")
paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
print(f"paragraphs (blank-line split): {len(paras)}")
inner = 0
total = 0
for p in paras:
    total += len(nltk.sent_tokenize(p))
print(f"sum of punkt units over paragraphs: {total}")

print()
print("CONTROL 2: normalise curly quotes to straight, then punkt")
norm = body.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
units2 = nltk.sent_tokenize(norm)
bad2 = [i for i, u in enumerate(units2) if suspect.search(u)]
print(f"normalised: {len(units2)} units, {len(bad2)} still suspect")

print()
print("CONTROL 3: a simple regex splitter (terminal punct + quote + whitespace)")
SIMPLE = re.compile(r"(?<=[.!?\u2026])[\u201d\"\u2019'\)\]]*\s+")
parts = [x.strip() for x in SIMPLE.split(body) if x.strip()]
bad3 = [i for i, u in enumerate(parts) if suspect.search(u)]
print(f"regex: {len(parts)} units, {len(bad3)} still suspect")
