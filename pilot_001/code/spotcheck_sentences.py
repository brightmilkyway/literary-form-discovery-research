"""Spot-check the new sentence splitter before trusting any sentence-scale result.

Checks
------
1.  Offset round-trip: body[s.start:s.end] must equal s.text for every unit.
2.  Context windows around the snow echo sentences, printed in full.
3.  A *random* sample of 30 units (the validation script used every 30th unit,
    which is systematic and can miss trouble).
4.  Suspicious patterns:
      * ends with a known abbreviation (Mr. / Mrs. / Dr. / St. / Miss.)
      * no terminal punctuation at all
      * starts with a lowercase word, a conjunction or a comma -> the previous
        split may have cut inside a sentence (OVER-splitting)
      * shorter than 3 words, or longer than 60 words
5.  Every unit that begins with a lowercase letter, listed in full: these are
    the strongest candidates for over-splitting.

Run from code/:  python spotcheck_sentences.py
"""

from __future__ import annotations

import random
import re

import common as C
import sentences as S

DERIVED = C.DATA_DIR / "derived"
OUT = []

ABBREV = re.compile(r"\b(Mr|Mrs|Dr|St|Miss|Messrs|Co|Rev|Hon)\.\s*$")
TERMINAL = re.compile(r"[.!?\u2026\u201d\"'\)\]]\s*$")
LOWER_START = re.compile(r"^[\u201c\"'\(\[]*([a-z])")
CONT_START = re.compile(r"^[\u201c\"'\(\[]*(and|but|or|nor|yet|so|because|which|"
                        r"that|who|whom|whose|where|when|while|as|than|then)\b",
                        re.I)
COMMA_START = re.compile(r"^[\u201c\"'\(\[]*,")


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    sents = S.split_sentences(body)
    n = len(sents)

    w("=" * 78)
    w("SENTENCE SPLITTER SPOT-CHECK")
    w("=" * 78)
    w()
    w(f"units: {n}   words: {sum(s.n_words for s in sents)}")

    # ---- 1. offset round-trip -------------------------------------------
    bad_rt = [s.index for s in sents if body[s.start:s.end] != s.text]
    w(f"1. offset round-trip failures: {len(bad_rt)}  {bad_rt[:10]}")
    # contiguity: consecutive units must not overlap and must be ordered
    gaps, overlaps = 0, 0
    for a, b in zip(sents, sents[1:]):
        if b.start < a.end:
            overlaps += 1
        elif b.start > a.end:
            gaps += 1
    w(f"   overlapping consecutive pairs: {overlaps}; "
      f"pairs with a gap between them: {gaps}")
    w()

    # ---- 2. context around the echo --------------------------------------
    w("-" * 78)
    w("2. FULL CONTEXT AROUND THE SNOW ECHO")
    w("-" * 78)
    for phrase, lo, hi in (("general all over Ireland", None, None),):
        hits = [s.index for s in sents if phrase in s.text]
        w(f"   units containing {phrase!r}: {hits}")
        for h in hits:
            w(f"   --- context {max(0, h-3)} .. {min(n-1, h+3)} ---")
            for s in sents[max(0, h - 3):min(n, h + 4)]:
                mark = " <== " if s.index == h else "      "
                w(f"   {mark}[{s.index:>4}] ({s.n_words:>2}w) "
                  f"{re.sub(r'\s+', ' ', s.text)}")
            w()
    w()

    # ---- 3. random sample ------------------------------------------------
    w("-" * 78)
    w("3. RANDOM SAMPLE OF 30 UNITS")
    w("-" * 78)
    rng = random.Random(20261022)
    for i in sorted(rng.sample(range(n), 30)):
        s = sents[i]
        w(f"   [{i:>4}] ({s.n_words:>2}w) {re.sub(r'\s+', ' ', s.text)}")
    w()

    # ---- 4. suspicious patterns -----------------------------------------
    w("-" * 78)
    w("4. SUSPICIOUS PATTERNS")
    w("-" * 78)
    abbrev = [s for s in sents if ABBREV.search(s.text)]
    no_term = [s for s in sents if not TERMINAL.search(s.text)]
    tiny = [s for s in sents if s.n_words < 3]
    huge = [s for s in sents if s.n_words > 60]
    lower = [s for s in sents if LOWER_START.match(s.text)]
    cont = [s for s in sents if CONT_START.match(s.text)]
    comma = [s for s in sents if COMMA_START.match(s.text)]
    w(f"   ending with an abbreviation : {len(abbrev)}")
    for s in abbrev[:8]:
        w(f"      [{s.index}] {re.sub(r'\s+', ' ', s.text)}")
    w(f"   no terminal punctuation    : {len(no_term)}")
    for s in no_term[:8]:
        w(f"      [{s.index}] {re.sub(r'\s+', ' ', s.text)}")
    w(f"   shorter than 3 words       : {len(tiny)}")
    for s in tiny[:10]:
        w(f"      [{s.index}] {re.sub(r'\s+', ' ', s.text)}")
    w(f"   longer than 60 words       : {len(huge)}")
    for s in huge[:5]:
        w(f"      [{s.index}] ({s.n_words}w) "
          f"{re.sub(r'\s+', ' ', s.text)[:120]}…")
    w(f"   starts with a lowercase letter : {len(lower)}   <- over-split?")
    for s in lower[:20]:
        w(f"      [{s.index}] {re.sub(r'\s+', ' ', s.text)}")
    w(f"   starts with a conjunction      : {len(cont)}   <- over-split?")
    for s in cont[:10]:
        w(f"      [{s.index}] {re.sub(r'\s+', ' ', s.text)}")
    w(f"   starts with a comma            : {len(comma)}   <- over-split?")
    for s in comma[:10]:
        w(f"      [{s.index}] {re.sub(r'\s+', ' ', s.text)}")
    w()

    # ---- 5. all lowercase-starting units ---------------------------------
    w("-" * 78)
    w("5. ALL UNITS BEGINNING WITH A LOWERCASE LETTER (over-split candidates)")
    w("-" * 78)
    for s in lower:
        w(f"   [{s.index:>4}] {re.sub(r'\s+', ' ', s.text)}")
    w()
    w("   Note: a lowercase start is normal when a sentence was broken by a")
    w("   *paragraph* break inside a quotation.  What matters is whether the")
    w("   preceding unit ends mid-clause.")

    (DERIVED / "sentence_spotcheck.txt").write_text("\n".join(OUT),
                                                    encoding="utf-8")
    print(f"units={n}  roundtrip_fail={len(bad_rt)}  lowercase_start={len(lower)}"
          f"  abbrev_end={len(abbrev)}  no_terminal={len(no_term)}")
    print("wrote", DERIVED / "sentence_spotcheck.txt")


if __name__ == "__main__":
    main()
