"""Validate the sentence splitter and record the check.

Writes ``data/derived/sentence_validation.txt``, which contains:

* the unit count and the number of units that still look multi-sentence;
* the effect of the curly-quote fix, measured directly;
* the specific sentences of the snow echo, with their offsets, so a reader can
  verify in the source file that each is one sentence;
* a deterministic sample of 25 units, printed in full for manual inspection;
* the distribution of unit lengths.

Run from code/:  python validate_sentences.py
"""

from __future__ import annotations

import re

import common as C
import sentences as S

DERIVED = C.DATA_DIR / "derived"
OUT = []
PHRASE = "general all over Ireland"


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def main():
    body = C.strip_title(C.load_raw())[0]

    # ---- what the naive call produced, for the record -------------------
    import nltk
    naive = nltk.sent_tokenize(body)
    naive_bad = [i for i, u in enumerate(naive)
                 if re.search(r"[.!?\u2026][\u201d\"\u2019'\)\]]*\s*\n", u)]

    sents = S.split_sentences(body)
    under = S.under_split(sents)
    over = S.over_split(sents)
    bad = S.invalid_units(sents)

    w("=" * 78)
    w("SENTENCE SPLITTER VALIDATION")
    w("=" * 78)
    w()
    w("A reviewer observed that the unit printed as 'sentence 534' in an")
    w("earlier run contained three separate speeches.  Diagnosis and fix:")
    w()
    w(f"  nltk.sent_tokenize(body) directly      : {len(naive)} units, "
      f"{len(naive_bad)} containing an internal line break (UNDER-split)")
    w(f"  split_sentences(body)  [this module]   : {len(sents)} units, "
      f"{len(under)} under-split, {len(over)} over-split")
    w()
    w("  Cause of the under-split: the text writes '.”' (period + closing")
    w("  CURLY quote) before a blank line, and punkt does not treat '.”' as")
    w("  a sentence end, so consecutive speeches were merged.")
    w("  Fix: split on blank lines first, then punkt inside each paragraph.")
    w()
    w("  A first fix attempt normalised curly quotes to straight.  That")
    w("  trades the under-split for an OVER-split: punkt treats '?”' as")
    w("  mid-sentence (correct) but '?\"' as a sentence end, so every")
    w("  dialogue tag was cut off its speech:")
    w("      “Is it snowing again, Mr Conroy?” asked Lily.   -> 1 part (curly)")
    w("      \"Is it snowing again, Mr Conroy?\" asked Lily.   -> 2 parts (straight)")
    w("  The module therefore keeps the original curly quotes.")
    w()
    w(f"  under-split units: {len(under)}  {under[:10]}")
    w(f"  over-split units : {len(over)}  {over[:10]}")
    w()

    # ---- the echo sentences --------------------------------------------
    w("-" * 78)
    w("THE SNOW ECHO, AS NOW SPLIT")
    w("-" * 78)
    hits = [s for s in sents if PHRASE in s.text]
    w(f"units containing {PHRASE!r}: {[s.index for s in hits]}")
    for s in hits:
        w(f"  [{s.index}] chars {s.start}-{s.end}  ({s.n_words} words)")
        w(f"        {s.text.strip()}")
    w()
    w("  Verify in data/source/the_dead.txt at the given character offsets:")
    for s in hits:
        w(f"        body[{s.start}:{s.end}] == the line above  ->  "
          f"{body[s.start:s.end] == s.text}")
    w()

    # ---- length distribution -------------------------------------------
    lens = [s.n_words for s in sents]
    lens_sorted = sorted(lens)
    w("-" * 78)
    w("UNIT LENGTH DISTRIBUTION")
    w("-" * 78)
    w(f"  units {len(sents)}; words {sum(lens)}")
    w(f"  words per unit: min {min(lens)}  median "
      f"{lens_sorted[len(lens)//2]}  max {max(lens)}")
    w(f"  units with < 3 words: {sum(1 for x in lens if x < 3)}")
    w()

    # ---- the splitter comparison table ----------------------------------
    w("-" * 78)
    w("SPLITTER COMPARISON (for the record)")
    w("-" * 78)
    norm_body = S.normalise_quotes(body)
    rows = [
        ("nltk.sent_tokenize(body)            ",
         nltk.sent_tokenize(body)),
        ("punkt on quote-normalised body      ",
         nltk.sent_tokenize(norm_body)),
        ("paragraph-first + punkt (curly)     ",
         [s.text for s in sents]),
    ]
    w(f"   {'method':<38}{'units':>7}{'under':>8}{'over':>7}")
    for label, units in rows:
        u = [S.Sentence(i, t, 0, 0) for i, t in enumerate(units)]
        w(f"   {label:<38}{len(units):>7}{len(S.under_split(u)):>8}"
          f"{len(S.over_split(u)):>7}")
    w("   (units are counted over the whole body; the module splits per")
    w("    paragraph first, which is what removes the under-splits)")
    w()

    # ---- manual sample -------------------------------------------------
    w("-" * 78)
    w("DETERMINISTIC SAMPLE FOR MANUAL INSPECTION (every 30th unit)")
    w("-" * 78)
    for s in sents[::30]:
        w(f"  [{s.index}] ({s.n_words}w) {re.sub(r'\s+', ' ', s.text)}")
    w()
    w("  Every entry above should be exactly one sentence.  The reviewer's")
    w("  check can be repeated by opening data/source/the_dead.txt and reading")
    w("  the text at each unit's character offsets.")

    (DERIVED / "sentence_validation.txt").write_text("\n".join(OUT),
                                                    encoding="utf-8")
    print(f"units={len(sents)}  invalid={len(bad)}  "
          f"(naive: {len(naive)} units, {len(naive_bad)} invalid)")
    print("wrote", DERIVED / "sentence_validation.txt")


if __name__ == "__main__":
    main()
