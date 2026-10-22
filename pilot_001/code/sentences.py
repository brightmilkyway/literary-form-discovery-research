"""Sentence units with *verified* boundaries.

Why this module exists
----------------------
An earlier version of this pilot called ``nltk.sent_tokenize(body)`` directly
and got 764 units, of which **152 contained internal line breaks** — they were
multi-sentence blocks, not sentences.  The cause is that the text writes
``.”`` (period + closing *curly* quote) before a blank line, and punkt does not
treat ``.”`` as a sentence end, so three consecutive speeches were returned as
one unit.  A reviewer caught this by noticing that the unit printed as
"sentence 534" in ``verbatim_echo_test.txt`` held three separate speeches.

Fix: two stages, both of which are part of the text's own structure.

1.  Split on blank lines — the paragraph breaks Joyce's text already has.  This
    removes the under-splitting (the ``.”`` + blank-line case).
2.  Within each paragraph, run punkt on the **original curly-quoted text**.
    Quote normalisation must NOT be used here: punkt treats ``?”`` as
    mid-sentence (correct: ``“Is it snowing again, Mr Conroy?” asked Lily.`` is
    one sentence) but ``?"`` as a sentence end, so normalising to straight
    quotes splits every dialogue tag off its speech.  Measured on four probe
    sentences, curly gives 1/1/3/1 parts and straight gives 2/2/3/1.

``validate_sentences.py`` and ``spotcheck_sentences.py`` check the result:
no unit may contain a line break (under-split), and no unit may begin with a
lowercase dialogue tag (over-split).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import nltk

from common import word_span

_PARA_RE = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")
_QUOTE_MAP = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
_TERMINAL = r"[.!?\u2026]"
_CLOSE = r"[\u201d\"\u2019'\)\]]*"
# a unit is invalid if it still contains terminal punctuation followed by a
# newline, i.e. it silently holds more than one sentence
_INTERNAL_BREAK = re.compile(_TERMINAL + _CLOSE + r"\s*\n")
# a unit is invalid if it opens with a lowercase dialogue tag, i.e. the
# previous split cut a speech off from its tag
_ORPHAN_TAG = re.compile(
    r"^(said|asked|cried|answered|repeated|exclaimed|murmured|added|continued|"
    r"he|she|they|it)\b")


@dataclass
class Sentence:
    index: int
    text: str
    start: int
    end: int

    @property
    def n_words(self) -> int:
        return len(word_span(self.text))

    @property
    def preview(self) -> str:
        s = re.sub(r"\s+", " ", self.text).strip()
        return s[:100] + ("…" if len(s) > 100 else "")


def normalise_quotes(s: str) -> str:
    """1:1 character substitution, so offsets are preserved.

    Kept for diagnostics only.  It must **not** be used before punkt: see the
    module docstring.
    """
    return "".join(_QUOTE_MAP.get(c, c) for c in s)


def split_sentences(body: str) -> List[Sentence]:
    """Return sentences with offsets into *body*.

    Paragraph-first, then punkt on the original text (curly quotes intact).
    """
    out: List[Sentence] = []
    for pm in _PARA_RE.finditer(body):
        para = pm.group(0)
        base = pm.start()
        pos = 0
        for piece in nltk.sent_tokenize(para):
            idx = para.find(piece, pos)
            if idx < 0:
                idx = pos
            end = idx + len(piece)
            out.append(Sentence(index=len(out), text=para[idx:end],
                                start=base + idx, end=base + end))
            pos = end
    return out


def under_split(sents: List[Sentence]) -> List[int]:
    """Units that still hold more than one sentence."""
    return [s.index for s in sents if _INTERNAL_BREAK.search(s.text)]


def over_split(sents: List[Sentence]) -> List[int]:
    """Units that open with a lowercase dialogue tag split off its speech."""
    return [s.index for s in sents
            if _ORPHAN_TAG.match(s.text.lstrip("\u201c\"' ([")[:24])]


def invalid_units(sents: List[Sentence]) -> List[int]:
    """Any unit that fails either boundary check."""
    return sorted(set(under_split(sents)) | set(over_split(sents)))
