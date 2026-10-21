"""Shared utilities for Pilot 001 (Joyce, *The Dead*).

Design goals
------------
* Keep the original reading order, paragraph boundaries and a traceable
  character offset for every analysis unit.
* Allow several chunking schemes (paragraph groups, sliding windows) so that
  methods can be compared on the *same* units and so that chunk-size
  sensitivity can be measured.
* Never bake literary labels (theme, symbol, turning point) into the pipeline.

Nothing here names a literary category.  It only produces positional units.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PILOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PILOT_DIR / "data"
RESULTS_DIR = PILOT_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

# The source text is looked up in several places so the pilot runs both in the
# authoring vault (where it sits next to the pilot folder) and in this
# repository (where it is stored under data/source/).
_SOURCE_CANDIDATES = (
    DATA_DIR / "source" / "the_dead.txt",
    PILOT_DIR.parent / "the_dead.txt",
    PILOT_DIR / "the_dead.txt",
)
SOURCE_TXT = next((p for p in _SOURCE_CANDIDATES if p.exists()),
                  _SOURCE_CANDIDATES[0])

for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED = 42


def set_seed(seed: int = SEED) -> int:
    """Seed every RNG we might touch, so runs are reproducible.

    The pipeline is deterministic by construction (no sampling, no training),
    but the neural encoder and any future extension may not be, so we pin the
    seed anyway and record it in the result metadata.
    """
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch optional
        pass
    return seed


# --------------------------------------------------------------------------
# Text loading & normalisation
# --------------------------------------------------------------------------

_TITLE_RE = re.compile(r"^\s*THE DEAD\s*\n", re.IGNORECASE)


def load_raw() -> str:
    """Return the source text exactly as stored (UTF-8, no BOM)."""
    text = SOURCE_TXT.read_text(encoding="utf-8-sig")
    # Normalise newlines only; keep every other character intact.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_title(raw: str) -> tuple[str, int]:
    """Remove the leading 'THE DEAD' heading.

    Returns the body text and the character offset at which the body begins,
    so that offsets computed on the body can be mapped back to the file.
    """
    m = _TITLE_RE.match(raw)
    if not m:
        return raw, 0
    return raw[m.end():], m.end()


# --------------------------------------------------------------------------
# Paragraphs
# --------------------------------------------------------------------------

@dataclass
class Paragraph:
    index: int          # 0-based, reading order
    text: str
    start: int          # char offset in the *body* string
    end: int            # char offset in the *body* string (exclusive)
    n_words: int

    @property
    def preview(self) -> str:
        s = re.sub(r"\s+", " ", self.text).strip()
        return s[:90] + ("…" if len(s) > 90 else "")


def paragraphs(body: str) -> List[Paragraph]:
    """Split on blank lines, preserving order and offsets."""
    paras: List[Paragraph] = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", body):
        chunk = m.group(0)
        if not chunk.strip():
            continue
        paras.append(
            Paragraph(
                index=len(paras),
                text=chunk,
                start=m.start(),
                end=m.end(),
                n_words=len(word_span(chunk)),
            )
        )
    return paras


# --------------------------------------------------------------------------
# Tokenisation (for TextTiling, NCD, TF-IDF baselines)
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")


def word_span(text: str) -> List[tuple[str, int, int]]:
    """Return [(surface, start, end)] word tokens with *local* offsets."""
    return [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def normalise_token(surface: str) -> str:
    """Lower-case and fold curly apostrophes; used only for bag-of-words work."""
    s = surface.replace("\u2019", "'").lower()
    return unicodedata.normalize("NFKC", s)


def tokens(text: str) -> List[str]:
    return [normalise_token(w) for w, _, _ in word_span(text)]


# --------------------------------------------------------------------------
# Analysis units
# --------------------------------------------------------------------------

@dataclass
class Unit:
    index: int
    start: int          # char offset in body string
    end: int
    text: str
    n_words: int
    first_para: int
    last_para: int
    kind: str           # 'group' | 'window' | 'paragraph'

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def units_by_group(paras: Sequence[Paragraph], target_words: int) -> List[Unit]:
    """Merge consecutive paragraphs until >= target_words; never split a para."""
    units: List[Unit] = []
    buf: List[Paragraph] = []
    wcount = 0

    def flush() -> None:
        nonlocal buf, wcount
        if not buf:
            return
        units.append(
            Unit(
                index=len(units),
                start=buf[0].start,
                end=buf[-1].end,
                text="\n\n".join(p.text for p in buf),
                n_words=wcount,
                first_para=buf[0].index,
                last_para=buf[-1].index,
                kind="group",
            )
        )
        buf, wcount = [], 0

    for p in paras:
        buf.append(p)
        wcount += p.n_words
        if wcount >= target_words:
            flush()
    flush()
    return units


def units_by_window(paras: Sequence[Paragraph], window: int, stride: int) -> List[Unit]:
    """Sliding windows over the word stream (uses char offsets from the source)."""
    # Build a flat word list with global offsets.
    flat: List[tuple[str, int, int]] = []
    for p in paras:
        for surface, s, e in word_span(p.text):
            flat.append((surface, p.start + s, p.start + e))

    # Paragraph lookup for first/last paragraph attribution.
    starts = [p.start for p in paras]

    def para_at(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    units: List[Unit] = []
    i = 0
    n = len(flat)
    while i < n:
        j = min(i + window, n)
        if j - i < max(8, window // 4):  # drop tiny tail windows
            break
        start = flat[i][1]
        end = flat[j - 1][2]
        units.append(
            Unit(
                index=len(units),
                start=start,
                end=end,
                text="",  # filled by fill_text() using the offsets
                n_words=j - i,
                first_para=para_at(start),
                last_para=para_at(end - 1),
                kind="window",
            )
        )
        i += stride
    # Fill text from body lazily by caller using offsets.
    return units


def fill_text(units: List[Unit], body: str) -> None:
    for u in units:
        u.text = body[u.start:u.end]


def units_by_paragraph(paras: Sequence[Paragraph]) -> List[Unit]:
    return [
        Unit(
            index=p.index,
            start=p.start,
            end=p.end,
            text=p.text,
            n_words=p.n_words,
            first_para=p.index,
            last_para=p.index,
            kind="paragraph",
        )
        for p in paras
    ]


def build_units(body: str, scheme: str, **kw) -> List[Unit]:
    paras = paragraphs(body)
    if scheme == "group":
        return units_by_group(paras, kw.get("target_words", 150))
    if scheme == "window":
        us = units_by_window(paras, kw["window"], kw.get("stride", kw["window"] // 2))
        fill_text(us, body)
        return us
    if scheme == "paragraph":
        return units_by_paragraph(paras)
    raise ValueError(f"unknown scheme: {scheme}")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def cite(body: str, unit: Unit, width: int = 160) -> str:
    """A short, verifiable quote taken from the unit's start."""
    s = re.sub(r"\s+", " ", body[unit.start:unit.end]).strip()
    return s[:width] + ("…" if len(s) > width else "")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def word_positions(body: str, unit: Unit) -> int:
    """Running word position of a unit's start (for plotting on a word axis)."""
    return len(word_span(body[: unit.start]))
