"""Shared coordinate and partition machinery for Pilot 002.

The only information used here is text position.  No literary labels or
target passages enter the partition definitions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "pilot_002"
P1_DIR = ROOT / "pilot_001"
DATA_DIR = PILOT_DIR / "data"
DERIVED_DIR = DATA_DIR / "derived"
MATRIX_DIR = DERIVED_DIR / "matrices"
RESULTS_DIR = PILOT_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"
DOCS_DIR = PILOT_DIR / "docs"
SOURCE_DIR = DATA_DIR / "source"

for directory in (SOURCE_DIR, DERIVED_DIR, MATRIX_DIR, TABLE_DIR, FIG_DIR,
                  DOCS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SOURCE = P1_DIR / "data" / "source" / "the_dead.txt"
DONORS = P1_DIR / "data" / "source" / "dubliners_other_stories.txt"
WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
PARA_RE = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")
TITLE_RE = re.compile(r"^\s*THE DEAD\s*\n", re.IGNORECASE)
SCALES = ("sentence", "paragraph", "w050", "w100", "w150", "w200", "w300")
WINDOWS = (50, 100, 150, 200, 300)
PHASE_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
REPRESENTATIONS = ("char_tfidf", "word_tfidf", "ncd", "minilm_white",
                   "bge_white")


@dataclass
class Unit:
    index: int
    scheme: str
    scale: str
    phase: int | None
    char_start: int
    char_end: int
    word_start: int
    word_end: int
    text: str

    @property
    def n_words(self) -> int:
        return self.word_end - self.word_start

    @property
    def center_word(self) -> float:
        return (self.word_start + self.word_end) / 2.0

    def record(self, include_text: bool = False) -> dict:
        out = asdict(self)
        if not include_text:
            out.pop("text")
        out["n_words"] = self.n_words
        out["center_word"] = self.center_word
        return out


def load_body() -> tuple[str, int]:
    raw = SOURCE.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    match = TITLE_RE.match(raw)
    return (raw[match.end():], match.end()) if match else (raw, 0)


def word_tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def _word_bounds(words: list[tuple[str, int, int]], start: int,
                 end: int) -> tuple[int, int]:
    """Map a character interval to its covering half-open word interval."""
    import bisect

    starts = [x[1] for x in words]
    ends = [x[2] for x in words]
    lo = bisect.bisect_right(ends, start)
    hi = bisect.bisect_left(starts, end)
    return lo, max(lo, hi)


def sentence_units(body: str, scheme: str = "sentence") -> list[Unit]:
    import nltk

    words = word_tokens(body)
    out = []
    # The verified Pilot 001 rule: paragraph-first, then Punkt on the original
    # curly-quoted text.  This avoids silently merging quoted paragraphs.
    for pm in PARA_RE.finditer(body):
        para, pos = pm.group(0), 0
        for piece in nltk.sent_tokenize(para):
            local = para.find(piece, pos)
            if local < 0:
                local = pos
            start, end = pm.start() + local, pm.start() + local + len(piece)
            lo, hi = _word_bounds(words, start, end)
            if hi > lo:
                out.append(Unit(len(out), scheme, "sentence", None, start,
                                end, lo, hi, body[start:end]))
            pos = local + len(piece)
    return out


def paragraph_units(body: str, scheme: str = "paragraph") -> list[Unit]:
    words = word_tokens(body)
    out = []
    for match in PARA_RE.finditer(body):
        lo, hi = _word_bounds(words, match.start(), match.end())
        if hi <= lo:
            continue
        out.append(Unit(len(out), scheme, "paragraph", None, match.start(),
                        match.end(), lo, hi, match.group(0)))
    return out


def window_units(body: str, window: int, phase_fraction: float) -> list[Unit]:
    words = word_tokens(body)
    phase = int(window * phase_fraction)
    scale = f"w{window:03d}"
    scheme = f"{scale}_p{int(phase_fraction * 100):02d}"
    out = []
    i = phase
    while i < len(words):
        j = min(i + window, len(words))
        if j - i < max(8, window // 4):
            break
        cs, ce = words[i][1], words[j - 1][2]
        out.append(Unit(len(out), scheme, scale, phase, cs, ce, i, j,
                        body[cs:ce]))
        i += window
    return out


def build_partitions(body: str) -> dict[str, list[Unit]]:
    out = {
        "sentence": sentence_units(body),
        "paragraph": paragraph_units(body),
    }
    for window in WINDOWS:
        for fraction in PHASE_FRACTIONS:
            units = window_units(body, window, fraction)
            out[units[0].scheme] = units
    return out


def source_fingerprint() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def preview(body: str, start_word: float, end_word: float,
            width: int = 130) -> str:
    words = word_tokens(body)
    if not words:
        return ""
    lo = max(0, min(len(words) - 1, int(start_word)))
    hi = max(lo + 1, min(len(words), int(end_word)))
    text = body[words[lo][1]:words[hi - 1][2]]
    text = re.sub(r"\s+", " ", text).strip()
    return text[:width] + ("…" if len(text) > width else "")


def validate_partitions(body: str, partitions: dict[str, list[Unit]]) -> list[dict]:
    n_words = len(word_tokens(body))
    rows = []
    for scheme, units in partitions.items():
        ok = True
        problems = []
        for expected, unit in enumerate(units):
            if unit.index != expected:
                ok = False
                problems.append("index")
            if not (0 <= unit.char_start < unit.char_end <= len(body)):
                ok = False
                problems.append("char_range")
            if not (0 <= unit.word_start < unit.word_end <= n_words):
                ok = False
                problems.append("word_range")
            if body[unit.char_start:unit.char_end] != unit.text:
                ok = False
                problems.append("char_roundtrip")
        rows.append({"scheme": scheme, "scale": units[0].scale,
                     "phase": units[0].phase, "n_units": len(units),
                     "ok": ok, "problems": sorted(set(problems))})
    return rows


def donor_story_texts() -> list[tuple[str, str]]:
    """Read the fourteen non-target stories without importing target labels."""
    names = ["THE SISTERS", "AN ENCOUNTER", "ARABY", "EVELINE",
             "AFTER THE RACE", "TWO GALLANTS", "THE BOARDING HOUSE",
             "A LITTLE CLOUD", "COUNTERPARTS", "CLAY", "A PAINFUL CASE",
             "IVY DAY IN THE COMMITTEE ROOM", "A MOTHER", "GRACE"]
    raw = DONORS.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paras = list(PARA_RE.finditer(raw))
    marks, cursor = [], 0
    for name in names:
        for idx in range(cursor, len(paras)):
            if re.sub(r"\s+", " ", paras[idx].group(0)).strip() == name:
                marks.append((idx, name))
                cursor = idx + 1
                break
    if len(marks) != len(names):
        raise RuntimeError(f"found {len(marks)} of {len(names)} donor titles")
    out = []
    for k, (idx, name) in enumerate(marks):
        stop = marks[k + 1][0] if k + 1 < len(marks) else len(paras)
        out.append((name, "\n\n".join(p.group(0) for p in paras[idx + 1:stop])))
    return out


def donor_units_by_scale(scale: str) -> list[Unit]:
    all_units = []
    for story_name, text in donor_story_texts():
        safe = re.sub(r"[^a-z0-9]+", "_", story_name.lower()).strip("_")
        if scale == "sentence":
            units = sentence_units(text, f"donor_{safe}_sentence")
        elif scale == "paragraph":
            units = paragraph_units(text, f"donor_{safe}_paragraph")
        else:
            window = int(scale[1:])
            # Pool all four phases for a phase-independent whitening fit.
            units = []
            for fraction in PHASE_FRACTIONS:
                us = window_units(text, window, fraction)
                for unit in us:
                    unit.scheme = f"donor_{safe}_{unit.scheme}"
                units.extend(us)
        all_units.extend(units)
    return all_units


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    import csv

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

