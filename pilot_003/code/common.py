"""Source coordinates, checkpoints, and reproducible I/O for Pilot 003."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "pilot_003"
SOURCE = PILOT_DIR / "data" / "source" / "the_dead.txt"
CONFIG_PATH = PILOT_DIR / "config.json"
DERIVED_DIR = PILOT_DIR / "data" / "derived"
TABLE_DIR = PILOT_DIR / "results" / "tables"
FIGURE_DIR = PILOT_DIR / "results" / "figures"
RESULTS_DIR = PILOT_DIR / "results"

WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
TITLE_RE = re.compile(r"^\s*THE DEAD\s*\n", re.IGNORECASE)
PARAGRAPH_RE = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)

for directory in (DERIVED_DIR, TABLE_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paragraph:
    index: int
    char_start: int
    char_end: int
    word_start: int
    word_end: int
    text: str


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_body() -> tuple[str, int]:
    raw = SOURCE.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    match = TITLE_RE.match(raw)
    return (raw[match.end():], match.end()) if match else (raw, 0)


def source_sha256() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def paragraphs(body: str) -> list[Paragraph]:
    words = list(WORD_RE.finditer(body))
    out: list[Paragraph] = []
    for match in PARAGRAPH_RE.finditer(body):
        lo = next((i for i, word in enumerate(words) if word.end() > match.start()), len(words))
        hi = lo
        while hi < len(words) and words[hi].start() < match.end():
            hi += 1
        if hi > lo:
            out.append(Paragraph(len(out), match.start(), match.end(), lo, hi,
                                 body[match.start():match.end()]))
    return out


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    if not 2 <= count <= size:
        raise ValueError("checkpoint count must be between 2 and paragraph count")
    indices = [int(i * (size - 1) / (count - 1) + 0.5) for i in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("rounded checkpoint indices are not unique")
    return indices


def build_checkpoints(body: str, paras: list[Paragraph], count: int) -> list[dict]:
    indices = evenly_spaced_indices(len(paras), count)
    total_words = len(WORD_RE.findall(body))
    records = []
    previous_index = -1
    for number, para_index in enumerate(indices, 1):
        para = paras[para_index]
        prefix = body[:para.char_end]
        records.append({
            "checkpoint_id": f"cp_{number:03d}",
            "checkpoint_number": number,
            "paragraph_index": para_index,
            "paragraphs_seen": para_index + 1,
            "paragraph_gap": para_index - previous_index,
            "char_end": para.char_end,
            "word_end": para.word_end,
            "word_fraction": para.word_end / total_words,
            "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        })
        previous_index = para_index
    return records


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
