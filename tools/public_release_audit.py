"""Public-release gate: provenance, privacy patterns, and documentation integrity."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NORMALIZED_SOURCE = "13284c0070a518ed4f4c7d5cc857bc2bcdc11f229c3e042cc22e593c570a9f2d"
EXPECTED_DONORS = "6aabd7a6fd6bf8fd5bdb5b7d9887dcddb61577f77c7773f175fc7d44cb264f59"


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def main() -> None:
    problems: list[str] = []
    required = [
        "README.md", "README.zh-CN.md", "LICENSE", "LICENSE-CONTENT.md",
        "DATA_SOURCES.md", "CITATION.cff", "SECURITY.md", "pilot_003/report.md",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            problems.append(f"missing required public file: {relative}")
    if (ROOT / "pilot_003/docs/pilot_003_report.md").exists():
        problems.append("duplicate Pilot 003 report exists")

    for relative in ("pilot_001/data/source/the_dead.txt",
                     "pilot_003/data/source/the_dead.txt"):
        if normalized_text_sha256(ROOT / relative) != EXPECTED_NORMALIZED_SOURCE:
            problems.append(f"normalized source hash mismatch: {relative}")
    if normalized_text_sha256(ROOT / "pilot_001/data/source/dubliners_other_stories.txt") != EXPECTED_DONORS:
        problems.append("donor source hash mismatch")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in ("Research vision", "Experimental pilots", "Methodological principles",
                    "Reproducibility", "Findings and limitations", "Citation"):
        if f"## {heading}" not in readme:
            problems.append(f"README missing heading: {heading}")

    credential_patterns = {
        "aws": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        "github": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{30,255})\b"),
        "openai_style": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,255}\b"),
        "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    }
    local_path = re.compile(rb"(?:[A-Za-z]:\\Users\\|[A-Za-z]:\\\xe9\xa1\xb9\xe7\x9b\xae\\)")
    for path in tracked_files():
        # A staged deletion is still reported by ``git ls-files`` until commit.
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\x00" in data[:8192]:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for name, pattern in credential_patterns.items():
            if pattern.search(data):
                problems.append(f"credential pattern {name}: {relative}")
        if local_path.search(data):
            problems.append(f"absolute local path: {relative}")

    response_count = 0
    for replicate in (1, 2):
        path = ROOT / f"pilot_003/data/derived/deepseek_responses_r{replicate:02d}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        response_count += len(rows)
        if len(rows) != 40 or len({row["checkpoint_id"] for row in rows}) != 40:
            problems.append(f"DeepSeek replicate {replicate} is incomplete")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        raise SystemExit(1)
    print("RESULT: PASS")
    print(f"required public files, source fingerprints, tracked text, and {response_count} responses verified")


if __name__ == "__main__":
    main()
