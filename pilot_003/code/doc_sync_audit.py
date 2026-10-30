"""Gate Pilot 003 documentation and retained evidence before release."""

from __future__ import annotations

import json
from pathlib import Path

import common as C


def main() -> None:
    problems: list[str] = []
    report = C.PILOT_DIR / "report.md"
    duplicate = C.PILOT_DIR / "docs" / "pilot_003_report.md"
    if duplicate.exists():
        problems.append("duplicate report exists; pilot_003/report.md is canonical")

    root_readme = C.ROOT / "README.md"
    docs = {
        "root_README": root_readme.read_text(encoding="utf-8")
        if root_readme.exists() else "",
        "README": (C.PILOT_DIR / "README.md").read_text(encoding="utf-8"),
        "protocol": (C.PILOT_DIR / "protocol.md").read_text(encoding="utf-8"),
        "report": report.read_text(encoding="utf-8"),
        "candidates": (C.PILOT_DIR / "docs" / "candidate_findings.md").read_text(
            encoding="utf-8"),
    }
    stale = ["解释模型尚未执行", "模型默认为 `null`", "当前没有已配置的解释模型"]
    for name, text in docs.items():
        for phrase in stale:
            if phrase in text:
                problems.append(f"{name}: stale phrase: {phrase}")

    for number in ("22,574", "862,667", "0.2575"):
        if number not in docs["report"]:
            problems.append(f"report missing current value {number}")
    for number in ("0.2575", "0.939", "0.511"):
        if number not in docs["candidates"]:
            problems.append(f"candidate findings missing current value {number}")
    if root_readme.exists() and "| 003 | James Joyce" not in docs["root_README"]:
        problems.append("root README missing Pilot 003 index row")

    expected_source = "13284c0070a518ed4f4c7d5cc857bc2bcdc11f229c3e042cc22e593c570a9f2d"
    actual_source = C.source_sha256()
    if actual_source != expected_source:
        problems.append(f"source hash changed: {actual_source}")

    for replicate in (1, 2):
        path = C.DERIVED_DIR / f"deepseek_responses_r{replicate:02d}.jsonl"
        rows = C.read_jsonl(path) if path.exists() else []
        if len(rows) != 40 or len({row.get("checkpoint_id") for row in rows}) != 40:
            problems.append(f"replicate {replicate}: expected 40 unique responses")
        for row in rows:
            try:
                parsed = json.loads(row["output"])
                if set(parsed) != {"current_understanding", "salient_uncertainties",
                                   "evidence_basis"}:
                    raise ValueError("schema")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                problems.append(f"replicate {replicate}: invalid response schema")
                break

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        raise SystemExit(1)
    print("RESULT: PASS")
    print("single report verified; current values present; source and 80 responses verified")


if __name__ == "__main__":
    main()
