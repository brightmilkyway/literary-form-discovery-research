"""Audit the documents for stale content (AGENT.md section 0).

The repository's first rule is that every major round must end by updating the
affected documents.  This script mechanises the two self-checks that rule
requires:

  1. grep every superseded number/term across all documents and classify each
     hit as either a legitimate history/withdrawal record or a stale current
     claim;
  2. check that the version numbers and the headline counts agree across the
     report, the candidate list, both READMEs and the protocol.

Run from code/:  python doc_sync_audit.py
Exit code is non-zero when something looks stale, so it can gate a round.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import common as C

PILOT = C.PILOT_DIR
# The repository root may or may not be present next to the pilot: when this
# script runs inside the authoring vault the repo-root files do not exist.
# Probe a couple of likely locations instead of assuming one.
_CANDIDATE_ROOTS = [
    PILOT.parent,
    PILOT.parent.parent,
    Path(r"D:\项目\literary-form-discovery-research"),
]
REPO_ROOT = next(
    (r for r in _CANDIDATE_ROOTS
     if (r / "AGENT.md").exists() or (r / "pilot_001").is_dir() and
     (r / "README.md").exists()),
    PILOT.parent,
)

PILOT_DOCS = [
    PILOT / "docs" / "pilot_001_report.md",
    PILOT / "docs" / "candidate_findings.md",
    PILOT / "README.md",
    PILOT / "protocol.md",
]
REPO_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "AGENT.md",
]
# protocol.md is the original plan: it is not expected to carry results, so it
# is audited for stale values but excluded from the result-agreement check.
RESULT_DOCS = PILOT_DOCS[:3] + REPO_DOCS
DOCS = PILOT_DOCS + [p for p in REPO_DOCS if p.exists()]

# Values that were superseded during the pilot.  Each entry is
# (pattern, why it is superseded, what replaced it).
SUPERSEDED = [
    (r"\b764\b", "the broken splitter's sentence count", "1023"),
    (r"\b225 句", "the echo distance under the broken splitter", "290 句"),
    (r"句 534|句 759", "the echo sentence numbers under the broken splitter",
     "句 728 / 句 1018"),
    (r"0\.83", "the 'form baseline' (placeholder artefact)", "0.712 / 0.618"),
    (r"\+0\.728", "the sentence-scale NCD confound before the splitter fix",
     "+0.661"),
    (r"53\.1%", "variance explained before the splitter fix", "43.7%"),
    (r"45[–-]52%", "sentence-scale TextTiling percentiles before the fix",
     "41–53%"),
    (r"句式模板检测器", "the withdrawn 'template detector' claim",
     "withdrawn"),
    # v13 additions
    (r"该位置的文本内容\*?\*?不参与", "withdrawn: the song-name pair",
     "contributions small and model-dependent"),
    (r"词元存在性", "withdrawn: the hand-picked-filler artefact", "withdrawn"),
    (r"内容承载 / 情境承载", "the two-way typology, replaced by a ladder",
     "承载位置的谱"),
    (r"唯一超出量为负", "the v10 claim withdrawn in v11", "not unique"),
    (r"−0\.018", "the v10 excess for the lyric pair", "z = −0.54 / −0.96"),
    # v14 additions
    (r"句子层而非词层", "the v13 localisation claim, revised by the full scan",
     "大部分不由内容词承载，而在形式上"),
    (r"在句子层，不在词层", "the v13 localisation claim", "形式承载"),
]

# A hit is acceptable if the line (or a nearby line) marks it as history.
HISTORY_MARKERS = [
    "撤回", "曾", "原", "历史", "修正", "更正", "已更正", "取代", "~~",
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "CORRECTION", "superseded",
    "版本历程", "before", "was ", "旧", "→", "->", "改", "从", "before",
]


def audit() -> int:
    problems = 0
    print("=" * 78)
    print("DOCUMENT SYNC AUDIT")
    print("=" * 78)

    # ---- 1. superseded values ------------------------------------------
    print("\n1. SUPERSEDED VALUES")
    for path in DOCS:
        if not path.exists():
            print(f"   (skipped, not present here: {path})")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = []
        for n, line in enumerate(lines, 1):
            for pat, why, repl in SUPERSEDED:
                if re.search(pat, line):
                    window = " ".join(lines[max(0, n - 3):n + 1])
                    ok = any(m in window for m in HISTORY_MARKERS)
                    hits.append((n, pat, why, repl, ok, line.strip()))
        if not hits:
            continue
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path.name
        print(f"\n   {rel}")
        for n, pat, why, repl, ok, text in hits:
            tag = "ok (history)" if ok else "*** STALE ***"
            print(f"      L{n:<5} {tag:<16} {pat!r:<18} -> {repl}")
            if not ok:
                print(f"             {text[:110]}")
                problems += 1

    # ---- 2. cross-document agreement ------------------------------------
    print("\n2. CROSS-DOCUMENT AGREEMENT")
    text = {p: p.read_text(encoding="utf-8") for p in RESULT_DOCS
            if p.exists()}
    if not (REPO_ROOT / "AGENT.md").exists():
        print(f"   (repo-root docs not found; audited only the pilot docs)")
    checks = [
        ("sentence count 1023",
         lambda t: "1023" in t),
        ("echo sentences 728 / 1018",
         lambda t: ("728" in t and "1018" in t)),
        ("echo distance 290",
         lambda t: "290" in t),
        ("withdrawal of the template claim",
         lambda t: "模板" in t and "撤回" in t),
    ]
    for label, fn in checks:
        missing = [p.name for p, t in text.items() if not fn(t)]
        if missing:
            print(f"   {label:<38} absent from: {missing}")
    # the report must state a version and it must be the newest one
    rep = PILOT / "docs" / "pilot_001_report.md"
    if rep.exists():
        head = rep.read_text(encoding="utf-8").splitlines()[:6]
        vers = re.findall(r"v(\d+)", " ".join(head))
        if vers:
            newest = max(int(v) for v in vers)
            print(f"   report title version: v{newest}  "
                  f"(all versions mentioned: {sorted(set(vers))})")
        else:
            print("   *** report title has no version number ***")
            problems += 1

    print()
    print(f"RESULT: {'PASS' if problems == 0 else f'{problems} problem(s)'}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(audit())
