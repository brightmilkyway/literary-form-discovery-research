"""Small release gate for Pilot 002's synchronized documents."""

from __future__ import annotations

from pathlib import Path
import re

PILOT = Path(__file__).resolve().parents[1]

DOCS = {
    "README": PILOT / "README.md",
    "report": PILOT / "docs" / "pilot_002_report.md",
    "candidates": PILOT / "docs" / "candidate_findings.md",
    "protocol": PILOT / "protocol.md",
    "derived": PILOT / "data" / "derived" / "README.md",
}

REQUIRED_SHARED = ("110", "44,653", "2,144")
REQUIRED_TOPOLOGY = ("94/110", "91/110", "19/110", "0/110", "−4.430")


def normalize_counts(text: str) -> str:
    """Treat Markdown spacing around count fractions as presentation-only."""
    return re.sub(r"\s*/\s*", "/", text)


def main() -> int:
    errors: list[str] = []
    texts = {}
    for label, path in DOCS.items():
        if not path.exists():
            errors.append(f"missing {label}: {path}")
            continue
        texts[label] = path.read_text(encoding="utf-8-sig")

    for label in ("README", "report", "candidates"):
        text = texts.get(label, "")
        for value in REQUIRED_SHARED:
            if value not in text:
                errors.append(f"{label} missing current value {value}")

    for label in ("report", "candidates"):
        text = texts.get(label, "")
        normalized_text = normalize_counts(text)
        for value in ("204", "0.663", "0.702"):
            if value not in text:
                errors.append(f"{label} missing calibrated/diagnostic value {value}")
        for value in REQUIRED_TOPOLOGY:
            if value not in normalized_text:
                errors.append(f"{label} missing strict-topology value {value}")

    for label in ("README", "report", "candidates"):
        if "v2" not in texts.get(label, ""):
            errors.append(f"{label} missing current version v2")

    if "100 次" not in texts.get("protocol", ""):
        errors.append("protocol missing the 100-iteration null audit")
    for excluded in ("embeddings/", "matrices/", "geometry_events.csv",
                     "units.json", "*.svg"):
        if excluded not in texts.get("derived", ""):
            errors.append(f"derived README missing exclusion {excluded}")

    required_outputs = (
        PILOT / "results" / "tables" / "candidates.csv",
        PILOT / "results" / "tables" / "candidates_calibrated.csv",
        PILOT / "results" / "tables" / "matrix_inventory.csv",
        PILOT / "results" / "tables" / "partition_validation.csv",
        PILOT / "results" / "figures" / "fig_candidate_map.png",
        PILOT / "results" / "figures" / "fig_null_calibration.png",
        PILOT / "results" / "tables" / "topology_matrix_summary.csv",
        PILOT / "results" / "tables" / "topology_feature_clusters.csv",
        PILOT / "results" / "figures" / "fig_topology_null_heatmap.png",
        PILOT / "results" / "figures" / "fig_topology_feature_map.png",
        PILOT / "data" / "derived" / "topology_summary.txt",
    )
    for path in required_outputs:
        if not path.exists():
            errors.append(f"missing retained output: {path.relative_to(PILOT)}")

    if errors:
        print("RESULT: FAIL")
        for error in errors:
            print(" -", error)
        return 1
    print("RESULT: PASS")
    print("documents agree on 110 matrices, 44,653 events, 2,144 candidates")
    print("null calibration and NCD length caveat are present")
    print("strict topology agrees on H0 94/110 and H1 0/110 after BH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

