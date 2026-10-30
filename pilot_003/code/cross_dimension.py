"""Align the three response dimensions on transitions and compare trajectories."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

import common as C


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    predictive = {row["checkpoint_id"]: row for row in read_csv(
        C.TABLE_DIR / "predictive_summary.csv")}
    representations = read_csv(C.TABLE_DIR / "representation_trajectory.csv")
    interpretations = read_csv(C.TABLE_DIR / "interpretation_trajectory.csv")
    interpretation_by_cp = defaultdict(list)
    for row in interpretations:
        interpretation_by_cp[row["to_checkpoint"]].append(
            float(row["adjacent_cosine_distance"]))
    rep_by_cp = defaultdict(dict)
    for row in representations:
        if row["scale"] == "p4":
            rep_by_cp[row["checkpoint_id"]][row["method"]] = float(
                row["adjacent_cosine_distance"])
    aligned = []
    for checkpoint_id in sorted(interpretation_by_cp):
        pred = predictive[checkpoint_id]
        aligned.append({
            "checkpoint_id": checkpoint_id,
            "predictive_abs_mean_change": abs(float(pred["mean_change"])),
            "representation_char_p4": rep_by_cp[checkpoint_id]["char_hash"],
            "representation_word_p4": rep_by_cp[checkpoint_id]["word_hash"],
            "interpretation_mean_adjacent_distance": float(np.mean(
                interpretation_by_cp[checkpoint_id])),
        })
    C.write_csv(C.TABLE_DIR / "cross_dimension_aligned.csv", aligned)
    metrics = [key for key in aligned[0] if key != "checkpoint_id"]
    correlations = []
    for i, left in enumerate(metrics):
        for right in metrics[i + 1:]:
            x = [row[left] for row in aligned]
            y = [row[right] for row in aligned]
            correlations.append({
                "left": left,
                "right": right,
                "n": len(aligned),
                "pearson_r": float(pearsonr(x, y).statistic),
                "pearson_p_uncorrected": float(pearsonr(x, y).pvalue),
                "spearman_rho": float(spearmanr(x, y).statistic),
                "spearman_p_uncorrected": float(spearmanr(x, y).pvalue),
            })
    C.write_csv(C.TABLE_DIR / "cross_dimension_correlations.csv", correlations)
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
