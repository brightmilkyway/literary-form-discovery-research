"""Aggregate and render the strict-topology outputs for Pilot 002."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
import topology as T


CLUSTERS_PATH = C.TABLE_DIR / "topology_feature_clusters.csv"
SUMMARY_TEXT_PATH = C.DERIVED_DIR / "topology_summary.txt"
FEATURE_MAP_PATH = C.FIG_DIR / "fig_topology_feature_map.png"


def cluster_features(features: pd.DataFrame, total_words: int,
                     candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    tolerance = 0.015 * total_words
    data = features.copy()
    data["a_bin"] = np.rint(data["a_center"] / tolerance).astype(int)
    data["b_bin"] = np.rint(data["b_center"] / tolerance).astype(int)
    data["matrix"] = data["scheme"] + "|" + data["representation"]
    grouped = data.groupby(["dimension", "a_bin", "b_bin"], as_index=False).agg(
        features=("matrix", "size"),
        matrices=("matrix", "nunique"),
        scales=("scale", "nunique"),
        representations=("representation", "nunique"),
        mean_lifetime=("lifetime_at_horizon", "mean"),
        max_lifetime=("lifetime_at_horizon", "max"),
        a_center=("a_center", "median"),
        b_center=("b_center", "median"),
        a_preview=("a_preview", "first"),
        b_preview=("b_preview", "first"),
    )
    grouped = grouped.sort_values(
        ["dimension", "matrices", "representations", "scales", "mean_lifetime"],
        ascending=[True, False, False, False, False]).reset_index(drop=True)
    counters = {0: 0, 1: 0}
    identifiers = []
    for dimension in grouped["dimension"]:
        counters[int(dimension)] += 1
        identifiers.append(f"H{int(dimension)}C{counters[int(dimension)]:04d}")
    grouped.insert(0, "cluster_id", identifiers)
    if candidates is not None:
        relations = candidates[candidates["family"] == "relation"].copy()
        matches = []
        for row in grouped.itertuples():
            distances = np.maximum(
                np.abs(relations["a_center"] - row.a_center),
                np.abs(relations["b_center"] - row.b_center))
            index = distances.idxmin()
            match = relations.loc[index]
            if float(distances.loc[index]) <= tolerance:
                matches.append((match["candidate_id"], match["geometry"],
                                match["persistence_empirical_p"],
                                float(distances.loc[index])))
            else:
                matches.append(("", "", np.nan, float(distances.loc[index])))
        grouped["v1_candidate_id"] = [row[0] for row in matches]
        grouped["v1_geometry"] = [row[1] for row in matches]
        grouped["v1_empirical_p"] = [row[2] for row in matches]
        grouped["v1_coordinate_error"] = [row[3] for row in matches]
    for column in ("mean_lifetime", "max_lifetime", "a_center", "b_center"):
        grouped[column] = grouped[column].round(6)
    return grouped


def plot_feature_map(clusters: pd.DataFrame, total_words: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    for dimension, axis in enumerate(axes):
        subset = clusters[clusters["dimension"] == dimension].head(80)
        sizes = 18 + 10 * np.sqrt(subset["matrices"])
        scatter = axis.scatter(subset["a_center"], subset["b_center"],
                               s=sizes, c=subset["matrices"], cmap="viridis",
                               alpha=0.75, edgecolors="white", linewidths=0.4)
        axis.plot([0, total_words], [0, total_words], color="black", lw=0.8,
                  alpha=0.5)
        axis.set_xlim(0, total_words)
        axis.set_ylim(0, total_words)
        axis.set_xlabel("earlier word coordinate")
        axis.set_ylabel("later word coordinate")
        axis.set_title(f"H{dimension} recurrent birth-cell clusters")
        fig.colorbar(scatter, ax=axis, shrink=0.82, label="supporting matrices")
    fig.suptitle("Pilot 002 strict topology: recurrent feature locations")
    fig.savefig(FEATURE_MAP_PATH, dpi=180)
    plt.close(fig)


def build_summary(matrix: pd.DataFrame, clusters: pd.DataFrame) -> str:
    h0_total_q = int((matrix["h0_total_q_bh"] <= 0.05).sum())
    h0_strong_q = int((matrix["h0_strong_q_bh"] <= 0.05).sum())
    h1_total_p = int((matrix["h1_total_p"] <= 0.05).sum())
    h1_total_q = int((matrix["h1_total_q_bh"] <= 0.05).sum())
    lines = [
        "PILOT 002 STRICT TOPOLOGY — v2",
        "=" * 72,
        f"matrices: {len(matrix)}; null permutations per matrix: 100",
        f"H0 total persistence: BH q<=0.05 in {h0_total_q}/{len(matrix)} matrices; "
        f"mean z={matrix['h0_total_z'].mean():.3f}",
        f"H0 strong-feature count: BH q<=0.05 in {h0_strong_q}/{len(matrix)} matrices; "
        f"mean z={matrix['h0_strong_z'].mean():.3f}",
        f"H0 maximum persistence: BH q<=0.05 in "
        f"{int((matrix['h0_max_q_bh'] <= .05).sum())}/{len(matrix)} matrices",
        f"H1 total persistence: uncorrected p<=0.05 in {h1_total_p}/{len(matrix)}; "
        f"BH q<=0.05 in {h1_total_q}/{len(matrix)}; mean z={matrix['h1_total_z'].mean():.3f}",
        "",
        "INTERPRETATION BOUNDARY",
        "The robust result is an H0 deficit relative to sequence permutation: the",
        "observed reading order contains fewer/less persistent isolated high-similarity",
        "islands, consistent with similarities being arranged into contiguous regions.",
        "No H1 excess survives correction across 110 matrices. H1 locations remain",
        "uncorrected candidates, not evidence of narrative cycles.",
        "",
        "TOP RECURRENT H0 BIRTH-CELL CLUSTERS",
    ]
    for row in clusters[clusters["dimension"] == 0].head(10).itertuples():
        lines.append(
            f"{row.cluster_id}: {row.a_center:.1f}<->{row.b_center:.1f}; "
            f"matrices={row.matrices}, scales={row.scales}, reps={row.representations}; "
            f"mean lifetime={row.mean_lifetime:.3f}; v1={row.v1_candidate_id or '-'}")
    lines.extend(["", "TOP RECURRENT H1 BIRTH-CELL CLUSTERS (UNCORRECTED)"])
    for row in clusters[clusters["dimension"] == 1].head(10).itertuples():
        lines.append(
            f"{row.cluster_id}: {row.a_center:.1f}<->{row.b_center:.1f}; "
            f"matrices={row.matrices}, scales={row.scales}, reps={row.representations}; "
            f"mean lifetime={row.mean_lifetime:.3f}; v1={row.v1_candidate_id or '-'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    matrix = pd.read_csv(T.SUMMARY_PATH, encoding="utf-8-sig")
    features = pd.read_csv(T.FEATURE_PATH, encoding="utf-8-sig")
    candidates = pd.read_csv(C.TABLE_DIR / "candidates_calibrated.csv",
                             encoding="utf-8-sig")
    unit_document = json.loads(T.UNITS_PATH.read_text(encoding="utf-8-sig"))
    total_words = int(unit_document["body_words"])
    clusters = cluster_features(features, total_words, candidates)
    clusters.to_csv(CLUSTERS_PATH, index=False, encoding="utf-8-sig")
    SUMMARY_TEXT_PATH.write_text(build_summary(matrix, clusters), encoding="utf-8")
    plot_feature_map(clusters, total_words)
    print(SUMMARY_TEXT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
