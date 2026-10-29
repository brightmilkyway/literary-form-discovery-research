"""Position-randomisation audit for the frozen Pilot 002 aggregation."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np

import common as C
import geometry as G


def read_events():
    path = C.TABLE_DIR / "geometry_events.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    numeric = ("a_start", "a_end", "a_center", "b_start", "b_end",
               "b_center", "best_percentile", "median_percentile")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        row["phase"] = None if row["phase"] in ("", "None") else int(row["phase"])
    return rows


def eligible_centers(units, total_words):
    mask = G.valid_remote_mask(units, total_words)
    i, j = np.where(mask)
    return np.column_stack(([(units[k].word_start + units[k].word_end) / 2 for k in i],
                            [(units[k].word_start + units[k].word_end) / 2 for k in j]))


def boundary_centers(units):
    if len(units) < 7:
        return np.array([], dtype=float)
    return np.array([(units[k - 1].word_end + units[k].word_start) / 2
                     for k in range(3, len(units) - 2)], dtype=float)


def persistence_for(rows):
    scales = {r["scale"] for r in rows}
    reps = {r["representation"] for r in rows}
    window_rows = [r for r in rows if r["scale"].startswith("w")]
    if window_rows:
        ws = {r["scale"] for r in window_rows}
        sp = {(r["scale"], r["phase"]) for r in window_rows}
        phase = len(sp) / (4 * len(ws))
    else:
        phase = 1.0
    scale = len(scales) / len(C.SCALES)
    rep = len(reps) / len(C.REPRESENTATIONS)
    return (scale * rep * phase) ** (1 / 3), len({
        f'{r["scheme"]}|{r["representation"]}' for r in rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    body, _ = C.load_body()
    total_words = len(C.word_tokens(body))
    tolerance = .015 * total_words
    partitions = C.build_partitions(body)
    events = read_events()

    relation_choices = {s: eligible_centers(u, total_words)
                        for s, u in partitions.items()}
    valley_choices = {s: boundary_centers(u) for s, u in partitions.items()}

    observed = G.aggregate(events, total_words, body)
    observed_by_key = {(r["family"], r["cluster_bin"]): r for r in observed}
    null_p = {key: [] for key in observed_by_key}
    null_support = {key: [] for key in observed_by_key}

    for iteration in range(args.iterations):
        buckets = defaultdict(list)
        for event in events:
            if event["geometry"] == "valley":
                choices = valley_choices[event["scheme"]]
                center = float(choices[rng.integers(len(choices))])
                key_tuple = (round(center / tolerance),)
                family = "valley"
            else:
                choices = relation_choices[event["scheme"]]
                a, b = choices[rng.integers(len(choices))]
                key_tuple = (round(float(a) / tolerance),
                             round(float(b) / tolerance))
                family = "relation"
            key = (family, ":".join(str(x) for x in key_tuple))
            buckets[key].append(event)
        for key in observed_by_key:
            if key in buckets:
                p, support = persistence_for(buckets[key])
            else:
                p, support = 0.0, 0
            null_p[key].append(p)
            null_support[key].append(support)
        if (iteration + 1) % 10 == 0:
            print(f"null iteration {iteration + 1}/{args.iterations}", flush=True)

    calibrated = []
    for key, row in observed_by_key.items():
        ps = np.asarray(null_p[key])
        supports = np.asarray(null_support[key], dtype=float)
        obs_p = float(row["persistence"])
        obs_support = float(row["support_matrices"])
        empirical = (1 + int((ps >= obs_p - 1e-12).sum())) / (len(ps) + 1)
        sd = supports.std(ddof=1)
        support_z = ((obs_support - supports.mean()) / sd
                     if sd > 0 else float("inf"))
        out = dict(row)
        out.update({
            "null_iterations": args.iterations,
            "null_mean_persistence": round(float(ps.mean()), 6),
            "null_p95_persistence": round(float(np.quantile(ps, .95)), 6),
            "persistence_empirical_p": round(empirical, 6),
            "null_mean_support_matrices": round(float(supports.mean()), 4),
            "support_matrices_z": round(float(support_z), 4),
        })
        calibrated.append(out)

    # Preserve the frozen primary order; create a second diagnostic view that
    # orders by the null audit without replacing candidates.csv.
    primary_ids = {r["candidate_id"]: i for i, r in enumerate(observed)}
    calibrated.sort(key=lambda r: primary_ids[r["candidate_id"]])
    C.write_csv(C.TABLE_DIR / "candidates_calibrated.csv", calibrated)
    diagnostic = sorted(calibrated, key=lambda r: (
        float(r["persistence_empirical_p"]), -float(r["support_matrices_z"]),
        -float(r["persistence"])))
    C.write_csv(C.TABLE_DIR / "candidates_by_null_calibration.csv", diagnostic)
    print("wrote null calibration for", len(calibrated), "candidates")


if __name__ == "__main__":
    main()

