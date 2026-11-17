"""Strict persistent-homology extension for Pilot 002.

This module is deliberately separate from ``geometry.py``.  The v1
``persistence`` score means recurrence across observational settings; here
``persistence`` has its mathematical meaning in a cubical filtration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

import common as C


UNITS_PATH = C.DERIVED_DIR / "units.json"
SUMMARY_PATH = C.TABLE_DIR / "topology_matrix_summary.csv"
FEATURE_PATH = C.TABLE_DIR / "topology_features.csv"
AGGREGATE_PATH = C.TABLE_DIR / "topology_aggregate.csv"
METADATA_PATH = C.DERIVED_DIR / "topology_run_metadata.json"
HEATMAP_PATH = C.FIG_DIR / "fig_topology_null_heatmap.png"
DIAGRAM_PATH = C.FIG_DIR / "fig_topology_example_diagrams.png"

MAX_UNITS = 128
HORIZON = 0.25
STRONG_LIFETIME = 0.05
BARRIER = 1.25
NULL_ITERATIONS = 100
SEED = 20261029
TOP_FEATURES_PER_DIMENSION = 8
METRICS = (
    "h0_count", "h0_total", "h0_max", "h0_censored", "h0_strong",
    "h1_count", "h1_total", "h1_max", "h1_censored", "h1_strong",
)


def landmark_indices(n_units: int, maximum: int = MAX_UNITS) -> np.ndarray:
    """Choose order-preserving landmarks without reading matrix values."""
    if n_units <= maximum:
        return np.arange(n_units, dtype=int)
    raw = np.linspace(0, n_units - 1, maximum)
    indices = np.unique(np.rint(raw).astype(int))
    if len(indices) != maximum:
        raise AssertionError("landmark selection did not preserve requested size")
    return indices


def remote_mask(units: list[dict], total_words: int) -> np.ndarray:
    threshold = max(300, int(0.03 * total_words))
    n = len(units)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        end = int(units[i]["word_end"])
        for j in range(i + 1, n):
            if int(units[j]["word_start"]) - end >= threshold:
                mask[i, j] = True
    return mask


def filtration_from_similarity(similarity: np.ndarray,
                               valid: np.ndarray) -> np.ndarray:
    """Construct the upper-triangular superlevel cubical filtration."""
    if similarity.shape != valid.shape:
        raise ValueError("similarity and validity masks must have equal shape")
    values = np.asarray(similarity[valid], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("valid similarity values must be finite and non-empty")
    percentiles = (rankdata(values, method="average") - 0.5) / len(values)
    filtration = np.full(similarity.shape, BARRIER, dtype=np.float64)
    filtration[valid] = 1.0 - percentiles
    return filtration


def _dimension_summary(intervals: np.ndarray, dimension: int) -> dict[str, float]:
    if intervals.size == 0:
        intervals = np.empty((0, 2), dtype=float)
    births = intervals[:, 0]
    deaths = intervals[:, 1]
    keep = births < HORIZON
    births, deaths = births[keep], deaths[keep]
    clipped = np.minimum(deaths, HORIZON)
    lifetimes = np.maximum(0.0, clipped - births)
    lifetimes = lifetimes[lifetimes > 1e-12]
    prefix = f"h{dimension}"
    return {
        f"{prefix}_count": int(len(lifetimes)),
        f"{prefix}_total": float(lifetimes.sum()),
        f"{prefix}_max": float(lifetimes.max()) if len(lifetimes) else 0.0,
        f"{prefix}_censored": int(np.sum(deaths >= HORIZON)),
        f"{prefix}_strong": int(np.sum(lifetimes >= STRONG_LIFETIME)),
    }


def cubical_summary(filtration: np.ndarray,
                    return_complex: bool = False):
    complex_ = gd.CubicalComplex(top_dimensional_cells=filtration)
    complex_.persistence(homology_coeff_field=2, min_persistence=0.0)
    summary = {}
    for dimension in (0, 1):
        summary.update(_dimension_summary(
            complex_.persistence_intervals_in_dimension(dimension), dimension))
    return (summary, complex_) if return_complex else summary


def _feature_rows(complex_: gd.CubicalComplex, filtration: np.ndarray,
                  scheme: str, representation: str,
                  selected_units: list[dict], body: str) -> list[dict]:
    regular, essential = complex_.cofaces_of_persistence_pairs()
    flat = filtration.ravel(order="F")
    rows = []
    for dimension in (0, 1):
        candidates = []
        if dimension < len(regular):
            for birth_index, death_index in np.asarray(regular[dimension]).reshape(-1, 2):
                candidates.append((int(birth_index), float(flat[int(birth_index)]),
                                   float(flat[int(death_index)]), False))
        if dimension < len(essential):
            for birth_index in np.asarray(essential[dimension]).reshape(-1):
                candidates.append((int(birth_index), float(flat[int(birth_index)]),
                                   math.inf, True))
        retained = []
        for birth_index, birth, death, essential_flag in candidates:
            if birth >= HORIZON:
                continue
            lifetime = max(0.0, min(death, HORIZON) - birth)
            if lifetime < STRONG_LIFETIME:
                continue
            i, j = np.unravel_index(birth_index, filtration.shape, order="F")
            if i > j:
                i, j = j, i
            if not (0 <= i < len(selected_units) and 0 <= j < len(selected_units)):
                continue
            a, b = selected_units[i], selected_units[j]
            a_center = (float(a["word_start"]) + float(a["word_end"])) / 2
            b_center = (float(b["word_start"]) + float(b["word_end"])) / 2
            retained.append({
                "scheme": scheme,
                "scale": selected_units[0]["scale"],
                "representation": representation,
                "dimension": dimension,
                "birth": round(birth, 8),
                "death": "inf" if essential_flag else round(death, 8),
                "lifetime_at_horizon": round(lifetime, 8),
                "right_censored": bool(essential_flag or death >= HORIZON),
                "birth_grid_i": int(i),
                "birth_grid_j": int(j),
                "a_center": round(a_center, 1),
                "b_center": round(b_center, 1),
                "a_preview": C.preview(body, max(0, a_center - 25), a_center + 25),
                "b_preview": C.preview(body, max(0, b_center - 25), b_center + 25),
            })
        retained.sort(key=lambda row: (-row["lifetime_at_horizon"], row["birth"]))
        rows.extend(retained[:TOP_FEATURES_PER_DIMENSION])
    return rows


def _null_statistics(observed: dict[str, float], null_rows: list[dict[str, float]]) -> dict:
    out = {}
    for metric in METRICS:
        values = np.array([float(row[metric]) for row in null_rows], dtype=float)
        obs = float(observed[metric])
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[f"{metric}_null_mean"] = mean
        out[f"{metric}_null_sd"] = sd
        out[f"{metric}_z"] = (obs - mean) / sd if sd > 0 else 0.0
        p_upper = (1 + int(np.sum(values >= obs))) / (len(values) + 1)
        p_lower = (1 + int(np.sum(values <= obs))) / (len(values) + 1)
        out[f"{metric}_p_upper"] = p_upper
        out[f"{metric}_p_lower"] = p_lower
        out[f"{metric}_p"] = min(1.0, 2.0 * min(p_lower, p_upper))
    return out


def _bh_adjust(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scale"], row["representation"]), []).append(row)
    output = []
    for (scale, representation), members in sorted(groups.items()):
        record = {
            "scale": scale,
            "representation": representation,
            "matrices": len(members),
        }
        for dimension in (0, 1):
            for base in ("total", "max", "strong"):
                key = f"h{dimension}_{base}"
                values = np.array([float(row[key]) for row in members])
                z_values = np.array([float(row[f"{key}_z"]) for row in members])
                p_values = np.array([float(row[f"{key}_p"]) for row in members])
                record[f"{key}_mean"] = float(values.mean())
                record[f"{key}_z_mean"] = float(z_values.mean())
                record[f"{key}_p_le_005"] = int(np.sum(p_values <= 0.05))
        output.append(record)
    return output


def _plot_heatmap(rows: list[dict]) -> None:
    scales = list(C.SCALES)
    reps = list(C.REPRESENTATIONS)
    lookup = {(row["scale"], row["representation"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    for axis, metric, title in zip(
            axes, ("h0_total_z_mean", "h1_total_z_mean"),
            ("H0 total persistence vs order-permutation null",
             "H1 total persistence vs order-permutation null")):
        grid = np.full((len(scales), len(reps)), np.nan)
        for i, scale in enumerate(scales):
            for j, rep in enumerate(reps):
                row = lookup.get((scale, rep))
                if row:
                    grid[i, j] = float(row[metric])
        bound = max(1.0, float(np.nanmax(np.abs(grid))))
        image = axis.imshow(grid, cmap="RdBu_r", vmin=-bound, vmax=bound,
                            aspect="auto")
        axis.set_xticks(range(len(reps)), reps, rotation=35, ha="right")
        axis.set_yticks(range(len(scales)), scales)
        axis.set_title(title)
        for i in range(len(scales)):
            for j in range(len(reps)):
                if np.isfinite(grid[i, j]):
                    axis.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                              fontsize=8, color="black")
        fig.colorbar(image, ax=axis, shrink=0.82, label="mean z")
    fig.suptitle("Pilot 002 strict topology: observed minus sequence-permutation null")
    fig.savefig(HEATMAP_PATH, dpi=180)
    plt.close(fig)


def _plot_example(summary_rows: list[dict], matrices: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    winner = max(summary_rows,
                 key=lambda row: max(float(row["h0_total_z"]), float(row["h1_total_z"])))
    key = f'{winner["scheme"]}__{winner["representation"]}'
    similarity, valid = matrices[key]
    filtration = filtration_from_similarity(similarity, valid)
    _, complex_ = cubical_summary(filtration, return_complex=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for dimension, axis in enumerate(axes):
        intervals = complex_.persistence_intervals_in_dimension(dimension)
        if len(intervals):
            keep = intervals[:, 0] < HORIZON
            births = intervals[keep, 0]
            deaths = np.minimum(intervals[keep, 1], HORIZON)
            axis.scatter(births, deaths, s=9, alpha=0.45)
        axis.plot([0, HORIZON], [0, HORIZON], color="black", lw=1)
        axis.set_xlim(0, HORIZON)
        axis.set_ylim(0, HORIZON)
        axis.set_xlabel("birth filtration")
        axis.set_ylabel("death / right-censor horizon")
        axis.set_title(f"H{dimension}")
    fig.suptitle(f"Persistence diagrams: {key}")
    fig.savefig(DIAGRAM_PATH, dpi=180)
    plt.close(fig)


def run(null_iterations: int = NULL_ITERATIONS, seed: int = SEED,
        jobs: int = 1) -> dict:
    if not UNITS_PATH.exists():
        raise FileNotFoundError(
            f"{UNITS_PATH} is absent; run pilot_002/code/run_pipeline.py first")
    unit_document = json.loads(UNITS_PATH.read_text(encoding="utf-8-sig"))
    total_words = int(unit_document["body_words"])
    body, _ = C.load_body()
    rng = np.random.default_rng(seed)
    summary_rows, feature_rows = [], []
    retained_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    matrix_files = sorted(C.MATRIX_DIR.glob("*.npz"))
    if len(matrix_files) != 110:
        raise RuntimeError(f"expected 110 matrices, found {len(matrix_files)}")
    started = time.time()
    for number, matrix_path in enumerate(matrix_files, 1):
        scheme, representation = matrix_path.stem.split("__", 1)
        all_units = unit_document["schemes"][scheme]
        archive = np.load(matrix_path, allow_pickle=False)
        full_similarity = np.asarray(archive["similarity"], dtype=float)
        indices = landmark_indices(len(all_units))
        selected_units = [all_units[int(index)] for index in indices]
        similarity = full_similarity[np.ix_(indices, indices)]
        valid = remote_mask(selected_units, total_words)
        filtration = filtration_from_similarity(similarity, valid)
        observed, complex_ = cubical_summary(filtration, return_complex=True)
        permutations = [rng.permutation(len(indices))
                        for _ in range(null_iterations)]

        def evaluate_null(permutation: np.ndarray) -> dict[str, float]:
            shuffled = similarity[np.ix_(permutation, permutation)]
            return cubical_summary(filtration_from_similarity(shuffled, valid))

        if jobs == 1:
            null_rows = [evaluate_null(permutation) for permutation in permutations]
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                null_rows = list(executor.map(evaluate_null, permutations))
        row = {
            "scheme": scheme,
            "scale": selected_units[0]["scale"],
            "phase": "" if selected_units[0]["phase"] is None else selected_units[0]["phase"],
            "representation": representation,
            "source_units": len(all_units),
            "landmark_units": len(indices),
            "eligible_remote_pairs": int(valid.sum()),
            **observed,
            **_null_statistics(observed, null_rows),
        }
        summary_rows.append(row)
        feature_rows.extend(_feature_rows(
            complex_, filtration, scheme, representation, selected_units, body))
        retained_matrices[f"{scheme}__{representation}"] = (similarity, valid)
        print(f"[{number:03d}/{len(matrix_files)}] {scheme} / {representation} "
              f"H0 total z={row['h0_total_z']:+.2f} "
              f"H1 total z={row['h1_total_z']:+.2f}", flush=True)

    for metric in METRICS:
        adjusted = _bh_adjust([float(row[f"{metric}_p"]) for row in summary_rows])
        for row, value in zip(summary_rows, adjusted):
            row[f"{metric}_q_bh"] = value
    aggregate_rows = _aggregate(summary_rows)
    _write_csv(SUMMARY_PATH, summary_rows)
    _write_csv(FEATURE_PATH, feature_rows)
    _write_csv(AGGREGATE_PATH, aggregate_rows)
    _plot_heatmap(aggregate_rows)
    _plot_example(summary_rows, retained_matrices)
    elapsed = time.time() - started
    metadata = {
        "method": "2D cubical persistent homology of upper-triangular remote similarity landscapes",
        "coefficients": "F2",
        "dimensions": [0, 1],
        "max_units": MAX_UNITS,
        "horizon": HORIZON,
        "strong_lifetime": STRONG_LIFETIME,
        "barrier": BARRIER,
        "null": "simultaneous row/column unit-order permutation",
        "null_iterations": null_iterations,
        "parallel_jobs": jobs,
        "seed": seed,
        "matrices": len(summary_rows),
        "feature_rows": len(feature_rows),
        "elapsed_seconds": round(elapsed, 3),
        "source_sha256": hashlib.sha256(C.SOURCE.read_bytes()).hexdigest(),
        "gudhi_version": gd.__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-iterations", type=int, default=NULL_ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    metadata = run(args.null_iterations, args.seed, args.jobs)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
