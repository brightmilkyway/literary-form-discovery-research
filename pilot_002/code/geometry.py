"""Geometry extraction and cross-run persistence for Pilot 002."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

import common as C


def valid_remote_mask(units: list[C.Unit], total_words: int) -> np.ndarray:
    threshold = max(300, int(0.03 * total_words))
    n = len(units)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            gap = units[j].word_start - units[i].word_end
            if gap >= threshold:
                mask[i, j] = True
    return mask


def percentile_matrix(S: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    values = S[valid]
    P = np.full(S.shape, np.nan, dtype=float)
    if len(values):
        ranks = rankdata(values, method="average")
        pct = (ranks - 0.5) / len(ranks)
        P[valid] = pct
        P.T[valid] = pct
    return P


def _base_event(kind: str, scheme: str, representation: str,
                units: list[C.Unit], i0: int, i1: int, j0: int, j1: int,
                values: np.ndarray) -> dict:
    a0, a1 = units[i0].word_start, units[i1].word_end
    b0, b1 = units[j0].word_start, units[j1].word_end
    return {
        "geometry": kind, "scheme": scheme,
        "scale": units[0].scale, "phase": units[0].phase,
        "representation": representation,
        "a_start": a0, "a_end": a1, "a_center": (a0 + a1) / 2,
        "b_start": b0, "b_end": b1, "b_center": (b0 + b1) / 2,
        "matrix_i0": i0, "matrix_i1": i1,
        "matrix_j0": j0, "matrix_j1": j1,
        "cells": int(len(values)), "best_percentile": float(np.max(values)),
        "median_percentile": float(np.median(values)),
    }


def point_events(P: np.ndarray, valid: np.ndarray, units: list[C.Unit],
                 scheme: str, representation: str) -> list[dict]:
    rows, cols = np.where(valid & (P >= 0.99))
    return [_base_event("point", scheme, representation, units,
                        int(i), int(i), int(j), int(j),
                        np.array([P[i, j]]))
            for i, j in zip(rows, cols)]


def stripe_events(P: np.ndarray, valid: np.ndarray, units: list[C.Unit],
                  scheme: str, representation: str) -> list[dict]:
    high = valid & (P >= 0.975)
    n = len(units)
    events = []
    for offset in range(1, n):
        flags = np.array([high[i, i + offset]
                          for i in range(n - offset)], dtype=bool)
        start = None
        for k in range(len(flags) + 1):
            active = k < len(flags) and flags[k]
            if active and start is None:
                start = k
            if not active and start is not None:
                if k - start >= 3:
                    vals = np.array([P[i, i + offset]
                                     for i in range(start, k)])
                    events.append(_base_event(
                        "stripe", scheme, representation, units,
                        start, k - 1, start + offset, k - 1 + offset, vals))
                start = None
    return events


def block_events(P: np.ndarray, valid: np.ndarray, units: list[C.Unit],
                 scheme: str, representation: str) -> list[dict]:
    from scipy.ndimage import label

    high = valid & (P >= 0.975)
    labels, count = label(high, structure=np.ones((3, 3), dtype=int))
    events = []
    for component in range(1, count + 1):
        ii, jj = np.where(labels == component)
        if len(ii) < 6:
            continue
        i0, i1, j0, j1 = int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())
        if i1 - i0 < 1 or j1 - j0 < 1:
            continue
        events.append(_base_event("block", scheme, representation, units,
                                  i0, i1, j0, j1, P[ii, jj]))
    return events


def valley_events(S: np.ndarray, units: list[C.Unit], scheme: str,
                  representation: str, width: int = 3) -> list[dict]:
    candidates = []
    for boundary in range(width, len(units) - width + 1):
        cross = S[boundary - width:boundary, boundary:boundary + width]
        candidates.append((boundary, float(cross.mean())))
    if not candidates:
        return []
    scores = np.array([x[1] for x in candidates])
    cutoff = float(np.quantile(scores, 0.05))
    events = []
    for boundary, score in candidates:
        if score > cutoff:
            continue
        strength = float((scores >= score).mean())
        left, right = units[boundary - width], units[boundary + width - 1]
        center = (units[boundary - 1].word_end + units[boundary].word_start) / 2
        events.append({
            "geometry": "valley", "scheme": scheme,
            "scale": units[0].scale, "phase": units[0].phase,
            "representation": representation,
            "a_start": left.word_start, "a_end": right.word_end,
            "a_center": center, "b_start": center, "b_end": center,
            "b_center": center, "matrix_i0": boundary - width,
            "matrix_i1": boundary - 1, "matrix_j0": boundary,
            "matrix_j1": boundary + width - 1, "cells": width * width,
            "best_percentile": strength, "median_percentile": strength,
            "raw_cross_similarity": score,
        })
    return events


def extract_all(S: np.ndarray, units: list[C.Unit], scheme: str,
                representation: str, total_words: int):
    valid = valid_remote_mask(units, total_words)
    P = percentile_matrix(S, valid)
    events = []
    events.extend(point_events(P, valid, units, scheme, representation))
    events.extend(stripe_events(P, valid, units, scheme, representation))
    events.extend(block_events(P, valid, units, scheme, representation))
    events.extend(valley_events(S, units, scheme, representation))
    return events, P, {"eligible_remote_pairs": int(valid.sum()),
                       "minimum_remote_word_gap": max(300, int(0.03 * total_words))}


def aggregate(events: list[dict], total_words: int, body: str) -> list[dict]:
    """Cluster mapped events without inspecting their text."""
    candidates = []
    tolerance = 0.015 * total_words
    for family, subset in (
        ("relation", [e for e in events if e["geometry"] != "valley"]),
        ("valley", [e for e in events if e["geometry"] == "valley"]),
    ):
        if not subset:
            continue
        # Complete-link-compatible fixed bins: every two members assigned to
        # one bin differ by at most the frozen tolerance on every axis.  A
        # density/single-link clustering is intentionally not used: chains of
        # individually close events can otherwise percolate across the whole
        # self-similarity triangle and collapse unrelated regions together.
        buckets: dict[tuple[int, ...], list[dict]] = {}
        if family == "relation":
            for event in subset:
                key = (round(event["a_center"] / tolerance),
                       round(event["b_center"] / tolerance))
                buckets.setdefault(key, []).append(event)
        else:
            for event in subset:
                key = (round(event["a_center"] / tolerance),)
                buckets.setdefault(key, []).append(event)
        for key in sorted(buckets):
            rows = buckets[key]
            candidate = _summarise_cluster(family, rows, body)
            candidate["cluster_bin"] = ":".join(str(x) for x in key)
            candidates.append(candidate)

    candidates.sort(key=lambda x: (-x["persistence"],
                                   -x["support_matrices"],
                                   -x["support_events"]))
    for rank, candidate in enumerate(candidates, 1):
        candidate["candidate_id"] = f"C{rank:04d}"
    return candidates


def _summarise_cluster(family: str, rows: list[dict], body: str) -> dict:
    scales = sorted({r["scale"] for r in rows})
    reps = sorted({r["representation"] for r in rows})
    matrices = sorted({f'{r["scheme"]}|{r["representation"]}' for r in rows})
    window_rows = [r for r in rows if r["scale"].startswith("w")]
    if window_rows:
        supported_scales = {r["scale"] for r in window_rows}
        supported_scale_phases = {(r["scale"], r["phase"])
                                  for r in window_rows}
        phase_coverage = len(supported_scale_phases) / (4 * len(supported_scales))
    else:
        phase_coverage = 1.0
    scale_coverage = len(scales) / len(C.SCALES)
    representation_coverage = len(reps) / len(C.REPRESENTATIONS)
    persistence = (scale_coverage * representation_coverage *
                   phase_coverage) ** (1 / 3)
    geom = Counter(r["geometry"] for r in rows)
    a0 = float(np.median([r["a_start"] for r in rows]))
    a1 = float(np.median([r["a_end"] for r in rows]))
    b0 = float(np.median([r["b_start"] for r in rows]))
    b1 = float(np.median([r["b_end"] for r in rows]))
    out = {
        "candidate_id": "",
        "family": family,
        "geometry": ";".join(f"{k}:{v}" for k, v in sorted(geom.items())),
        "a_start": round(a0, 1), "a_end": round(a1, 1),
        "a_center": round(float(np.median([r["a_center"] for r in rows])), 1),
        "b_start": round(b0, 1), "b_end": round(b1, 1),
        "b_center": round(float(np.median([r["b_center"] for r in rows])), 1),
        "scale_coverage": round(scale_coverage, 4),
        "representation_coverage": round(representation_coverage, 4),
        "phase_coverage": round(phase_coverage, 4),
        "persistence": round(persistence, 4),
        "support_events": len(rows), "support_matrices": len(matrices),
        "scales": ";".join(scales), "representations": ";".join(reps),
        "schemes": ";".join(sorted({r["scheme"] for r in rows})),
        "best_percentile": round(max(r["best_percentile"] for r in rows), 6),
        "median_percentile": round(float(np.median(
            [r["median_percentile"] for r in rows])), 6),
        "a_preview": C.preview(body, a0, a1),
        "b_preview": "" if family == "valley" else C.preview(body, b0, b1),
    }
    return out

