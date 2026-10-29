"""Compare adjacent and repeated interpretive responses without literary labels."""

from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import common as C


def load_run(path):
    rows = C.read_jsonl(path)
    return [{**row, "parsed": json.loads(row["output"])} for row in rows]


def word_count(parsed: dict) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", " ".join(parsed.values())))


def main() -> None:
    paths = sorted(C.DERIVED_DIR.glob("interpretation_runs_r*.jsonl"))
    if not paths:
        raise RuntimeError("no completed interpretation runs found")
    runs = [load_run(path) for path in paths]
    checkpoint_ids = [row["checkpoint_id"] for row in runs[0]]
    if any([row["checkpoint_id"] for row in run] != checkpoint_ids for run in runs):
        raise RuntimeError("replicates do not share checkpoint order")
    vectorizer = HashingVectorizer(n_features=32768, alternate_sign=False,
                                   ngram_range=(1, 2), norm="l2")
    adjacent_rows = []
    for replicate, run in enumerate(runs, 1):
        texts = [" ".join(row["parsed"].values()) for row in run]
        vectors = vectorizer.transform(texts)
        for i in range(1, len(run)):
            adjacent_rows.append({
                "replicate": replicate,
                "from_checkpoint": checkpoint_ids[i - 1],
                "to_checkpoint": checkpoint_ids[i],
                "adjacent_cosine_distance": float(
                    1 - cosine_similarity(vectors[i - 1], vectors[i])[0, 0]),
                "output_words": word_count(run[i]["parsed"]),
            })
    C.write_csv(C.TABLE_DIR / "interpretation_trajectory.csv", adjacent_rows)
    replicate_rows = []
    if len(runs) >= 2:
        for i, checkpoint_id in enumerate(checkpoint_ids):
            texts = [" ".join(run[i]["parsed"].values()) for run in runs]
            matrix = cosine_similarity(vectorizer.transform(texts))
            distances = 1 - matrix[np.triu_indices(len(runs), k=1)]
            replicate_rows.append({
                "checkpoint_id": checkpoint_id,
                "replicates": len(runs),
                "mean_pairwise_cosine_distance": float(distances.mean()),
                "max_pairwise_cosine_distance": float(distances.max()),
            })
        C.write_csv(C.TABLE_DIR / "interpretation_repeatability.csv", replicate_rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(2, len(checkpoint_ids) + 1)
    adjacent_matrix = np.array([
        [row["adjacent_cosine_distance"] for row in adjacent_rows
         if row["replicate"] == replicate]
        for replicate in range(1, len(runs) + 1)
    ])
    axes[0].plot(x, adjacent_matrix.mean(axis=0), color="#ff595e", marker="o",
                 markersize=3, label="mean across replicates")
    if len(runs) > 1:
        axes[0].fill_between(x, adjacent_matrix.min(axis=0),
                             adjacent_matrix.max(axis=0), color="#ff595e",
                             alpha=0.18, label="replicate range")
    axes[0].set_ylabel("Adjacent response distance")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    if replicate_rows:
        axes[1].plot(np.arange(1, len(checkpoint_ids) + 1),
                     [row["mean_pairwise_cosine_distance"]
                      for row in replicate_rows], color="#8ac926", marker="o",
                     markersize=3)
    axes[1].set_ylabel("Same-checkpoint repeat distance")
    axes[1].set_xlabel("Checkpoint number")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Pilot 003 — DeepSeek interpretive response")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(C.FIGURE_DIR / f"interpretation_trajectory.{suffix}", dpi=180)
    plt.close(fig)
    usage = []
    for replicate, run in enumerate(runs, 1):
        prompt_tokens = sum(row["provider_metadata"]["usage"].get("prompt_tokens", 0)
                            for row in run)
        completion_tokens = sum(
            row["provider_metadata"]["usage"].get("completion_tokens", 0)
            for row in run)
        usage.append({
            "replicate": replicate,
            "model": run[0]["model"],
            "checkpoints": len(run),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "mean_output_words": float(np.mean([word_count(row["parsed"])
                                                  for row in run])),
            "outputs_within_120_180_words": sum(
                120 <= word_count(row["parsed"]) <= 180 for row in run),
        })
    C.write_csv(C.TABLE_DIR / "interpretation_usage.csv", usage)
    adjacent_means = adjacent_matrix.mean(axis=1)
    repeat_values = [row["mean_pairwise_cosine_distance"] for row in replicate_rows]
    C.write_json(C.RESULTS_DIR / "interpretation_summary.json", {
        "runs": usage,
        "mean_adjacent_distance": float(adjacent_matrix.mean()),
        "mean_adjacent_distance_by_replicate": adjacent_means.tolist(),
        "adjacent_trajectory_pearson_between_first_two_runs": (
            float(np.corrcoef(adjacent_matrix[0], adjacent_matrix[1])[0, 1])
            if len(runs) >= 2 else None),
        "mean_repeat_distance": (float(np.mean(
            repeat_values))
            if replicate_rows else None),
        "fraction_mean_adjacent_above_destination_repeat_distance": (
            float(np.mean(adjacent_matrix.mean(axis=0) > np.array(repeat_values[1:])))
            if replicate_rows else None),
    })
    print(json.dumps(json.loads(
        (C.RESULTS_DIR / "interpretation_summary.json").read_text(encoding="utf-8")),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
