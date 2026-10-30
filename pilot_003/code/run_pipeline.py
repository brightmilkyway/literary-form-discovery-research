"""Run the deterministic, no-download Pilot 003 representation pipeline."""

from __future__ import annotations

import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import common as C
import interpretive


def vectorizer(method: str, features: int) -> HashingVectorizer:
    if method == "char_hash":
        return HashingVectorizer(analyzer="char", ngram_range=(3, 5),
                                 n_features=features, alternate_sign=False,
                                 norm="l2", lowercase=True)
    if method == "word_hash":
        return HashingVectorizer(analyzer="word", ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[A-Za-z][A-Za-z’']*\b",
                                 n_features=features, alternate_sign=False,
                                 norm="l2", lowercase=True)
    raise ValueError(method)


def texts_at_scale(body: str, paras: list[C.Paragraph], checkpoints: list[dict],
                   scale: str) -> list[str]:
    texts = []
    for checkpoint in checkpoints:
        end = checkpoint["paragraph_index"]
        if scale == "cumulative":
            texts.append(body[:checkpoint["char_end"]])
        else:
            width = int(scale[1:])
            start = max(0, end - width + 1)
            texts.append(body[paras[start].char_start:paras[end].char_end])
    return texts


def pairwise_percentile(value: float, all_distances: np.ndarray) -> float:
    return float((np.count_nonzero(all_distances <= value) + 0.5) /
                 (len(all_distances) + 1.0))


def permutation_summary(matrix, observed_mean: float, iterations: int,
                        seed: int) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    n = matrix.shape[0]
    for i in range(iterations):
        order = rng.permutation(n)
        means[i] = np.mean([1.0 - matrix[order[j - 1], order[j]]
                            for j in range(1, n)])
    p_lower = (np.count_nonzero(means <= observed_mean) + 1) / (iterations + 1)
    p_upper = (np.count_nonzero(means >= observed_mean) + 1) / (iterations + 1)
    return {
        "observed_mean_adjacent_distance": observed_mean,
        "permutation_mean": float(means.mean()),
        "permutation_sd": float(means.std(ddof=1)),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_two_sided": float(min(1.0, 2 * min(p_lower, p_upper))),
    }


def plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    scale_order = ["p1", "p4", "cumulative"]
    colors = {"char_hash": "#345995", "word_hash": "#d1495b"}
    for axis, scale in zip(axes, scale_order):
        for method in colors:
            subset = [r for r in rows if r["scale"] == scale and r["method"] == method]
            axis.plot([r["word_fraction"] for r in subset],
                      [r["adjacent_cosine_distance"] for r in subset],
                      marker="o", markersize=3, linewidth=1.2,
                      color=colors[method], label=method)
        axis.set_title(f"Representation object: {scale}")
        axis.set_ylabel("Adjacent cosine distance")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Fraction of source words seen")
    fig.suptitle("Pilot 003 — prefix-ordered representation response")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(C.FIGURE_DIR / f"representation_trajectory.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.time()
    config = C.load_config()
    body, title_offset = C.load_body()
    paras = C.paragraphs(body)
    checkpoints = C.build_checkpoints(body, paras, config["checkpoint_count"])
    if checkpoints[-1]["char_end"] != paras[-1].char_end:
        raise RuntimeError("last checkpoint does not reach the final paragraph")
    C.write_csv(C.TABLE_DIR / "checkpoints.csv", checkpoints)
    C.write_json(C.DERIVED_DIR / "checkpoints.json", checkpoints)

    trajectory_rows: list[dict] = []
    summaries: list[dict] = []
    scales = [f"p{x}" for x in config["local_paragraph_scales"]] + ["cumulative"]
    for method_index, method in enumerate(("char_hash", "word_hash")):
        encoder = vectorizer(method, config["hash_features"])
        for scale_index, scale in enumerate(scales):
            texts = texts_at_scale(body, paras, checkpoints, scale)
            vectors = encoder.transform(texts)
            similarities = cosine_similarity(vectors)
            triangle = 1.0 - similarities[np.triu_indices(len(checkpoints), k=1)]
            method_rows = []
            for i in range(1, len(checkpoints)):
                distance = float(1.0 - similarities[i - 1, i])
                row = {
                    "checkpoint_id": checkpoints[i]["checkpoint_id"],
                    "checkpoint_number": checkpoints[i]["checkpoint_number"],
                    "word_fraction": checkpoints[i]["word_fraction"],
                    "paragraph_gap": checkpoints[i]["paragraph_gap"],
                    "method": method,
                    "scale": scale,
                    "adjacent_cosine_distance": distance,
                    "all_pairs_percentile": pairwise_percentile(distance, triangle),
                }
                method_rows.append(row)
                trajectory_rows.append(row)
            distances = np.array([r["adjacent_cosine_distance"] for r in method_rows])
            peak = method_rows[int(np.argmax(distances))]
            permutation = permutation_summary(
                similarities, float(distances.mean()),
                config["permutation_iterations"],
                config["random_seed"] + method_index * 10 + scale_index)
            summaries.append({
                "method": method,
                "scale": scale,
                "n_transitions": len(method_rows),
                "mean_adjacent_distance": float(distances.mean()),
                "sd_adjacent_distance": float(distances.std(ddof=1)),
                "median_adjacent_distance": float(np.median(distances)),
                "maximum_adjacent_distance": float(distances.max()),
                "maximum_checkpoint_id": peak["checkpoint_id"],
                "maximum_word_fraction": peak["word_fraction"],
                "mean_all_pairs_percentile": float(np.mean(
                    [r["all_pairs_percentile"] for r in method_rows])),
                **permutation,
            })

    C.write_csv(C.TABLE_DIR / "representation_trajectory.csv", trajectory_rows)
    C.write_csv(C.TABLE_DIR / "representation_summary.csv", summaries)
    agreement = []
    for scale in scales:
        char_values = [r["adjacent_cosine_distance"] for r in trajectory_rows
                       if r["scale"] == scale and r["method"] == "char_hash"]
        word_values = [r["adjacent_cosine_distance"] for r in trajectory_rows
                       if r["scale"] == scale and r["method"] == "word_hash"]
        agreement.append({
            "scale": scale,
            "n_transitions": len(char_values),
            "pearson_r": float(np.corrcoef(char_values, word_values)[0, 1]),
            "spearman_rho": float(spearmanr(char_values, word_values).statistic),
        })
    C.write_csv(C.TABLE_DIR / "method_agreement.csv", agreement)
    plot(trajectory_rows)
    request_path = interpretive.prepare()
    metadata = {
        "status": "completed_representation_component",
        "protocol": "protocol.md",
        "source": str(C.SOURCE.relative_to(C.ROOT)).replace("\\", "/"),
        "source_sha256": C.source_sha256(),
        "title_offset": title_offset,
        "body_characters": len(body),
        "body_words": len(C.WORD_RE.findall(body)),
        "paragraphs": len(paras),
        "checkpoints": len(checkpoints),
        "representation_methods": ["char_hash", "word_hash"],
        "representation_scales": scales,
        "hash_features": config["hash_features"],
        "permutation_iterations": config["permutation_iterations"],
        "random_seed": config["random_seed"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "interpretive_status": "separate component; see interpretation_summary.json",
        "interpretive_requests": str(request_path.relative_to(C.ROOT)).replace("\\", "/"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    C.write_json(C.RESULTS_DIR / "representation_metadata.json", metadata)
    C.write_json(C.RESULTS_DIR / "summary.json", {
        "representations": summaries, "method_agreement": agreement})
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
