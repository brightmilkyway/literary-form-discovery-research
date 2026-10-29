"""Compute token surprisal with a local causal LM and explicit offsets."""

from __future__ import annotations

import json
import math
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C


def chunk_plan(n_tokens: int, maximum: int, stride: int) -> list[tuple[int, int, int]]:
    """Return (context_start, target_start, target_end), covering tokens 1..n-1."""
    if not 0 < stride < maximum:
        raise ValueError("stride must be positive and below model context length")
    plan = []
    target_start = 1
    while target_start < n_tokens:
        target_end = min(target_start + stride, n_tokens)
        context_start = max(0, target_end - maximum)
        if context_start >= target_start:
            raise RuntimeError("chunk has no predictive context")
        plan.append((context_start, target_start, target_end))
        target_start = target_end
    return plan


def checkpoint_summary(token_rows: list[dict], checkpoints: list[dict]) -> list[dict]:
    rows = []
    previous_char_end = 0
    previous_mean = None
    for checkpoint in checkpoints:
        values = np.array([
            row["surprisal_nats"] for row in token_rows
            if previous_char_end < row["char_end"] <= checkpoint["char_end"]
        ], dtype=float)
        if not len(values):
            raise RuntimeError(f"no tokens assigned to {checkpoint['checkpoint_id']}")
        mean = float(values.mean())
        rows.append({
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_number": checkpoint["checkpoint_number"],
            "word_fraction": checkpoint["word_fraction"],
            "tokens": len(values),
            "mean_surprisal_nats": mean,
            "sd_surprisal_nats": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median_surprisal_nats": float(np.median(values)),
            "p90_surprisal_nats": float(np.quantile(values, 0.9)),
            "mean_change": (mean - previous_mean) if previous_mean is not None else "",
        })
        previous_mean = mean
        previous_char_end = checkpoint["char_end"]
    return rows


def plot(token_rows: list[dict], summaries: list[dict]) -> None:
    values = np.array([row["surprisal_nats"] for row in token_rows])
    char_ends = np.array([row["char_end"] for row in token_rows])
    width = 200
    rolling = np.convolve(values, np.ones(width) / width, mode="valid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(char_ends[width - 1:] / char_ends[-1], rolling,
                 color="#6a4c93", linewidth=1)
    axes[0].set_ylabel("200-token mean surprisal (nats)")
    axes[0].set_xlabel("Fraction of body characters")
    axes[0].grid(alpha=0.25)
    axes[1].plot([row["word_fraction"] for row in summaries],
                 [row["mean_surprisal_nats"] for row in summaries],
                 marker="o", markersize=3, color="#1982c4")
    axes[1].set_ylabel("Checkpoint interval mean (nats)")
    axes[1].set_xlabel("Fraction of source words seen")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Pilot 003 — DistilGPT2 predictive response")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(C.FIGURE_DIR / f"predictive_trajectory.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.time()
    config = C.load_config()["predictive"]
    body, _ = C.load_body()
    checkpoints = C.build_checkpoints(body, C.paragraphs(body),
                                      C.load_config()["checkpoint_count"])
    model_name = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Tokenization is offset mapping only; model inference remains chunked.
    tokenizer.model_max_length = 10**9
    encoded = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(config["device"])
    maximum = int(getattr(model.config, "n_positions",
                          getattr(model.config, "max_position_embeddings")))
    plan = chunk_plan(len(token_ids), maximum, config["stride"])
    surprisals = np.full(len(token_ids), np.nan, dtype=float)
    contexts = np.zeros(len(token_ids), dtype=int)
    with torch.inference_mode():
        for number, (context_start, target_start, target_end) in enumerate(plan, 1):
            ids = torch.tensor(token_ids[context_start:target_end],
                               dtype=torch.long, device=config["device"])[None, :]
            logits = model(ids).logits[0]
            local_start = target_start - context_start
            local_end = target_end - context_start
            positions = torch.arange(local_start - 1, local_end - 1,
                                     device=config["device"])
            targets = ids[0, local_start:local_end]
            losses = torch.nn.functional.cross_entropy(
                logits[positions], targets, reduction="none")
            surprisals[target_start:target_end] = losses.cpu().numpy()
            contexts[target_start:target_end] = np.arange(
                local_start, local_end, dtype=int)
            print(f"[{number:02d}/{len(plan)}] tokens {target_start}:{target_end}",
                  flush=True)
    if not np.isfinite(surprisals[1:]).all():
        raise RuntimeError("not all predictable tokens received surprisal")
    token_rows = []
    for i in range(1, len(token_ids)):
        start, end = offsets[i]
        token_rows.append({
            "token_index": i,
            "token_id": token_ids[i],
            "token_text": tokenizer.decode([token_ids[i]]),
            "char_start": start,
            "char_end": end,
            "context_tokens": int(contexts[i]),
            "surprisal_nats": float(surprisals[i]),
            "surprisal_bits": float(surprisals[i] / math.log(2)),
        })
    summaries = checkpoint_summary(token_rows, checkpoints)
    C.write_csv(C.TABLE_DIR / "token_surprisal.csv", token_rows)
    C.write_csv(C.TABLE_DIR / "predictive_summary.csv", summaries)
    plot(token_rows, summaries)
    revision = (getattr(model.config, "_commit_hash", None) or
                tokenizer.init_kwargs.get("_commit_hash"))
    metadata = {
        "status": "completed",
        "model_requested": model_name,
        "model_type": model.config.model_type,
        "model_revision": revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocabulary_size": tokenizer.vocab_size,
        "tokens_in_body": len(token_ids),
        "scored_tokens": len(token_rows),
        "first_token_unscored": True,
        "context_length": maximum,
        "stride": config["stride"],
        "context_tokens_min": int(contexts[1:].min()),
        "context_tokens_max": int(contexts[1:].max()),
        "device": config["device"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "source_sha256": C.source_sha256(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    C.write_json(C.RESULTS_DIR / "predictive_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
