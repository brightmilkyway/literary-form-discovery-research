"""Figures for Pilot 001.  All figures are saved as PNG + SVG at 150 dpi."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR


def _save(fig, name: str, also_svg: bool = True) -> Path:
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    if also_svg:
        fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


# --------------------------------------------------------------------------
# TextTiling
# --------------------------------------------------------------------------

def plot_texttiling(runs: List[dict], labels: List[str], name: str) -> Path:
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), squeeze=False)
    for ax, run, lab in zip(axes[:, 0], runs, labels):
        x = np.array(run["gap_word_pos"])
        raw = np.array(run["block_sim"])
        sm = np.array(run["smoothed"])
        depth = np.array(run["depth"])
        ax.plot(x, raw, color="#a0aec0", lw=0.8, label="block cosine similarity")
        ax.plot(x, sm, color="#2b6cb0", lw=1.4, label="smoothed similarity")
        ax.plot(x, depth, color="#c05621", lw=1.2, alpha=0.9, label="depth score")
        ax.axhline(run["threshold"], color="#c05621", ls=":", lw=1.0,
                   label=f"threshold ({run['params']['cutoff_policy']})")
        for b in run["boundaries"]:
            ax.axvline(b["word_pos"], color="#718096", lw=0.5, alpha=0.45)
        ax.set_title(f"{lab}  ({len(run['boundaries'])} candidate boundaries)",
                     fontsize=10)
        ax.set_xlabel("word position")
        ax.set_ylabel("score")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("TextTiling — lexical cohesion change along the text", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, name)


# --------------------------------------------------------------------------
# Similarity / distance heat maps
# --------------------------------------------------------------------------

def plot_heatmap(M: np.ndarray, name: str, title: str,
                 value_label: str = "similarity",
                 cmap: str = "viridis",
                 invert: bool = False,
                 unit_word_pos: Sequence[int] | None = None,
                 highlight: Sequence[tuple[int, int]] = ()) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    data = -M if invert else M
    im = ax.imshow(data, cmap=cmap, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(value_label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("unit index (reading order)")
    ax.set_ylabel("unit index (reading order)")
    n = M.shape[0]
    step = max(1, n // 15)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for (i, j) in highlight:
        ax.plot([j], [i], marker="o", ms=5, mfc="none", mec="red", mew=1.2)
        ax.plot([i], [j], marker="o", ms=5, mfc="none", mec="red", mew=1.2)
    fig.tight_layout()
    return _save(fig, name)


def plot_heatmap_pair(M1: np.ndarray, M2: np.ndarray, name: str,
                      t1: str, t2: str, l1: str, l2: str,
                      highlight=()) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    for ax, M, t, lab in ((axes[0], M1, t1, l1), (axes[1], M2, t2, l2)):
        im = ax.imshow(M, cmap="viridis", aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(lab)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("unit index")
        ax.set_ylabel("unit index")
        for (i, j) in highlight:
            ax.plot([j], [i], "o", ms=5, mfc="none", mec="red", mew=1.2)
            ax.plot([i], [j], "o", ms=5, mfc="none", mec="red", mew=1.2)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Adjacency profiles
# --------------------------------------------------------------------------

def plot_adjacency(profiles: Dict[str, Dict[int, float]], name: str,
                   ylabel: str = "mean similarity") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, prof in profiles.items():
        gaps = sorted(prof)
        ys = [prof[g] for g in gaps]
        ax.plot(gaps, ys, marker="o", ms=3, lw=1.2, label=lab)
    ax.set_xlabel("index distance between units  (|i − j|)")
    ax.set_ylabel(ylabel)
    ax.set_title("Adjacency effect: similarity vs. distance in reading order",
                 fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Method agreement
# --------------------------------------------------------------------------

def plot_method_agreement(labels: List[str], matrix: np.ndarray, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("correlation")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Pairwise agreement of method matrices\n(upper-triangle Pearson)",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, name)
