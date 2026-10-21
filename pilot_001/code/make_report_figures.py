"""Extra figures for the report: side-by-side heat maps and a sensitivity panel.

Writes:
* ``results/figures/fig_three_heatmaps.png`` — NCD / SBERT / TF-IDF on group150
* ``results/figures/fig_sensitivity_summary.png`` — what is sensitive to what

Run from code/:  python make_report_figures.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
import baselines as BL

DERIVED = C.DATA_DIR / "derived"


def main():
    units = json.loads((DERIVED / "units.json").read_text(encoding="utf-8"))["group150"]
    NCD = np.load(DERIVED / "ncd_group150_zlib.npy")
    SB = np.load(DERIVED / "sbert_group150.npy")
    TF = np.load(DERIVED / "tfidf_group150.npy")
    LN = np.load(DERIVED / "length_group150.npy")
    res = json.loads((DERIVED / "results_summary.json").read_text(encoding="utf-8"))

    # ---- 1. three heat maps ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.0))
    panels = [
        (NCD, "NCD distance (zlib) — lower = more similar", "magma", None),
        (SB, "Sentence-BERT cosine — higher = more similar", "viridis", (0.15, 0.85)),
        (TF, "TF-IDF cosine — higher = more similar", "viridis", None),
    ]
    for ax, (M, title, cmap, lim) in zip(axes, panels):
        im = ax.imshow(M, cmap=cmap, aspect="equal",
                       vmin=(lim[0] if lim else None),
                       vmax=(lim[1] if lim else None))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("unit index")
        ax.set_ylabel("unit index")
    fig.suptitle("Three representations of the same 78 units "
                 "(150-word paragraph groups)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(C.FIG_DIR / "fig_three_heatmaps.png", dpi=150, bbox_inches="tight")
    fig.savefig(C.FIG_DIR / "fig_three_heatmaps.svg", bbox_inches="tight")
    plt.close(fig)

    # ---- 2. sensitivity summary -----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    # (a) method agreement with content-free baselines
    names = ["NCD", "SBERT", "TF-IDF"]
    to_len = [BL.compare_matrices(1 - NCD, LN)["pearson"],
              BL.compare_matrices(SB, LN)["pearson"],
              BL.compare_matrices(TF, LN)["pearson"]]
    ax = axes[0]
    bars = ax.bar(names, to_len, color=["#805ad5", "#2b6cb0", "#38a169"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-0.1, 0.8)
    ax.set_ylabel("Pearson r with content-free length baseline")
    ax.set_title("(a) How much is 'length'?\nNCD is mostly length; SBERT is not", fontsize=10)
    for b, v in zip(bars, to_len):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)

    # (b) TextTiling boundary count vs w
    ax = axes[1]
    sweep = res["texttiling"]["sweep"]
    ws, ks, labels = [], [], []
    for lab, v in sweep.items():
        if lab in ("w10_k6", "w20_k6 (main)", "w40_k6"):
            ws.append(v["params"]["w"])
            ks.append(v["n_boundaries"])
            labels.append(lab)
    order = np.argsort(ws)
    ax.plot(np.array(ws)[order], np.array(ks)[order], "o-", color="#c05621", lw=1.6)
    for x, y in zip(np.array(ws)[order], np.array(ks)[order]):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xlabel("token-sequence length w (k = 6)")
    ax.set_ylabel("candidate boundaries")
    ax.set_title("(b) TextTiling is highly parameter-sensitive\n66 → 271 boundaries",
                 fontsize=10)

    # (c) NCD vs SBERT vs TF-IDF: similarity decay with distance
    ax = axes[2]
    for lab, M in (("NCD (1-d)", 1 - NCD), ("SBERT", SB), ("TF-IDF", TF)):
        prof = BL.adjacency_matrix_profile(M)
        g = sorted(prof)
        ax.plot(g, [prof[x] for x in g], lw=1.4, label=lab)
    ax.set_xlabel("index distance |i - j|")
    ax.set_ylabel("mean similarity")
    ax.set_title("(c) Only SBERT decays with distance\nNCD's profile is nearly flat",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_sensitivity_summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(C.FIG_DIR / "fig_sensitivity_summary.svg", bbox_inches="tight")
    plt.close(fig)

    print("wrote fig_three_heatmaps and fig_sensitivity_summary")

    # ---- 3. before/after the two representation fixes --------------------
    NCD_res = np.load(DERIVED / "ncd_group150_zlib_residual.npy")
    SB_po = np.load(DERIVED / "sbert_group150_pooled.npy")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 12.0))
    panels = [
        (axes[0, 0], SB, "SBERT BEFORE — 25/78 units truncated (32%)",
         "viridis", None),
        (axes[0, 1], SB_po, "SBERT AFTER — paragraph pooling, 0/184 truncated",
         "viridis", None),
        (axes[1, 0], NCD, "NCD BEFORE — r(distance, size diff) = +0.689",
         "magma", None),
        (axes[1, 1], NCD_res, "NCD AFTER — size term regressed out, r = 0.000",
         "RdBu_r", None),
    ]
    for ax, M, title, cmap, lim in panels:
        if cmap == "RdBu_r":
            v = np.abs(M[~np.isinf(M)])
            vmax = np.quantile(v, 0.98)
            im = ax.imshow(M, cmap=cmap, aspect="equal", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(M, cmap=cmap, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("unit index")
        ax.set_ylabel("unit index")
    fig.suptitle("Representation fixes: SBERT truncation and the NCD length confound",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(C.FIG_DIR / "fig_fix_before_after.png", dpi=150, bbox_inches="tight")
    fig.savefig(C.FIG_DIR / "fig_fix_before_after.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_fix_before_after")


if __name__ == "__main__":
    main()
