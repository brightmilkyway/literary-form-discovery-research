"""Pilot 001 — run the whole exploratory pipeline.

Usage (from the code/ directory):
    python run_all.py            # everything
    python run_all.py --no-sbert # skip the neural encoder
    python run_all.py --only ncd

Outputs go to ../results/{figures,tables} and ../data/derived.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import common as C
import texttiling as TT
import ncd as NCD
import baselines as BL
import plots as P

DERIVED = C.DATA_DIR / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def top_pairs_from_similarity(S: np.ndarray, units, top=15, min_gap=2):
    n = len(units)
    pairs = []
    for i in range(n):
        for j in range(i + min_gap, n):
            pairs.append((float(S[i, j]), i, j))
    pairs.sort(key=lambda t: -t[0])
    out = []
    for s, i, j in pairs[:top]:
        out.append({
            "i": i, "j": j, "similarity": s, "gap": j - i,
            "i_span": [units[i].start, units[i].end],
            "j_span": [units[j].start, units[j].end],
            "i_preview": _prev(units[i].text),
            "j_preview": _prev(units[j].text),
        })
    return out


def _prev(text, width=120):
    import re
    s = re.sub(r"\s+", " ", text).strip()
    return s[:width] + ("…" if len(s) > width else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sbert", action="store_true")
    ap.add_argument("--only", default="all")
    args = ap.parse_args()

    t0 = time.time()
    C.set_seed(42)
    raw = C.load_raw()
    body, off = C.strip_title(raw)
    paras = C.paragraphs(body)
    log(f"[data] body={len(body)} chars, {len(paras)} paragraphs, "
        f"{sum(p.n_words for p in paras)} words")

    # ---- units -----------------------------------------------------------
    U150 = C.build_units(body, "group", target_words=150)
    U250 = C.build_units(body, "group", target_words=250)
    Uwin = C.build_units(body, "window", window=150, stride=75)
    Upar = C.build_units(body, "paragraph")
    log(f"[units] group150={len(U150)} group250={len(U250)} "
        f"window150/75={len(Uwin)} paragraphs={len(Upar)}")

    units_meta = {
        "group150": [u.to_dict() for u in U150],
        "group250": [u.to_dict() for u in U250],
        "window150_75": [u.to_dict() for u in Uwin],
    }
    C.save_json(units_meta, DERIVED / "units.json")

    results = {
        "meta": {
            "source": "James Joyce, The Dead (Project Gutenberg text)",
            "seed": C.SEED,
            "source_chars": len(raw),
            "body_chars": len(body),
            "n_paragraphs": len(paras),
            "n_words": sum(p.n_words for p in paras),
            "title_offset": off,
        }
    }

    # ---- TextTiling ------------------------------------------------------
    if args.only in ("all", "texttiling"):
        log("[texttiling] running parameter sweep …")
        tt_runs, tt_labels = [], []
        sweep = [
            ("w20_k6 (main)", dict(w=20, k=6)),
            ("w20_k10 (NLTK default)", dict(w=20, k=10)),
            ("w10_k6", dict(w=10, k=6)),
            ("w40_k6", dict(w=40, k=6)),
            ("w20_k4", dict(w=20, k=4)),
            ("w20_k6 no-smoothing", dict(w=20, k=6, smoothing_rounds=0)),
            ("w20_k6 LC cutoff", dict(w=20, k=6, cutoff_policy="LC")),
            ("w20_k6 word-tokenised", dict(w=20, k=6, tokenize_mode="word")),
        ]
        tt_summary = {}
        for label, kw in sweep:
            r = TT.run_texttiling(body, paras, units=U150, **kw)
            tt_summary[label] = {
                "params": r["params"], "n_boundaries": len(r["boundaries"]),
                "threshold": r["threshold"], "mean_depth": r["mean_depth"],
                "sd_depth": r["sd_depth"],
            }
            if label in ("w20_k6 (main)", "w20_k10 (NLTK default)",
                         "w10_k6", "w40_k6", "w20_k6 no-smoothing",
                         "w20_k6 LC cutoff"):
                tt_runs.append(r)
                tt_labels.append(label)

        main_run = TT.run_texttiling(body, paras, w=20, k=6, units=U150)
        results["texttiling"] = {
            "sweep": tt_summary,
            "main": {
                "params": main_run["params"],
                "n_boundaries": len(main_run["boundaries"]),
                "threshold": main_run["threshold"],
                "mean_depth": main_run["mean_depth"],
                "sd_depth": main_run["sd_depth"],
                "boundaries": main_run["boundaries"],
                "top20_by_depth": sorted(
                    main_run["boundaries"], key=lambda b: -b["depth"])[:20],
            },
        }
        P.plot_texttiling(tt_runs, tt_labels, "fig_texttiling_sweep")
        # boundary list with quotes for the main run
        rows = []
        for b in main_run["boundaries"]:
            rows.append({
                "word_pos": b["word_pos"], "char_pos": b["char_pos"],
                "depth": round(b["depth"], 4),
                "block_sim": round(b["block_sim"], 4),
                "left_unit": b["left_unit"], "right_unit": b["right_unit"],
                "quote": _prev(body[b["char_pos"]:b["char_pos"] + 400], 140),
            })
        _write_csv(rows, C.TABLE_DIR / "texttiling_boundaries.csv")
        C.save_json({"main": main_run, "sweep": tt_summary},
                    DERIVED / "texttiling_full.json")
        log(f"[texttiling] main boundaries: {len(main_run['boundaries'])} "
            f"(sweep counts: "
            f"{ {k: v['n_boundaries'] for k, v in tt_summary.items()} })")

    # ---- NCD -------------------------------------------------------------
    if args.only in ("all", "ncd"):
        log("[ncd] zlib on group150 …")
        res_zlib = NCD.ncd_matrix(U150, "zlib")
        results["ncd"] = {"group150_zlib": {
            "n_pairs": len(U150) * (len(U150) - 1) // 2,
            "size_stats": {"min": float(res_zlib.sizes.min()),
                           "median": float(np.median(res_zlib.sizes)),
                           "max": float(res_zlib.sizes.max())},
            "length_confound": NCD.length_confound(res_zlib),
            "top_non_adjacent": NCD.non_adjacent_pairs(res_zlib, top=20, min_gap=3),
            "adjacency_profile": NCD.adjacent_profile(res_zlib),
        }}
        np.save(DERIVED / "ncd_group150_zlib.npy", res_zlib.matrix)

        log("[ncd] compressor sensitivity (zlib / bz2 / lzma) on group150 …")
        matrices, profs = {"NCD-zlib": res_zlib.matrix}, {}
        profs["NCD-zlib"] = {int(g): v["mean"] for g, v in
                             NCD.adjacent_profile(res_zlib).items()}
        comp_results = {}
        for comp in ("bz2", "lzma"):
            r = NCD.ncd_matrix(U150, comp)
            matrices[f"NCD-{comp}"] = r.matrix
            profs[f"NCD-{comp}"] = {int(g): v["mean"] for g, v in
                                    NCD.adjacent_profile(r).items()}
            comp_results[comp] = {
                "corr_with_zlib": BL.compare_matrices(res_zlib.matrix, r.matrix),
                "length_confound": NCD.length_confound(r),
            }
            np.save(DERIVED / f"ncd_group150_{comp}.npy", r.matrix)
        results["ncd"]["compressor_sensitivity"] = comp_results

        log("[ncd] chunk-size sensitivity (group250, paragraph) …")
        r250 = NCD.ncd_matrix(U250, "zlib")
        np.save(DERIVED / "ncd_group250_zlib.npy", r250.matrix)
        results["ncd"]["group250_zlib"] = {
            "length_confound": NCD.length_confound(r250),
            "top_non_adjacent": NCD.non_adjacent_pairs(r250, top=20, min_gap=3),
        }
        rpar = NCD.ncd_matrix(Upar, "zlib")
        np.save(DERIVED / "ncd_paragraph_zlib.npy", rpar.matrix)
        results["ncd"]["paragraph_zlib"] = {
            "length_confound": NCD.length_confound(rpar),
        }

        # figures
        P.plot_heatmap(res_zlib.matrix, "fig_ncd_group150_heatmap",
                       "NCD distance heat map — zlib, 150-word paragraph groups",
                       value_label="NCD distance (0 = identical)",
                       cmap="magma",
                       highlight=[(p["i"], p["j"]) for p in
                                  NCD.non_adjacent_pairs(res_zlib, top=8, min_gap=3)])
        P.plot_adjacency(profs, "fig_ncd_adjacency",
                         ylabel="mean NCD distance")
        C.save_json({"profiles": profs}, DERIVED / "ncd_adjacency.json")
        log("[ncd] done")

    # ---- SBERT -----------------------------------------------------------
    if args.only in ("all", "sbert") and not args.no_sbert:
        log("[sbert] encoding group150 …")
        try:
            import sbert_semantic as SB
            model, emb, info = SB.encode_units(U150)
            S = SB.similarity_from_embeddings(emb)
            np.save(DERIVED / "sbert_group150.npy", S)
            np.save(DERIVED / "sbert_group150_emb.npy", emb)
            results["sbert"] = {
                "info": info,
                "top_non_adjacent": top_pairs_from_similarity(S, U150, top=20, min_gap=3),
                "adjacency_profile": BL.adjacency_matrix_profile(S),
            }
            P.plot_heatmap(S, "fig_sbert_group150_heatmap",
                           "Sentence-BERT cosine similarity — all-MiniLM-L6-v2, "
                           "150-word paragraph groups",
                           value_label="cosine similarity",
                           highlight=[(p["i"], p["j"]) for p in
                                      top_pairs_from_similarity(S, U150, top=8, min_gap=3)])
            log("[sbert] done")
        except Exception as e:  # noqa: BLE001
            log(f"[sbert] FAILED: {e!r}")
            results["sbert"] = {"error": repr(e)}

    # ---- baselines + agreement ------------------------------------------
    if args.only in ("all", "baselines"):
        log("[baselines] TF-IDF / Jaccard / length on group150 …")
        T = BL.tfidf_matrix(U150)
        J = BL.jaccard_matrix(U150)
        L = BL.length_similarity(U150)
        np.save(DERIVED / "tfidf_group150.npy", T)
        np.save(DERIVED / "jaccard_group150.npy", J)
        np.save(DERIVED / "length_group150.npy", L)

        # NCD similarity = 1 - distance (for comparison only)
        res_zlib = NCD.ncd_matrix(U150, "zlib")
        N_sim = 1.0 - res_zlib.matrix
        mats = {"NCD": N_sim, "TF-IDF": T, "Jaccard": J, "length": L}
        if (DERIVED / "sbert_group150.npy").exists():
            mats["SBERT"] = np.load(DERIVED / "sbert_group150.npy")

        names = list(mats)
        agree = np.ones((len(names), len(names)))
        agree_tbl = {}
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                c = BL.compare_matrices(mats[names[a]], mats[names[b]])
                agree[a, b] = agree[b, a] = c["pearson"]
                agree_tbl[f"{names[a]}|{names[b]}"] = c
        results["method_agreement"] = agree_tbl
        P.plot_method_agreement(names, agree, "fig_method_agreement")
        P.plot_heatmap(T, "fig_tfidf_group150_heatmap",
                       "TF-IDF cosine baseline — 150-word paragraph groups",
                       value_label="cosine similarity")
        P.plot_adjacency(
            {k: BL.adjacency_matrix_profile(v) for k, v in mats.items()},
            "fig_baseline_adjacency", ylabel="mean similarity")

        results["baselines"] = {
            "tfidf_top_non_adjacent": top_pairs_from_similarity(T, U150, 10, 3),
            "jaccard_top_non_adjacent": top_pairs_from_similarity(J, U150, 10, 3),
        }
        log("[baselines] done")

    C.save_json(results, DERIVED / "results_summary.json")
    log(f"[done] total {time.time() - t0:.1f}s -> {DERIVED}")


def _write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
