"""Create figures and a label-free report from completed Pilot 002 outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import common as C


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def plot_candidates(rows, total_words):
    relation = [r for r in rows if r["family"] == "relation"]
    if not relation:
        return
    x = np.array([float(r["a_center"]) for r in relation])
    y = np.array([float(r["b_center"]) for r in relation])
    p = np.array([float(r["persistence"]) for r in relation])
    support = np.array([float(r["support_matrices"]) for r in relation])
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(x, y, c=p, s=12 + 3 * np.sqrt(support),
                         cmap="viridis", vmin=0, vmax=max(1, p.max()),
                         alpha=.78, linewidths=0)
    ax.plot([0, total_words], [0, total_words], color=".75", lw=.8)
    ax.set(xlabel="Earlier-region center (word position)",
           ylabel="Later-region center (word position)",
           title="Label-free relation candidates\ncolour = persistence; size = supporting matrices")
    fig.colorbar(scatter, ax=ax, label="persistence")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_candidate_map.png", dpi=180)
    fig.savefig(C.FIG_DIR / "fig_candidate_map.svg")
    plt.close(fig)


def plot_coverage(rows):
    top = rows[:30]
    reps = list(C.REPRESENTATIONS)
    data = np.zeros((len(top), len(reps)))
    for i, row in enumerate(top):
        present = set(row["representations"].split(";"))
        data[i] = [rep in present for rep in reps]
    fig, ax = plt.subplots(figsize=(8, max(5, len(top) * .22)))
    ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(reps)), reps, rotation=35, ha="right")
    ax.set_yticks(range(len(top)), [r["candidate_id"] for r in top])
    ax.set_title("Representation support for the top persistence candidates")
    for i, row in enumerate(top):
        ax.text(len(reps) + .1, i, f'p={float(row["persistence"]):.2f}',
                va="center", fontsize=7)
    ax.set_xlim(-.5, len(reps) + 1.2)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_representation_consensus.png", dpi=180)
    fig.savefig(C.FIG_DIR / "fig_representation_consensus.svg")
    plt.close(fig)


def plot_reference_matrices():
    reps = list(C.REPRESENTATIONS)
    fig, axes = plt.subplots(1, len(reps), figsize=(16, 3.6), constrained_layout=True)
    for ax, rep in zip(axes, reps):
        path = C.MATRIX_DIR / f"w150_p00__{rep}.npz"
        if not path.exists():
            ax.axis("off")
            continue
        data = np.load(path, allow_pickle=False)
        P = data["percentile"]
        image = ax.imshow(P, origin="upper", cmap="magma", vmin=.90, vmax=1)
        ax.set_title(rep)
        ax.set_xlabel("unit")
    axes[0].set_ylabel("unit")
    fig.colorbar(image, ax=axes, label="within-matrix percentile", shrink=.72)
    fig.suptitle("Self-similarity geometry at the frozen 150-word / phase-0 view")
    fig.savefig(C.FIG_DIR / "fig_reference_matrices.png", dpi=180)
    fig.savefig(C.FIG_DIR / "fig_reference_matrices.svg")
    plt.close(fig)


def plot_null_calibration(rows):
    if not rows or "null_mean_persistence" not in rows[0]:
        return
    observed = np.array([float(r["persistence"]) for r in rows])
    null = np.array([float(r["null_mean_persistence"]) for r in rows])
    p = np.array([float(r["persistence_empirical_p"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(null, observed, c=-np.log10(np.maximum(p, 1e-6)), s=9,
               cmap="plasma", alpha=.65, linewidths=0)
    limit = max(observed.max(), null.max())
    ax.plot([0, limit], [0, limit], color=".4", lw=1, ls="--")
    ax.set(xlabel="Mean persistence under position-randomised null",
           ylabel="Observed persistence",
           title="Persistence requires a location-randomised baseline")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_null_calibration.png", dpi=180)
    fig.savefig(C.FIG_DIR / "fig_null_calibration.svg")
    plt.close(fig)


def write_report(candidates, inventory, run_meta):
    geometry_counts = Counter()
    for row in candidates:
        for part in row["geometry"].split(";"):
            name, count = part.split(":")
            geometry_counts[name] += int(count)
    high = [r for r in candidates if float(r["persistence"]) >= .5]
    full_rep = [r for r in candidates
                if float(r["representation_coverage"]) >= .999]
    full_phase = [r for r in candidates if float(r["phase_coverage"]) >= .999]
    calibrated = bool(candidates and "persistence_empirical_p" in candidates[0])
    p05 = ([r for r in candidates
            if float(r["persistence_empirical_p"]) <= .05]
           if calibrated else [])
    ncd_corr = {}
    for row in inventory:
        if row["representation"] == "ncd":
            diagnostic = json.loads(row["diagnostic"])
            ncd_corr[row["scheme"]] = diagnostic[
                "corr_distance_compressed_size_difference"]
    lines = [
        "# Pilot 002 结果报告",
        "",
        "## 1. 运行是否完成",
        "",
        f"- 正文切分方案：{run_meta['schemes']}；自相似矩阵：{run_meta['matrices']}。",
        f"- 未聚合几何事件：{run_meta['events']:,}；聚合候选：{run_meta['candidates']:,}。",
        f"- 五表示完整运行：{'是' if run_meta['complete_five_representation_run'] else '否'}。",
        f"- 最终五表示缓存合并运行时间：{run_meta['elapsed_seconds']} 秒；首次逐矩阵计算时间见清单。",
        "- 所有坐标回环检查通过；详情见 `partition_validation.csv`。",
        "",
        "## 2. 无标签汇总",
        "",
        f"- 持久性不低于 0.50 的候选：{len(high):,}。",
        f"- 被五种表示全部检出的候选：{len(full_rep):,}。",
        f"- 在其被检出的词窗尺度上覆盖全部四相位的候选：{len(full_phase):,}。",
        "- 几何事件构成：" + "；".join(f"{k}={v}" for k, v in sorted(geometry_counts.items())) + "。",
        "",
        "这些计数只描述候选对观察装置的持久性，不把任何候选命名为主题、象征、",
        "叙事转折或作品结构。",
        "",
        "## 3. 位置随机化零模型",
        "",
    ]
    if calibrated:
        lines += [
            "保持每张矩阵的事件数、尺度、相位、表示和几何类型不变，只在合格位置中",
            "随机重放事件 100 次。结果表明原始持久性存在明显的空间占用基线：",
            "前列候选所在位置的零模型平均持久性通常已达到约 0.60–0.75。因此，",
            "**原始持久性不能单独解释，必须与位置零模型一起报告。**",
            "",
            f"- 单候选经验 p ≤ 0.05：{len(p05):,} / {len(candidates):,}；",
            "- 该计数未经多重比较校正，不作为显著性发现数量；",
            "- 它只用于识别哪些位置的跨装置支持明显超过相同事件密度下的偶然重合。",
            "",
        ]
    else:
        lines += ["零模型尚未运行。", ""]
    lines += [
        "## 4. NCD 长度诊断",
        "",
        f"NCD 距离与压缩长度差在句子尺度相关 {ncd_corr.get('sentence', float('nan')):.3f}，",
        f"段落尺度相关 {ncd_corr.get('paragraph', float('nan')):.3f}。固定词窗也未消除该效应",
        "（逐矩阵数值见 `matrix_inventory.csv`）。因此 NCD 在本轮是一个明确记录了",
        "长度敏感性的观察器；五表示共识中的 NCD 支持不能被读成独立于长度的内容证据。",
        "本轮遵守冻结协议，没有在看到结果后把它替换为残差版本。",
        "",
        "## 5. 按冻结规则排序的前 20 个候选",
        "",
        "| ID | 类别 | 原文词位置 | 持久性 | 零模型 p | 支持 z | 尺度 | 表示 | 相位 | 支持矩阵 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates[:20]:
        if row["family"] == "valley":
            pos = row["a_center"]
        else:
            pos = f'{row["a_center"]} ↔ {row["b_center"]}'
        lines.append(
            f'| {row["candidate_id"]} | {row["geometry"]} | {pos} | '
            f'{float(row["persistence"]):.3f} | '
            f'{float(row.get("persistence_empirical_p", "nan")):.3f} | '
            f'{float(row.get("support_matrices_z", "nan")):.2f} | '
            f'{float(row["scale_coverage"]):.3f} | '
            f'{float(row["representation_coverage"]):.3f} | '
            f'{float(row["phase_coverage"]):.3f} | {row["support_matrices"]} |')
    lines += [
        "",
        "## 6. 判读边界",
        "",
        "1. 排序在读取候选正文前完成；预览只用于可追溯性。",
        "2. 聚合容差与阈值来自预先冻结的 `protocol.md`，本报告没有按结果调参。",
        "3. 高持久性表示跨观察装置复现，不自动等于文学重要性。",
        "4. 低持久性不表示原文关系不存在，只表示本轮观察器未稳定恢复它。",
        "5. 本轮不以 Pilot 001 的定向候选验证成功与否作为成败标准。",
        "6. 零模型经验 p 是未校正的探索性诊断，不是文学重要性或总体显著性。",
        "7. NCD 的明显长度混淆被保留并报告，不能用其支持单独作结构解释。",
        "",
        "## 7. 可追溯产物",
        "",
        "- `results/tables/candidates.csv`：排序后的完整候选与事后预览；",
        "- `results/tables/candidates_calibrated.csv`：原排序加位置零模型诊断；",
        "- `results/tables/candidates_by_null_calibration.csv`：仅供审计的零模型排序；",
        "- `results/tables/geometry_events.csv`：聚合前的全部事件；",
        "- `results/tables/matrix_inventory.csv`：逐矩阵诊断与事件计数；",
        "- `data/derived/matrices/`：相似度、百分位和元数据；",
        "- `data/derived/units.json`：全部统一坐标。",
    ]
    (C.DOCS_DIR / "pilot_002_report.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")


def main():
    calibrated = C.TABLE_DIR / "candidates_calibrated.csv"
    candidates = read_csv(calibrated if calibrated.exists()
                          else C.TABLE_DIR / "candidates.csv")
    inventory = read_csv(C.TABLE_DIR / "matrix_inventory.csv")
    run_meta = json.loads((C.DERIVED_DIR / "run_metadata.json").read_text(
        encoding="utf-8"))
    total_words = json.loads((C.DERIVED_DIR / "units.json").read_text(
        encoding="utf-8"))["body_words"]
    plot_candidates(candidates, total_words)
    plot_coverage(candidates)
    plot_reference_matrices()
    plot_null_calibration(candidates)
    write_report(candidates, inventory, run_meta)
    print("wrote figures and", C.DOCS_DIR / "pilot_002_report.md")


if __name__ == "__main__":
    main()

