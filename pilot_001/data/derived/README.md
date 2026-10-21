# pilot_001 — 排除的产物与再生方式

本目录保存 Pilot 001 的**可核查结果**（可读摘要、结构化数值、图表）。
为保持仓库精简、并使 pull request 可审阅，两类**体积大且可一条命令再生**的
产物不入库：

| 排除的产物 | 原因 | 再生方式 |
|---|---|---|
| `results/figures/*.svg` | 与同名 `.png` 内容重复（矢量版），单个文件可达 2 万行 | `python code/make_report_figures.py` 与 `python code/run_all.py --only texttiling` |
| `data/derived/texttiling_full.json` | TextTiling 的逐间隙完整曲线（5,270 行）；其**结论**已完整保存在 `results/tables/texttiling_boundaries.csv`、`texttiling_strongest_boundaries.csv` 与 `results_summary.json` 的 `texttiling.sweep` 中 | `python code/run_all.py --only texttiling`（约 6 秒） |

因此，若要运行 `code/analyze_findings.py` 或 `code/verify_fixes.py`
（二者会读取 `texttiling_full.json`），请先执行：

```bash
cd code
python run_all.py --only texttiling     # 重建 texttiling_full.json
python analyze_findings.py
python verify_fixes.py
```

其余全部产物均随仓库提供，包括：

* `*.txt` — 人可读的结果摘要与判词（文档直接引用这些文件）
* `*.json` — 结构化数值结果（`results_summary.json` 为总表）
* `*.npy` — 各方法的相似度/距离矩阵（`numpy.load` 可直接读取，
  是全部图表的数值来源）
* `units.json` — 每个分析单位的字符偏移与段号（位置可回溯性的依据）

**注意**：`*.npy` 与 `units.json` 也可由 `python code/run_all.py` 完整重建，
保留它们是为了让审阅者无需重新运行即可直接检验数值。
