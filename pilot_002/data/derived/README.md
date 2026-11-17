# Pilot 002 derived artifacts

本目录只入库体积小、能直接审计本轮状态的 `run_metadata.json`。以下产物在本地
完整生成并用于报告，但因体积大或一条命令即可再生而不入库：

| 排除项 | 本地首次运行规模 | 再生方式 |
|---|---:|---|
| `embeddings/` | 58 个缓存，约 109 MB | `python pilot_002/code/run_pipeline.py` |
| `matrices/` | 110 张矩阵，约 54 MB | 同上 |
| `units.json` | 22 种切分的完整坐标，约 1.2 MB | 同上 |
| `results/tables/geometry_events.csv` | 44,653 行，约 6.5 MB | 同上 |
| `results/figures/*.svg` | 4 张可缩放图 | `python pilot_002/code/make_report.py` |

入库保留：协议与代码、PNG 图、矩阵清单、坐标验证表、原始候选表、零模型校准
候选表、报告以及本运行元数据。候选表保留了原文词坐标和短预览，可回溯到
Pilot 001 中同一份公共领域原文。

严格拓扑依赖本地排除的 `matrices/` 与 `units.json`，但其小型审计产物入库：
`topology_run_metadata.json`、`topology_summary.txt`、`topology_matrix_summary.csv`、
`topology_features.csv`、`topology_feature_clusters.csv`、`topology_aggregate.csv` 及
三张 `fig_topology_*.png`。其中完整再生约需 20 分钟（本机 128 landmarks、
110 × 100 次零模型）。

完整再生顺序（从仓库根目录）：

```powershell
python -m pytest pilot_002/tests -q
python pilot_002/code/run_pipeline.py
python pilot_002/code/null_calibration.py --iterations 100
python pilot_002/code/make_report.py
python pilot_002/code/topology.py --null-iterations 100
python pilot_002/code/topology_report.py
python pilot_002/code/doc_sync_audit.py
```

神经编码器强制使用本地 Hugging Face 缓存；首次运行前需按 `requirements.txt`
准备依赖及两套模型缓存。

