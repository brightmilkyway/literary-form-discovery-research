# Pilot 002 — Multi-scale structural persistence（v2，已完成）

这是 *The Dead* 的无文学标签结构探索。研究规则见 [`protocol.md`](protocol.md)。

主流程将完成：统一原文坐标、多尺度与切分相位、自相似矩阵几何提取、五种表示
的跨运行聚合，以及可追溯的表格、图和报告。

目录约定：

```text
pilot_002/
├─ protocol.md
├─ README.md
├─ requirements.txt
├─ code/
├─ data/
│  ├─ source/
│  └─ derived/
├─ results/
│  ├─ tables/
│  └─ figures/
└─ docs/
```

## 已完成的实现

- 7 个尺度、22 种尺度—相位切分；
- 统一字符与词坐标，全部通过原文回环检查；
- 字符 TF-IDF、词 TF-IDF、双向平均 NCD、外部拟合白化 MiniLM、外部拟合白化 BGE；
- `point / stripe / block / valley` 四种矩阵几何；
- 跨尺度、相位与表示的持久性聚合；
- 100 次位置随机化零模型，校准空间占用造成的偶然共识；
- 二维 cubical persistent homology（`H0/H1`）及每矩阵 100 次顺序置换零模型；
- 表格、图和不作文学命名的结果报告。

完整运行得到 110 张矩阵、44,653 个未聚合事件和 2,144 个坐标候选。原始
持久性存在明显的空间占用基线，因此必须同时阅读零模型校准表，不能只看
`persistence` 排名。

v2 严格区分两种“持久性”：v1 的 `persistence` 是候选跨观察装置复现率；v2 的
topological persistence 是阈值 filtration 中同调类的寿命。严格拓扑的稳健结果
是：`H0` 总持久度在双侧 BH 后有 **94/110** 张矩阵低于顺序置换零模型，强
`H0` 特征数有 **91/110** 张通过；单个最长 `H0` 为 **0/110**，说明差异来自
整体较少碎片化而非一座异常巨峰。`H1` 有 19/110 张仅在未校正 p 下过线，BH 后
**0/110**，因此不声称发现叙事环。

## 运行

在仓库根目录执行：

```powershell
python -m pytest pilot_002/tests -q
python pilot_002/code/run_pipeline.py
python pilot_002/code/null_calibration.py --iterations 100
python pilot_002/code/make_report.py
python pilot_002/code/topology.py --null-iterations 100
python pilot_002/code/topology_report.py
python pilot_002/code/doc_sync_audit.py
```

神经模型强制从 Pilot 001 已使用的本地 Hugging Face 缓存载入；完整首次运行较慢，
各尺度嵌入与矩阵都会缓存，后续合并运行不会重新推理。

## 关键产物

- `docs/pilot_002_report.md`：无标签结果报告；
- `results/tables/candidates.csv`：冻结规则的原始排序；
- `results/tables/candidates_calibrated.csv`：原排序加零模型诊断；
- `results/tables/candidates_by_null_calibration.csv`：零模型审计视图；
- `results/tables/topology_matrix_summary.csv`：110 张矩阵的 `H0/H1` 与双侧零模型校准；
- `results/tables/topology_feature_clusters.csv`：强 birth cell 的跨矩阵坐标聚合；
- `data/derived/topology_summary.txt`：严格拓扑的可读摘要；
- `results/tables/geometry_events.csv`：全部未聚合事件；
- `data/derived/units.json`：统一坐标；
- `data/derived/matrices/`：110 张矩阵及逐矩阵元数据。


