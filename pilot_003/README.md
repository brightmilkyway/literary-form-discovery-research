# Pilot 003 — Sequence response in *The Dead*

这是一个严格按阅读顺序、无事件标注的探索性实验。冻结设计见
[`protocol.md`](protocol.md)，实际结果与限制见 [`report.md`](report.md)，候选计算
观察及其降级理由见 [`docs/candidate_findings.md`](docs/candidate_findings.md)。

本轮已完成三类响应：本地固定哈希表示、DeepSeek 官方 API 的两轮 prefix-only
解释响应，以及 CPU 上的 DistilGPT2 逐 token surprisal。所有组件共享同一组 40 个
检查点；模型、版本、tokenizer、滑窗、原始输入输出和 token 用量均留档。

## 目录

```text
pilot_003/
├─ protocol.md
├─ README.md
├─ report.md
├─ config.json
├─ requirements.txt
├─ code/
├─ tests/
├─ data/derived/
└─ results/{tables,figures}/
```

## 复现

在仓库根目录使用已经满足依赖的环境：

```powershell
python -m pytest pilot_003/tests -q
python pilot_003/code/run_pipeline.py
python pilot_003/code/interpretive.py run-deepseek --replicate 1
python pilot_003/code/interpretive.py run-deepseek --replicate 2
python pilot_003/code/analyze_interpretations.py
python pilot_003/code/predictive.py
python pilot_003/code/cross_dimension.py
python pilot_003/code/doc_sync_audit.py
```

主流程输出：

- `results/tables/checkpoints.csv`：40 个固定检查点与前缀哈希；
- `results/tables/representation_trajectory.csv`：6 条相邻距离轨迹；
- `results/tables/representation_summary.csv`：基本统计与顺序置换基线；
- `results/tables/method_agreement.csv`：字符与词特征轨迹的 Pearson/Spearman 一致性；
- `results/figures/representation_trajectory.{png,svg}`：轨迹图；
- `results/tables/token_surprisal.csv`：22,574 个有上下文 token 的真实 surprisal；
- `results/tables/predictive_summary.csv`：检查点区间预测摘要；
- `results/tables/interpretation_trajectory.csv`：两轮相邻解释响应距离；
- `results/tables/interpretation_repeatability.csv`：同检查点跨轮距离；
- `results/tables/cross_dimension_correlations.csv`：三个维度的探索性对齐；
- `results/{representation,predictive}_metadata.json`：模型、环境与执行元数据；
- `data/derived/interpretation_requests.jsonl`：包含完整原始输入的请求包；
- `data/derived/interpretation_runs_r*.jsonl`：完整原始输入、输出与提供方元数据。

## 解释响应执行与断点续跑

先在 `config.json` 中明确记录选定模型，再重新生成请求：

```powershell
python pilot_003/code/interpretive.py prepare
```

官方 DeepSeek 运行器从 `DEEPSEEK_API_KEY` 读取密钥，不打印或保存密钥。每个检查点
完成后原子写入响应文件；重复命令会跳过已完成检查点。也可由其他提供方适配器逐行
返回如下 JSONL，不得改写原请求：

```json
{"checkpoint_id":"cp_001","model":"exact-model-id","prompt_sha256":"copy-from-request","prefix_sha256":"copy-from-request","output":"{\"current_understanding\":\"...\",\"salient_uncertainties\":\"...\",\"evidence_basis\":\"...\"}","provider_metadata":{}}
```

返回 40 行后执行：

```powershell
python pilot_003/code/interpretive.py validate path/to/responses.jsonl
```

校验通过才会生成 `data/derived/interpretation_runs.jsonl`。该文件同时保留原始输入、
输出、模型标识、参数和记录时间。API 成本、数据治理和模型选择须在执行前另行确认。

## 依赖与模型状态

锁定版本见 `requirements.txt`。DeepSeek 运行需要网络和 `DEEPSEEK_API_KEY`；预测
流程首次运行会从 Hugging Face 下载 `distilbert/distilgpt2`，之后使用本地缓存。
完整实测环境见各 metadata 文件。
