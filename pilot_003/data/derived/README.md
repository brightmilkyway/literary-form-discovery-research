# Derived-data inclusion policy

PR 中保留不可确定重建的 DeepSeek 精简原始响应，以及所有小型摘要、元数据、表格
和 PNG 图。以下大体积或重复产物不入库：

| 排除产物 | 原因 | 再生命令 |
|---|---|---|
| `checkpoints.json` | 与 CSV 重复 | `python pilot_003/code/run_pipeline.py` |
| `interpretation_requests.jsonl` | 约 1.9 MB 的重复前缀包 | `python pilot_003/code/interpretive.py prepare` |
| `interpretation_runs_r*.jsonl` | 输入与精简响应的重复合并副本 | `python pilot_003/code/interpretive.py validate pilot_003/data/derived/deepseek_responses_rNN.jsonl` |
| `results/tables/token_surprisal.csv` | 22,574 行逐 token 明细 | `python pilot_003/code/predictive.py` |
| `results/figures/*.svg` | PNG 的大型矢量副本 | 运行相应分析脚本 |

`deepseek_responses_r01.jsonl` 与 `deepseek_responses_r02.jsonl` 入库：它们只含模型
输出、请求哈希和提供方元数据，不重复完整前缀，且 API 生成不能确定性重建。
