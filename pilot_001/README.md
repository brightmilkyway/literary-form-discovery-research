# Pilot 001 — *The Dead* (James Joyce): open-ended computational structure probe

探索性预实验（EDA），**不是**假设验证研究。本目录是 `Pilot_001_The_Dead.md`
所规定交付物的实现与产物。三种归纳偏置不同的现成方法在不预先标注主题、象征、
叙事转折或"涌现结构"的前提下，考察正文内部是否存在值得进一步文学解释的组织关系。

## 目录

```
pilot_001/
├─ protocol.md                本实验的原始方案（研究问题、方法、交付要求、判读边界）
├─ README.md
├─ requirements.txt
├─ code/                      可复现脚本（全部确定性，seed=42）
│  ├─ common.py               读入、段落/片段切分、可回溯字符偏移、随机种子
│  ├─ texttiling.py           TextTiling 忠实实现（Hearst 1997）
│  ├─ validate_against_nltk.py 与 NLTK 参考实现逐位对照（含 PASS/FAIL 断言）
│  ├─ ncd.py                  NCD（zlib / bz2 / lzma）
│  ├─ sbert_semantic.py       Sentence-BERT 编码与相似度
│  ├─ sbert_sensitivity.py    100 词分块的 SBERT 敏感性 + 截断量化
│  ├─ baselines.py            TF-IDF / Jaccard / 长度 对照
│  ├─ plots.py                全部图表
│  ├─ make_report_figures.py  三法热图对照 + 敏感性汇总图
│  ├─ run_all.py              主流程
│  ├─ analyze_findings.py     汇总摘要、中心性、hub 校正残差、边界↔相似度对照
│  ├─ alternative_explanations.py 长度/引语密度等廉价解释的检验
│  ├─ fix_representations.py  ★修复 NCD 长度混淆与 SBERT 截断
│  ├─ verify_fixes.py         ★跨全部表示复核每条结论的稳健性
│  ├─ adjudicate_snow_pair.py ★(2,73) 雪对的裁决实验（四项检验）
│  ├─ model_agreement_check.py★第二个编码器 + 跨模型控制
│  ├─ content_ablation.py     ★实词消融（记为失败的对照，产出"形式基线"）
│  ├─ mutual_rank_check.py    ★免校正的互为近邻排名检验
│  ├─ snow_source_experiment.py★结尾段外围性的来源（受控替换 + 移植）
│  ├─ song_cluster_analysis.py★歌簇细读（歌词消融、音乐词密度、子簇结构）
│  ├─ name_ablation_control.py★人物名消融对照（记为失败的对照）
│  ├─ weather_set_test.py     ★天气集置换检验（六种表示，150 词尺度）
│  ├─ scale_adjudication.py   ★尺度裁决（句子/段落 × 三种零假设）
│  ├─ verbatim_echo_test.py   ★逐字呼应的句级微观检验
│  ├─ scale_recheck.py        ★句子尺度复核（TextTiling 正交性 / NCD 混淆）
│  ├─ sentences.py            ★经检验的分句器（先按段落切，再段内 punkt，保留弯引号）
│  ├─ validate_sentences.py   ★分句器验证（欠切分/过切分/偏移回环）
│  ├─ spotcheck_sentences.py  ★分句器人工抽查（随机样本 + 上下文 + 可疑模式）
│  ├─ diagnose_sentences.py   分句器问题的诊断（保留为记录）
│  ├─ diagnose_punkt_oversplit.py 弯引号 vs 直引号的对照（保留为记录）
│  ├─ ncd_reviewer_checks.py  ★NCD 拼接方向敏感性 + 等长分箱检验
│  ├─ form_vs_content.py      ★形式 vs 内容（删除而非替换的修正设计）
│  ├─ echo_matched_control.py ★雪呼应的匹配对照（距离/句长/词汇重叠）
│  └─ lookup_unit.py          按片段号打印完整原文与段号（用于核对引文）
├─ data/
│  ├─ source/the_dead.txt     原文（Project Gutenberg 英文正文，UTF-8）
│  └─ derived/                中间产物（矩阵 .npy、参数与数值 .json、可读摘要 .txt）
├─ results/
│  ├─ figures/                图（PNG + SVG）
│  └─ tables/                 可追溯的 CSV（边界、远距离片段对）
└─ docs/
   ├─ candidate_findings.md   交付物 B：候选现象清单
   └─ pilot_001_report.md     交付物 C：文学研究报告
```

### 关键可读产物（建议先看这些）

| 文件 | 内容 |
|---|---|
| `docs/candidate_findings.md` | 5 项候选现象，每项含位置/指标/替代解释/待追问问题 |
| `docs/pilot_001_report.md` | 文学研究报告（四问） |
| `data/derived/findings_digest.txt` | 方法一致性、混淆、top 对、最强边界、hub 校正残差 |
| `data/derived/alternative_explanations.txt` | 长度/引语密度等廉价解释的检验结果 |
| `data/derived/sbert_sensitivity.txt` | 100 词分块下结论是否存活 + 截断量化 |
| `data/derived/representation_fixes.txt` | **两项修复的前后对比与修复后结论** |
| `data/derived/robustness_check.txt` | **六种表示下每条候选的稳健性汇总** |
| `data/derived/adjudication_verdict.txt` | **★ (2,73) 雪对的最终判词与全部证据** |
| `data/derived/mutual_rank_check.txt` | **★ 免校正的互为近邻排名（两模型）** |
| `data/derived/model_agreement_check.txt` | **★ 跨模型控制与消融对照** |
| `data/derived/round4_summary.txt` | **★ v4 受控替换实验总结** |
| `data/derived/snow_source_experiment.txt` | 结尾段外围性的来源（消融/移植/词序） |
| `data/derived/song_cluster_analysis.txt` | 歌簇细读（歌词消融、子簇结构） |
| `data/derived/weather_set_test.txt` | 天气集置换检验（六种表示，150 词尺度） |
| `data/derived/scale_verdict.txt` | **★ 尺度判词：v4 的"雪不可见"结论已撤回** |
| `data/derived/scale_adjudication.txt` | **★ 句子/段落尺度 × 三种零假设设计** |
| `data/derived/verbatim_echo_test.txt` | **★ 逐字呼应的句级微观检验（最强远距离配对）** |
| `data/derived/scale_recheck_verdict.txt` | **★ 尺度复核判词（正交性稳健 / NCD 更严重 / 模板检测）** |
| `data/derived/scale_recheck.txt` | **★ 句子尺度的 TextTiling 与 NCD 复核数据** |
| `data/derived/reviewer_response.txt` | **★ 审稿回应总表（第一轮：四项问题与处置）** |
| `data/derived/reviewer_response_2.txt` | **★ 审稿回应（第二轮：证据表述的限定）** |
| `data/derived/echo_matched_control.txt` | **★ 雪呼应的匹配对照（距离 + 句长 + 词汇重叠）** |
| `data/derived/sentence_validation.txt` | **★ 分句器验证（764 → 1023；在已定义的自动检查中未发现欠/过切分）** |
| `data/derived/sentence_spotcheck.txt` | **★ 分句器人工抽查（随机样本、上下文、可疑模式）** |
| `data/derived/ncd_reviewer_checks.txt` | **★ NCD 拼接方向 + 等长分箱检验** |
| `data/derived/form_vs_content.txt` | **★ 形式 vs 内容（修正设计）** |
| `data/derived/content_ablation.txt` | 实词消融（失败的对照 + 形式基线） |
| `data/derived/name_ablation_control.txt` | 人物名消融对照（失败） |
| `results/figures/fig_fix_before_after.png` | **修复前后对照（SBERT 截断 / NCD 长度混淆）** |
| `results/figures/fig_three_heatmaps.png` | NCD / SBERT / TF-IDF 三联热图 |
| `results/figures/fig_sensitivity_summary.png` | 长度混淆 / 参数敏感性 / 距离衰减 |
| `results/figures/fig_texttiling_sweep.png` | TextTiling 六种参数下的曲线与边界 |
| `results/tables/texttiling_strongest_boundaries.csv` | 最强边界 + 原文引文 |
| `results/tables/ncd_top_nonadjacent_pairs.csv` | NCD 最近的非相邻片段对 |
| `results/tables/ncd_residual_top_pairs.csv` | **长度残差校正后的 NCD 最近片段对** |
| `results/tables/sbert_top_nonadjacent_pairs.csv` | SBERT 最相似的非相邻片段对 |

## 已应用的两项表示层修复（v2）与一轮审稿修正（v7）

### v7 审稿修正——三项撤回、一处加强

一位审稿人发现**分句器是错的**：`nltk.sent_tokenize` 直接调用给出 764 个"句子"，
其中 **152 个含内部换行**（是多句块）——原文在段末写 `.”`（句点 + **弯引号**），
punkt 不把它当句末。**已修复为"先按段落切、再在段内用 punkt（保留弯引号）"，
得 1023 个句子**；**在已定义的自动检查中未发现欠切分、过切分或偏移失配**
（措辞限定见「分句器：切分正确性的证据边界」）；句尺度全部重算。

| 项 | 处置 |
|---|---|
| **分句器** | **修复**：1023 个句子（原 764 个中 20% 不是句子）。雪的呼应**加强**：cos 0.581 → **0.726** |
| **NCD 对称矩阵** | **承认为假设**：正/反拼接 top-10 只重合 **6/10**，29.6% 的对变化 > 0.01 |
| **"NCD = 模板检测器"** | **撤回**：等长分箱内框架效应只在 9/17 箱成立 |
| **"0.83 形式基线"** | **撤回**：占位符假象。改用"删除而非替换"后结论**反向**——相似度由**内容**驱动（r = +0.78），形式相似度高但不变（0.712） |
| **"句子尺度混淆更严重"** | **撤回**：修正后 0.661，与 150 词的 0.689 **相当** |
| **母题集合检验** | **重述为不具区分力**：音乐/宗教/死亡/饮食集同样达 ~100%，不能支持"雪特殊" |

### v8 审稿修正（第二轮）——证据表述的限定

| 项 | 处置 |
|---|---|
| **雪的呼应的随机对照** | **承认不够有力**：≤200 个有效样本、只按距离筛选、未匹配句长与词汇重叠、未校正选择效应。表述改为"计算上极为突出的远距离文本呼应"，**不是**"统计上已证明异常" |
| **分句器"零错误"** | **限定措辞**为"在已定义的自动检查中未发现错误"，**不是**"已证明 1023 个边界全部正确" |
| **文档同步** | 报告、候选清单、README、protocol 全部统一到当前版本；被撤回的结论**只以撤回记录形式出现**，不与现行结论并列 |
| **研究立场** | 明确区分**开放式发现 / 针对性复核 / 解释**三阶段；"全程未使用主题标签"的表述限定为**仅指第一阶段** |
| **根 README 的方法学条目** | 由"后续实验的硬性要求"改为"**Pilot 001 提出的工作原则**"，待更多作品检验 |

完整回应见 `data/derived/reviewer_response.txt`（第一轮）与
`data/derived/reviewer_response_2.txt`（第二轮）。

### v2 表示层修复

| 缺陷 | 修复 | 效果 |
|---|---|---|
| **NCD 长度混淆**：距离与"压缩长度差"相关 0.689（解释 47.5% 方差） | 等词数单位（n=105）＋ 长度残差校正 | r → 0.179 → 0.000；修复后 NCD↔TF-IDF 由 0.439 升至 **0.688**（长度是抑制变量） |
| **SBERT 截断**：25/78（32%）片段超 256 token 被静默截断 | 段落感知池化（≤130 词子块，均值池化） | 截断 **0/184**；池化与截断版 top-20 仅重叠 11/20 |

**修复后的结论变化**：
* "歌"簇**加强**，并成为**唯一**跨方法一致（修复前 SBERT∩NCD top-20 = 0 对，修复后 = 1 对）。
* 修道院一段的语义孤立**加强**（截断版、池化版均排名 1/78）。
* "雪"的首尾呼应**进一步被证伪**（hub 校正名次 91 → 164/3003），已从候选清单撤下。
* TextTiling 边界与其他表示正交的负结果**在五种表示下均成立**，不是表示缺陷所致。

## 跨模型裁决（v3）

v2 留下一个未决问题：**(2,73) 的"雪"首尾呼应**到底是真实但被 hub 校正掩盖，
还是假象？用四类检验 + 第二个编码器（`BAAI/bge-small-en-v1.5`）裁决。

**判词：单编码器产物，撤回。** 完整证据见 `data/derived/adjudication_verdict.txt`。

| 检验 | (2,73) 雪 | 对照：(30,39) / (55,58) / (12,16) / (71,74) |
|---|---|---|
| MiniLM 长距离限定（gap≥40，741 对） | raw **5/741**，hub **20/741** | — |
| 留一法 hub 校正 | 17/741（非自膨胀） | — |
| 跨分块复现（80/100 词） | 96–98 百分位 | — |
| **跨模型（bge）** | raw **457/741**，hub **282/741** | hub **1 / 4 / 1 / 1**（全部复现） |
| **互为近邻排名（免校正）** | **5（MiniLM）/ 55（bge）** | **11/2、1/2、5/4、2/9** |

bge **复现了其他每一对**，却**单独否决**了 (2,73)——所以这不是"模型更差"。

**裁决副产物（方法学，影响所有后续研究）**
1. **hub 校正有对称缺陷**：它夸大"外围×外围"对。(30,39) 的两个单位外围名次是
   5 和 1，其 MiniLM hub 校正"第 1 名"部分是校正造成的；真正可信的是 bge 的
   互为近邻第 2 名。→ hub 校正列表必须附免校正检验。
2. **MiniLM 余弦存在 ≈0.83 的形式基线**：把全部实词替换为占位符后，所有片段对
   余弦都升到 ~0.83。观察值 0.44–0.76 全部低于基线，**绝对余弦值不可读作语义
   相似度**。
3. **一处自我纠正**：v2 称修道院孤立"跨设置稳健"，实际只跨了三个 MiniLM 设置；
   bge 下它掉到 21/78，已降级为单模型观察。
4. **一个失败的对照被保留**：实词消融的 drop 指标是循环的（drop ≈ 0.83 − cos），
   无法裁决单个片段对。

**最终存活的候选**：歌簇 (55,58)（互为近邻 1/2，两模型）、结尾雪段的最外围性
（2/78 与 1/78）、以及 (71,74)、(12,16)、(69,72)、(30,39)。

## 受控替换实验（v4）

对两个存活候选问"它们到底靠什么成立"。结论见 `data/derived/round4_summary.txt`。

**歌簇被改写**：最稳健的簇**不是"歌"这个文本对象**，而是
**"听见"（54–58）与"讲述"（69–73）两个场景 + 桥接句 unit 69**
（全篇唯一同时说出歌与人名的句子）。证据：逐字删掉 unit 55 的**歌词原文**后
cos(55,58) 变化 **0.000**（随机删除对照的第 100 百分位）；删掉歌名 +0.010；
音乐词密度也不定义该簇（unit 54 密度第 3 却不在簇里，unit 56 密度 0.00 却在簇里）。

**结尾段外围性不是词汇造成的**：删掉全部冬季/西行/死亡词汇（37 个词型）后，
其中心性落在**同等规模随机消融的第 62 百分位**（两模型相同）；
把该词表移植进更中心的段落也不能稳定使其变外围。相反，**打乱句内词序**使它
在两个模型下都**变中心**（bge 外围名次 1 → 11）——外围性部分来自**句法/形式**。

**强负结果**：按内容定义的天气集（13 段；核心 6 段）在**六种表示下都不成集合**
（置换检验百分位 12–80%，bge 甚至低于随机）；全篇唯一可逐字核对的呼应
"the snow is general all over Ireland"（unit 56）↔"snow was general all over
Ireland"（unit 77）**六种方法都没抓到**。

**两个失败的对照被如实保留**：内容词消融的 drop 指标是循环的；人物名消融看似
显著，但 82% 的随机片段对也有同样效应。

## 尺度裁决（v5）——上一节的强负结果已撤回

**v4 的"雪不可见"结论是 150 词分析单元的稀释效应造成的假象。** 见
`data/derived/scale_verdict.txt`。

降到**句子尺度**后：

| 模型 | cos | 互为近邻 | 200 个随机远距离对的**最大值** |
|---|---|---|---|
| MiniLM | **0.726** | **1 / 1** | 0.437 |
| bge | **0.757** | **1 / 1** | 0.720 |

* **句 728**（Mary Jane 引报纸）："…the snow **is** general all over Ireland."
* **句 1018**（Gabriel，结尾）："Yes, the newspapers **were** right: snow **was**
  general all over Ireland."

相隔 **290 句**，**两个模型都把这两句判为彼此在全篇 1023 句中的第一近邻**。

**⚠️ 关于证据强度（v8 修正，见审稿意见 1）**：上表的随机对照**不够有力**——
它只有 ≤200 个有效样本、只按距离（≥200 句）筛选，**没有匹配目标的距离（290）、
句长或词汇重叠**；而且该配对是**探索后选出**的，与一小批随机配对的最大值比较
**没有校正选择效应**。因此现在的表述是：
**"它是一个计算上极为突出的远距离文本呼应"**，
**不是**"统计上已证明异常"，也不是"全篇的结构枢纽"。
`data/derived/echo_matched_control.txt` 给出了匹配对照的结果。

**尺度检验的三种零假设设计，两种失败**：置换零假设在句子尺度**饱和**
（随机单词集也接近 100%）；规模匹配的单词集零假设**偏向场景局部词**
（把"离散度"误当成"不抱团"）。只有**形状匹配零假设**（同游程数、同游程长度、
随机放置）成立——但**它同样不具区分力**（见下）。

**结论**：**分析单元的尺度比方法选择更能决定结论。** 150 词单元有系统性盲区——
它会摧毁由单个句子承载的信号。**这是 Pilot 001 提出的工作原则**，
是否适用于其他作品需要另行检验。

## 尺度复核（v6，v7 更正）——另外两条 150 词结论的命运

见 `data/derived/scale_recheck_verdict.txt`（文件顶部有 v7 更正标注）。

| 结论 | 150 词尺度 | 句子尺度（1023 句） | 判词 |
|---|---|---|---|
| **TextTiling 正交性** | 百分位 47–56% | **41–53%**（词基 + 句基两种变体 × 4 个表示） | **尺度稳健，保留** |
| **NCD 长度混淆** | r = +0.689（解释 47.5%） | **r = +0.661（解释 43.7%）** | **与 150 词相当**；v6 的"更严重"已撤回 |
| ~~**NCD 检测句式模板**~~ | — | 等长分箱内框架效应只在 **9/17 箱**成立 | **v7 撤回** |

**关键对照**：那对逐字内容呼应（句 728 ↔ 1018）——
MiniLM / bge / TF-IDF 都判 **1/1023**，Jaccard 2/1023；
**原始 NCD 抓不到**（名次 37,076 / 420,903，后 8.8%），
**但长度残差校正后进入前 0.5%**（2,207 / 420,903）。
即：**正是长度混淆把这对呼应藏了起来**，而不是 NCD 看不见内容。

## 数据与切分

* 原文：`data/source/the_dead.txt`（Project Gutenberg 英文正文，UTF-8）。
  脚本按 UTF-8-sig 读入，只规范化换行，其余字符不动；去掉开头的 `THE DEAD`
  标题行，记下标题偏移以便回溯。正文 **85,683 字符 / 455 段 / 15,732 词**。
  （`common.py` 也会在上级目录查找 `the_dead.txt`，以便在原始工作目录中直接运行。）
* **分析片段**（analysis units）不预设为"真实的文学单位"：
  * `group150`：把连续段落合并到 ≥150 词，**不切段**（78 个片段）——跨方法比较的主单位。
  * `group250`：同上，250 词（51 个片段）——分块尺度敏感性。
  * `window150/75`：在词流上 150 词、步长 75 的滑动窗口（210 个片段）。
  * `paragraph`：455 个自然段。
  * `sentence`：**1023 个句子**，由 `sentences.py` 切分（先按段落切，再段内 punkt，
    保留原文弯引号）。**切分精度见下一节。**
  * 每个片段都保留 `start/end` 字符偏移与首末段号，任何结论都能跳回原文。
* **顺序扰动不做距离矩阵对照**：打乱片段顺序不会改变固定片段间的两两相似度，
  因此该操作不能作为有效对照（文档 §3 已明确）。

### 分句器：切分正确性的证据边界

`sentences.py` 的切分精度由两组自动检查加人工抽查支撑
（`validate_sentences.py`、`spotcheck_sentences.py`，输出在
`data/derived/sentence_validation.txt` 与 `sentence_spotcheck.txt`）：

1. **欠切分**：单元内是否仍含"终止标点 + 换行"；
2. **过切分**：单元是否以**预设的**常见对话标签（said/asked/cried/…）开头；
3. **偏移回环**：`body[start:end]` 是否逐字等于单元文本；
4. 随机抽取 30 个单元打印全文供人工阅读，并列出全部可疑模式。

**措辞限定（v8，见审稿意见 2）**：这三项检查的结果是
**"在已定义的自动检查中未发现错误"**，
**不是**"已证明 1023 个边界全部正确"。
过切分检查只覆盖一组预先列出的开头模式；偏移回环只证明切分结果能对应回原文，
不证明边界本身正确。
若要把分句精度作为方法学结论发表，需另行构建覆盖不同引号、对话标签与标点情形的
**人工标注样本**并报告一致性。

## 三种方法与基线

| 模块 | 方法 | 关键设置 |
|---|---|---|
| `texttiling.py` | TextTiling (Hearst 1997) | w=20, k=6 主设置；扫描 w∈{10,20,40}、k∈{4,6,10}、平滑开/关、HC/LC 阈值、NLTK/word 两种分词 |
| `ncd.py` | NCD (Cilibrasi & Vitányi 2005) | zlib/bz2/lzma；按阅读顺序拼接；报告长度混淆 |
| `sbert_semantic.py` | Sentence-BERT (Reimers & Gurevych 2019) | `all-MiniLM-L6-v2`，256 token 上限，记录截断风险 |
| `baselines.py` | TF-IDF 余弦、词集 Jaccard、纯长度相似 | 用于判断复杂方法是否带来额外观察 |

**TextTiling 忠实性**：`validate_against_nltk.py` 显示本实现与 NLTK
`TextTilingTokenizer` 在本文上 **gap 分数最大差 1e-16、边界集合完全相同**
（停止词表也逐项一致）。

## 运行

```bash
cd code
python run_all.py                # 全流程（texttiling + ncd + sbert + baselines）
python run_all.py --only ncd     # 单阶段：texttiling|ncd|sbert|baselines
python run_all.py --no-sbert     # 跳过神经编码器
python validate_against_nltk.py  # 与 NLTK 参考实现逐位对照（断言）
python analyze_findings.py       # 汇总摘要 + 中心性 + hub 校正残差
python alternative_explanations.py  # 廉价替代解释检验
python sbert_sensitivity.py      # 100 词分块敏感性 + 截断量化
python fix_representations.py    # 修复 NCD 长度混淆与 SBERT 截断
python verify_fixes.py           # 跨全部表示复核结论稳健性
python adjudicate_snow_pair.py   # (2,73) 雪对的裁决实验（四项检验）
python model_agreement_check.py  # 第二个编码器 + 跨模型控制
python content_ablation.py       # 实词消融（失败的对照 → 形式基线）
python mutual_rank_check.py      # 免校正的互为近邻排名检验
python snow_source_experiment.py # 结尾段外围性的来源
python song_cluster_analysis.py  # 歌簇细读
python name_ablation_control.py  # 人物名消融对照（失败）
python weather_set_test.py       # 天气集置换检验（150 词尺度）
python scale_adjudication.py     # 尺度裁决（句子/段落 × 三种零假设）
python verbatim_echo_test.py     # 逐字呼应的句级微观检验
python scale_recheck.py          # 句子尺度复核 TextTiling 正交性与 NCD 混淆
python sentences.py              # （模块）经检验的分句器
python validate_sentences.py     # 分句器验证（应输出 units=1023）
python spotcheck_sentences.py    # 分句器人工抽查（应输出 lowercase_start=0）
python ncd_reviewer_checks.py    # NCD 拼接方向 + 等长分箱检验
python form_vs_content.py        # 形式 vs 内容（修正设计）
python echo_matched_control.py   # 雪呼应的匹配对照
python make_report_figures.py    # 报告用图
python lookup_unit.py 2 73 39    # 打印指定片段的完整原文与段号
```

环境见 `requirements.txt`。**随机种子固定为 42**（`common.set_seed`），并写入
`data/derived/results_summary.json` 的 `meta.seed`。流程本身无采样、无训练，
确定性由构造保证；种子用于神经编码器及未来扩展。

## 判读边界（与文档 §5 一致）

聚类、分段或高相似度 **≠** 自动发现文学结构；它们首先是待解释的计算现象。
出版文本中的位置关系 **≠** 创作历史中的时间关系。**没有正面发现也是有效的预实验结果。**
