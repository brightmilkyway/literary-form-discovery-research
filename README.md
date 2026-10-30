# Literary Form Discovery

**Open, reproducible experiments on how computational responses vary across literary form.**

[中文说明](README.zh-CN.md) · [Data sources](DATA_SOURCES.md) · [Citation](CITATION.cff) · [Licensing](#licensing)

## Research vision

This repository studies literary texts without treating an algorithmic output as a literary
structure by default. Each pilot begins with position-preserving, label-free measurements and
then asks a narrower question: which observations survive changes of scale, representation,
segmentation, control design, or repeated model execution?

The goal is not to recover authorial intention, prove emergence, or equate an AI system with a
human reader. The goal is to produce auditable empirical objects that can later enter literary
interpretation without hiding their computational and experimental conditions.

## Experimental pilots

| Pilot | Text | Main question | Status | Report |
|---|---|---|---|---|
| 001 | James Joyce, *The Dead* | Which local and non-local relations appear under segmentation, compression, and embedding methods? | Complete, v18 | [Report](pilot_001/docs/pilot_001_report.md) · [Findings](pilot_001/docs/candidate_findings.md) |
| 002 | James Joyce, *The Dead* | Which matrix geometries persist across 7 scales, 22 scale/phase partitions, and 5 representations? | Complete, v1 | [Report](pilot_002/docs/pilot_002_report.md) · [Findings](pilot_002/docs/candidate_findings.md) |
| 003 | James Joyce, *The Dead* | How do predictive, representational, and interpretive responses change over 40 prefix-ordered checkpoints? | Complete, v1 | [Report](pilot_003/report.md) · [Findings](pilot_003/docs/candidate_findings.md) |

## Findings and limitations

The pilots support limited computational observations, not a single recovered literary map.

- Scale can change a conclusion more than the choice between two similar encoders.
- Surface-form observers may agree strongly while sharing the same lexical bias.
- Cumulative-prefix representations can become mechanically smooth because adjacent inputs share
  almost all prior text.
- In Pilot 003, repeated temperature-zero interpretive responses varied almost as much as adjacent
  checkpoints; isolated interpretation peaks are therefore not stable evidence.
- Cross-dimensional agreement between surprisal, representation distance, and interpretation
  distance was weak under rank correlation and does not support one common latent structure.

All findings remain specific to one literary work and the recorded experimental settings.

## Failure cases and revisions

Failures and withdrawals are first-class results in this project.

- Pilot 001's first sentence segmentation silently merged quoted paragraphs; the affected analysis
  was corrected and the failure retained in the report.
- A claimed “form baseline” produced by replacing content words with one shared placeholder was
  withdrawn: the placeholder itself created artificial similarity.
- Several random-control designs saturated or failed to match length and lexical overlap.
- Pilot 002's raw cross-observer persistence required a spatial-occupancy null calibration.
- Pilot 003's JSON schema succeeded, but only 12 of 40 outputs per replicate followed the frozen
  120–180-word limit.

The repository rule is simple: a corrected or withdrawn claim remains visible as history and is
never silently overwritten.

## Methodological principles

1. Preserve original text order and trace every unit back to character or word coordinates.
2. Separate open discovery, targeted review, and literary interpretation.
3. Freeze primary metrics before reading their results.
4. Treat preprocessing and controls as components that require their own validation.
5. Match wording strength to evidence: “computationally prominent” is not “statistically unusual.”
6. Keep negative results, failed controls, and retractions in the evidence record.
7. Never interpret model surprisal directly as human surprise, emotion, or reading difficulty.

The complete collaboration and evidence rules are in [AGENT.md](AGENT.md).

## Reproducibility

Each pilot pins its dependencies, records source hashes and random seeds, and documents which large
artifacts are regenerated rather than committed.

```bash
# Pilot 002
python -m pytest pilot_002/tests -q
python pilot_002/code/run_pipeline.py
python pilot_002/code/null_calibration.py --iterations 100
python pilot_002/code/make_report.py
python pilot_002/code/doc_sync_audit.py

# Pilot 003 local components
python -m pytest pilot_003/tests -q
python pilot_003/code/run_pipeline.py
python pilot_003/code/interpretive.py validate pilot_003/data/derived/deepseek_responses_r01.jsonl
python pilot_003/code/interpretive.py validate pilot_003/data/derived/deepseek_responses_r02.jsonl
python pilot_003/code/analyze_interpretations.py
python pilot_003/code/predictive.py
python pilot_003/code/cross_dimension.py
python pilot_003/code/doc_sync_audit.py
```

Pilot 003's DeepSeek calls are optional and require `DEEPSEEK_API_KEY`; the retained compact raw
responses allow the published summaries to be audited without repeating paid API calls. See each
pilot README for the complete workflow and model requirements.

## Data and provenance

The literary source is derived from Project Gutenberg eBook #2814, *Dubliners*. Exact repository
paths, SHA-256 fingerprints, extraction notes, upstream dates, and jurisdictional caveats are in
[DATA_SOURCES.md](DATA_SOURCES.md). Third-party literary text is not relicensed under this
repository's code or research-content licenses.

## Repository layout

```text
.
├─ README.md / README.zh-CN.md
├─ DATA_SOURCES.md
├─ AGENT.md
├─ pilot_001/
├─ pilot_002/
└─ pilot_003/
```

## Citation

Use the metadata in [CITATION.cff](CITATION.cff). Until an archived DOI is available, cite the
repository and the specific pilot report, version, and commit or release tag used in your work.

## Licensing

- Source code is licensed under the [MIT License](LICENSE).
- Original research prose, figures, and result tables are licensed under
  [CC BY 4.0](LICENSE-CONTENT.md), unless a file says otherwise.
- Literary source texts and raw third-party model outputs are excluded from those grants; see
  [DATA_SOURCES.md](DATA_SOURCES.md) and [LICENSE-CONTENT.md](LICENSE-CONTENT.md).

## Security and responsible disclosure

Do not commit API keys or private datasets. Report a suspected exposed credential privately as
described in [SECURITY.md](SECURITY.md).
