# Public-release audit — v1.0.0

Audit date: **2026-10-30**

## Scope

The audit was performed while the repository was private and covered `main`, all remote feature
branches then present, all reachable Git blobs, commit messages, author identities, retained data,
and the files planned for the public-readiness change.

## Results

| Check | Result |
|---|---|
| Pilot 003 PR | PR #12 squash-merged into `main` |
| Remote feature branches | Three stale snapshot branches audited and deleted; only `main` remained before the public-readiness branch was created |
| Secret-pattern history scan | 347 unique reachable blobs across all local and remote refs plus all commit messages scanned; 0 findings |
| GitHub secret scanning | Unavailable while the repository was private; must be rechecked after visibility changes |
| Largest historical blob | 1.58 MiB; no GitHub large-file limit risk |
| Author identities | `brightmilkyway@users.noreply.github.com` and `yuanyuhao2004@outlook.com`; repository owner explicitly accepted public attribution for both |
| API credentials | No key found; Pilot 003 reads `DEEPSEEK_API_KEY` from the environment and does not persist it |
| Source text | Project Gutenberg #2814 provenance, fingerprints, normalization, and jurisdiction caveat documented in `DATA_SOURCES.md` |
| Licenses | MIT for code; CC BY 4.0 for original research content; third-party text and raw provider outputs excluded |
| Pilot 003 report | Root `pilot_003/report.md` is the single authoritative report; duplicate removed |
| Automated tests | Pilot 002: 3 passed; Pilot 003: 5 passed; all three documentation sync gates passed |
| Public working-tree gate | Passed: required files, fingerprints, tracked text, and 80 retained responses verified |

## Scanner note

An official gitleaks v8.30.1 binary download was attempted. The first ZIP failed the published
SHA-256 check and was rejected without execution; retries then failed with GitHub release-asset TLS
handshake timeouts. The completed history scan therefore used a local read-only scanner over every
reachable blob and commit message for common AWS, GitHub, OpenAI-style, Slack, Google, private-key,
and high-entropy credential-assignment patterns. This limitation is recorded rather than presenting
the audit as stronger than it was.

## Reproducible working-tree gate

Run before a public release:

```bash
python tools/public_release_audit.py
python pilot_001/code/doc_sync_audit.py
python pilot_002/code/doc_sync_audit.py
python pilot_003/code/doc_sync_audit.py
```

After changing visibility to public, recheck GitHub's secret-scanning endpoint and record the release
tag and GitHub release URL in the release notes.

The two pytest directories must currently be invoked separately. Collecting both in one pytest
process causes their identically named `test_core.py` modules to collide; this is a test-packaging
limitation rather than a failing test assertion.
