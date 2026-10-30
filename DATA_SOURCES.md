# Data sources, versions, and rights

## Primary literary source

The English text used in all three pilots derives from James Joyce's *Dubliners*, Project Gutenberg
eBook **#2814**:

- Catalog: <https://www.gutenberg.org/ebooks/2814>
- Plain-text upstream reference: <https://www.gutenberg.org/cache/epub/2814/pg2814.txt>
- Project Gutenberg release date: 2001-09-01
- Catalog last-update date observed during the public-release audit: 2021-05-21
- Catalog status: public domain in the United States
- Catalog credits: David Reed, Karol Pietrzak, and David Widger

The repository's `The Dead` snapshot was extracted from the final story, converted to UTF-8,
reflowed from hard-wrapped prose while preserving paragraph and verse boundaries, and had plain-text
underscore italics markers removed. The original one-step extraction script was not preserved, so
the repository fingerprints below—not byte identity with a future upstream download—define the
experimental source version. The current upstream file was compared during the 2026-10-30 audit;
the remaining differences are consistent with the documented reflow and italics-marker handling.

## Repository fingerprints

| File | Role | SHA-256 |
|---|---|---|
| `pilot_001/data/source/the_dead.txt` | Pilot 001 canonical snapshot; Pilot 002 reads this file | `13284c0070a518ed4f4c7d5cc857bc2bcdc11f229c3e042cc22e593c570a9f2d` |
| `pilot_003/data/source/the_dead.txt` | Pilot 003 self-contained copy | `13284c0070a518ed4f4c7d5cc857bc2bcdc11f229c3e042cc22e593c570a9f2d` |
| `pilot_001/data/source/dubliners_other_stories.txt` | Other fourteen stories used as external donors | `6aabd7a6fd6bf8fd5bdb5b7d9887dcddb61577f77c7773f175fc7d44cb264f59` |

`pilot_001/code/fetch_external_donors.py` records the donor extraction endpoint and story-boundary
logic. Pilot 003 records the normalized source hash in generated metadata so Windows and Unix
checkout line endings do not change the experiment identity.

The source files are now explicitly checked out with LF endings. Pilot 002's retained
`run_metadata.json` records the raw hash of its original Windows CRLF working copy
(`b414bf48...89e389`); after newline normalization, that input is identical to the canonical
`13284c00...a9f2d` content above. The historical run record is intentionally not rewritten.

## Rights and redistribution note

Project Gutenberg's catalog marks eBook #2814 as public domain in the United States. Project
Gutenberg also explains that copyright status can differ outside the United States and that users
must check the law of their own jurisdiction. References to Project Gutenberg here are source
acknowledgements, not an assertion of endorsement.

The literary files are **not** covered by this repository's MIT or CC BY 4.0 grants. The Project
Gutenberg name and trademark remain subject to Project Gutenberg's terms:
<https://www.gutenberg.org/policy/license.html>.

## Model-derived data

Pilot 003 retains compact raw DeepSeek responses for auditability. They contain no API key and are
excluded from the repository's CC BY grant. Users must consider the provider's applicable terms and
the possibility that a pretrained model had prior exposure to the literary work.
