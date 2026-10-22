"""Redo the 'form baseline' experiment, without the placeholder artefact.

The withdrawn design replaced every content word with the same token "thing"
and observed that cosine rose to ~0.83, concluding that ~0.83 was a "form
baseline".  A reviewer pointed out that this is not a neutral form-only text:
it manufactures a large number of *identical* tokens across all passages, so
the rise measures the shared placeholder, not shared form.  The conclusion was
therefore overclaimed and has been retracted.

Corrected design — **delete, do not replace**:

  * ``form-only``    text = the passage with every content word removed
                     (function words, punctuation and order survive);
  * ``content-only`` text = the passage with every function word removed.

No artificial token is introduced, so a high ``form-only`` cosine really does
mean the two passages share syntactic/functional structure.

For each pair we then ask which of the two components predicts the original
cosine:

  * if ``cos_orig`` tracks ``cos_form``    -> the encoder is reading form;
  * if ``cos_orig`` tracks ``cos_content`` -> it is reading content.

Writes ``data/derived/form_vs_content.txt``.
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C
import sentences as S

DERIVED = C.DATA_DIR / "derived"
OUT = []

STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while yes o oh well""".split())

PAIRS = [
    ("song: lyric x title", 55, 58),
    ("religion: pope x monks", 30, 39),
    ("Mr Browne x Freddy", 12, 16),
    ("cold interrogation x sobbing", 71, 74),
    ("singers past x song name", 37, 58),
    ("Gabriel asks x revelation", 69, 72),
    ("snow frame (withdrawn)", 2, 73),
]


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def is_content(tok: str) -> bool:
    return tok.lower().strip("'") not in STOP and len(tok) > 1


def form_only(text: str) -> str:
    """Delete content words; keep function words, punctuation and order."""
    return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                  lambda m: "" if is_content(m.group(0)) else m.group(0),
                  text)


def content_only(text: str) -> str:
    return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                  lambda m: m.group(0) if is_content(m.group(0)) else "",
                  text)


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    n = len(units)

    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("all-MiniLM-L6-v2")
    m.eval()

    cache: dict[str, np.ndarray] = {}

    def e(text: str) -> np.ndarray:
        if text not in cache:
            with torch.no_grad():
                cache[text] = m.encode([text or " "], convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False)[0]
        return cache[text]

    w("=" * 78)
    w("FORM vs CONTENT — corrected design (delete, do not replace)")
    w("=" * 78)
    w()
    w("  form-only    = passage with every content word DELETED")
    w("  content-only = passage with every function word DELETED")
    w("  No placeholder token is introduced, so identical tokens cannot")
    w("  inflate the similarity.")
    w()
    w("  Sanity check — how much text survives each deletion:")
    for k in (2, 55, 58, 73):
        t = units[k].text
        w(f"    unit {k:>2}: {len(t.split()):>3} words -> form-only "
          f"{len(form_only(t).split()):>3}, content-only "
          f"{len(content_only(t).split()):>3}")
    w()

    w(f"  {'pair':<28}{'cos_orig':>10}{'cos_form':>10}{'cos_content':>13}")
    rows = []
    for label, i, j in PAIRS:
        a, b = units[i].text, units[j].text
        c0 = float(e(a) @ e(b))
        cf = float(e(form_only(a)) @ e(form_only(b)))
        cc = float(e(content_only(a)) @ e(content_only(b)))
        w(f"  {label:<28}{c0:>10.3f}{cf:>10.3f}{cc:>13.3f}")
        rows.append({"pair": f"{i}&{j}", "label": label,
                     "cos_orig": round(c0, 4), "cos_form": round(cf, 4),
                     "cos_content": round(cc, 4)})
    w()

    # ---- null sample -----------------------------------------------------
    w("  NULL — 120 random distant pairs (gap >= 3), same three quantities")
    rng = random.Random(42)
    seen, pairs = set(), []
    while len(pairs) < 120:
        i, j = rng.randrange(n), rng.randrange(n)
        if j - i < 3 or (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j))
    O, F, Cn = [], [], []
    for i, j in pairs:
        a, b = units[i].text, units[j].text
        O.append(float(e(a) @ e(b)))
        F.append(float(e(form_only(a)) @ e(form_only(b))))
        Cn.append(float(e(content_only(a)) @ e(content_only(b))))
    O, F, Cn = np.array(O), np.array(F), np.array(Cn)
    w(f"     mean: cos_orig {O.mean():.3f}  cos_form {F.mean():.3f}  "
      f"cos_content {Cn.mean():.3f}")
    w(f"     sd  : cos_orig {O.std():.3f}  cos_form {F.std():.3f}  "
      f"cos_content {Cn.std():.3f}")
    w()
    w("  WHICH COMPONENT PREDICTS THE ORIGINAL COSINE?")
    w(f"     r(cos_orig, cos_form)    = {np.corrcoef(O, F)[0,1]:+.3f}")
    w(f"     r(cos_orig, cos_content) = {np.corrcoef(O, Cn)[0,1]:+.3f}")
    w(f"     r(cos_form, cos_content) = {np.corrcoef(F, Cn)[0,1]:+.3f}")
    w()
    w("  The withdrawn claim was that a 'form baseline' of about 0.83 sits")
    w("  above every observed cosine.  Under the corrected design the")
    w(f"  form-only cosines average {F.mean():.3f} (sd {F.std():.3f}) and the")
    w(f"  content-only cosines average {Cn.mean():.3f} (sd {Cn.std():.3f}),")
    w("  so the two components are NOT separable by a single threshold; what")
    w("  matters is which one the original cosine tracks (see the")
    w("  correlations above).")
    w()

    # ---- sentence scale, where the echo lives ---------------------------
    w("-" * 78)
    w("  SENTENCE SCALE — the snow echo and its controls")
    w("-" * 78)
    sents = S.split_sentences(body)
    hits = [s for s in sents if "general all over Ireland" in s.text]
    if len(hits) >= 2:
        a, b = hits[0].text, hits[1].text
        w(f"    echo pair ({hits[0].index}, {hits[1].index}):")
        w(f"      cos_orig    {float(e(a) @ e(b)):.3f}")
        w(f"      cos_form    {float(e(form_only(a)) @ e(form_only(b))):.3f}")
        w(f"      cos_content "
          f"{float(e(content_only(a)) @ e(content_only(b))):.3f}")
        w(f"      form-only text  : {re.sub(r'\\s+', ' ', form_only(a))[:90]}")
        w(f"      content-only text: "
          f"{re.sub(r'\\s+', ' ', content_only(a))[:90]}")

    import csv
    with (C.TABLE_DIR / "form_vs_content.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    (DERIVED / "form_vs_content.txt").write_text("\n".join(OUT),
                                                 encoding="utf-8")
    print("wrote", DERIVED / "form_vs_content.txt")


if __name__ == "__main__":
    main()
