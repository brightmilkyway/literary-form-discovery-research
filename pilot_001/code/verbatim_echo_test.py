"""The decisive micro-test: the verbatim snow echo at sentence scale.

Two sentences carry the same phrase:

  534  (Mary Jane, quoting the newspaper)
       "They say," said Mary Jane, "we haven't had snow like it for thirty
        years; and I read this morning in the newspapers that the snow is
        general all over Ireland."
  759  (Gabriel, closing)
       "Yes, the newspapers were right: snow was general all over Ireland."

At the 150-word analysis unit used by the main run, no method recovered this
pair.  At sentence scale both encoders rank each sentence as the other's
single most similar sentence in the whole story.

This script reproduces that result and also reports how the snow sentences
compare with the rest.

Writes ``data/derived/verbatim_echo_test.txt``.
"""

from __future__ import annotations

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []
PHRASE = "general all over Ireland"


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    import sentences as S
    sunits = S.split_sentences(body)
    sents = [s.text for s in sunits]
    n = len(sents)

    import torch
    from sentence_transformers import SentenceTransformer

    models = {"MiniLM": "all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}
    mats = {}
    for name, mid in models.items():
        m = SentenceTransformer(mid)
        m.eval()
        with torch.no_grad():
            e = m.encode(sents, batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        mats[name] = e @ e.T

    idx = [i for i, s in enumerate(sents) if PHRASE in s]
    w("=" * 78)
    w("VERBATIM SNOW ECHO AT SENTENCE SCALE")
    w("=" * 78)
    w()
    w(f"sentence scale: {n} sentences")
    w(f"sentences containing {PHRASE!r}: indices {idx}")
    for i in idx:
        w(f"   [{i}] {sents[i]}")
    w()
    if len(idx) < 2:
        w("fewer than two sentences found; nothing to test")
    else:
        i, j = idx[0], idx[-1]
        w(f"pair ({i}, {j}), index distance {j - i} sentences")
        w(f"   {'model':<8}{'cos':>8}{'i->j':>7}{'j->i':>7}{'worse':>7}"
          f"{'of':>6}")
        for name, S in mats.items():
            a = int((np.delete(S[i], i) > S[i, j]).sum()) + 1
            b = int((np.delete(S[j], j) > S[j, i]).sum()) + 1
            w(f"   {name:<8}{S[i, j]:>8.3f}{a:>7}{b:>7}{max(a, b):>7}{n:>6}")
        w()
        w("   (worse = the larger of the two row ranks; 1 means each sentence")
        w("    is the other's single most similar sentence in the story)")
        w()
        w("   how the echo sentence compares with the snow sentences and with")
        w("   the text as a whole:")
        snow = [k for k, s in enumerate(sents) if "snow" in s.lower()]
        for name, S in mats.items():
            row = np.delete(S[i], i)
            ws = [S[i, k] for k in snow if k != i]
            w(f"      {name:<8} mean cos to the other {len(ws)} snow sentences "
              f"= {np.mean(ws):.3f}   to all sentences = {row.mean():.3f}")
        w()
        w("   control: 200 random sentence pairs at a comparable distance")
        rng = np.random.default_rng(42)
        for name, S in mats.items():
            vals = []
            for _ in range(200):
                a = int(rng.integers(0, n))
                b = int(rng.integers(0, n))
                if abs(a - b) < 200 or a == b:
                    continue
                vals.append(S[a, b])
            w(f"      {name:<8} random distant pairs: mean {np.mean(vals):.3f}, "
              f"95th pct {np.percentile(vals, 95):.3f}, max {np.max(vals):.3f}")

    (DERIVED / "verbatim_echo_test.txt").write_text("\n".join(OUT),
                                                    encoding="utf-8")
    print("wrote", DERIVED / "verbatim_echo_test.txt")


if __name__ == "__main__":
    main()
