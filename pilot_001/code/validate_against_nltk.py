"""Validate our TextTiling re-implementation against NLTK's reference.

We run NLTK's ``TextTilingTokenizer`` in demo mode on the same body text and
compare (a) the token-sequence count, (b) the gap-score curve, and (c) the set
of boundary gaps.  Agreement here is what lets us call our implementation
"faithful to Hearst (1997)".

Run from the code/ directory:  python validate_against_nltk.py
"""

from __future__ import annotations

import numpy as np

import common as C
import texttiling as TT


def main():
    raw = C.load_raw()
    body, _ = C.strip_title(raw)
    paras = C.paragraphs(body)

    # ---- NLTK reference ---------------------------------------------------
    from nltk.tokenize import TextTilingTokenizer
    tt = TextTilingTokenizer(w=20, k=6, demo_mode=True)
    n_gap, n_smooth, n_depth, n_bounds = tt.tokenize(body)

    # ---- ours -------------------------------------------------------------
    mine = TT.run_texttiling(body, paras, w=20, k=6, _return_internal=True)
    m_gap = np.array(mine["block_sim"])
    m_depth = np.array(mine["depth"])
    m_isb = np.array(mine["_internal"]["is_b"])

    print("token sequences: nltk", len(n_gap) + 1, " ours", mine["n_token_seqs"])
    print("gaps:            nltk", len(n_gap), " ours", len(m_gap))
    ok = True
    if len(n_gap) == len(m_gap):
        d = np.abs(np.array(n_gap) - m_gap)
        print(f"gap-score  max abs diff = {d.max():.3e}")
        dd = np.abs(np.array(n_depth) - m_depth)
        print(f"depth      max abs diff = {dd.max():.3e}")
        ok &= d.max() < 1e-9 and dd.max() < 1e-9
    else:
        ok = False
    n_b = [i for i, v in enumerate(n_bounds) if v]
    m_b = [i for i, v in enumerate(m_isb) if v]
    print("nltk boundaries:", len(n_b), n_b)
    print("ours boundaries:", len(m_b), m_b)
    ok &= (n_b == m_b)

    # ---- stopword list check ---------------------------------------------
    from nltk.corpus import stopwords
    theirs = set(stopwords.words("english"))
    ours = set(TT._NLTK_EN_STOPWORDS)
    print("stopwords: nltk", len(theirs), " ours", len(ours),
          " only-nltk", sorted(theirs - ours)[:10],
          " only-ours", sorted(ours - theirs)[:10])
    ok &= (theirs == ours)

    print()
    print("VALIDATION:", "PASS — our TextTiling reproduces NLTK exactly"
          if ok else "FAIL — see differences above")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
