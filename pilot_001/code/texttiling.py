"""TextTiling (Hearst, 1997) — lexical cohesion change along the text.

Faithful re-implementation of the algorithm as described in Hearst (1997) and
cross-checked against NLTK's reference implementation
(``nltk.tokenize.TextTilingTokenizer``).  ``validate_against_nltk.py`` shows
that the gap scores and the boundary set agree with NLTK on this text.

Pipeline
--------
1.  Tokenise (lower-case ASCII letter runs; punctuation removed, as NLTK does).
2.  Chop into *token sequences* of ``w`` words (pseudo-sentences).
3.  Drop stopwords, then for every gap between two blocks of ``k`` token
    sequences compute the cosine similarity of the two blocks' frequency
    vectors.  At the text edges the block window is shortened (Hearst's rule).
4.  Smooth the similarity curve with a flat moving average
    (window ``smoothing_width + 1``, one pass) — Hearst's "smoothing" step that
    removes tiny valleys.
5.  Depth score of each gap: ``left_peak + right_peak - 2 * gap_score``.
6.  A gap is a boundary when its depth exceeds ``mean − sd/2`` (Hearst's
    default, NLTK's ``HC`` policy) or ``mean − sd`` (``LC``), and no *stronger*
    boundary lies within 4 gaps (non-maximum suppression, as in NLTK).

The result is a curve plus a list of candidate boundaries, each mapped back to
a character offset and a paragraph index so a reader can jump to the text.
No literary category is used anywhere.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from common import Paragraph, Unit

# --------------------------------------------------------------------------
# Stopwords (NLTK English list, embedded so no corpus download is required)
# --------------------------------------------------------------------------

_NLTK_EN_STOPWORDS = """i me my myself we our ours ourselves you you're you've you'll
you'd your yours yourself yourselves he him his himself she she's her hers herself
it it's its itself they them their theirs themselves what which who whom this that
that'll these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about against
between into through during before after above below to from up down in out on off
over under again further then once here there when where why how all any both each few
more most other some such no nor not only own same so than too very s t can will just
don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn
didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't
mustn mustn't needn needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won
won't wouldn wouldn't""".split()


def _smooth_flat(x: np.ndarray, window_len: int) -> np.ndarray:
    """Flat moving average with reflected edges (SciPy-cookbook ``smooth``).

    Copied in spirit from the function NLTK uses, so results match exactly.
    """
    if x.size < window_len:
        raise ValueError("input shorter than smoothing window")
    if window_len < 3:
        return x
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = np.ones(window_len, "d")
    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1: -window_len + 1]


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

import re

_WORD_TOK = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
_WORDCHAR = re.compile(r"\w+")
# NLTK keeps only these characters (after lower-casing) before matching \w+.
_NLTK_ALLOWED = set("abcdefghijklmnopqrstuvwxyz-' \n\t")


def _tokenise(body: str, mode: str = "nltk"):
    """Return (tokens, char_starts) with global char offsets in *body*.

    * ``nltk`` — exact replication of NLTK's pipeline: lower-case, drop every
      character outside ``[a-z-' \\n\\t]`` (so a curly apostrophe disappears and
      ``don’t`` becomes one token ``dont``), then match ``\\w+``.  Offsets are
      carried through the filtering so boundaries map back to the body.
    * ``word`` — keeps internal apostrophes (``don’t`` → ``don’t``), as in the
      other modules of this pilot.
    """
    if mode == "word":
        toks, starts = [], []
        for m in _WORD_TOK.finditer(body):
            toks.append(m.group(0).lower())
            starts.append(m.start())
        return toks, starts

    kept_chars: List[str] = []
    kept_pos: List[int] = []
    for i, ch in enumerate(body):
        lc = ch.lower()
        if len(lc) != 1:
            continue
        if lc in _NLTK_ALLOWED:
            kept_chars.append(lc)
            kept_pos.append(i)
    text = "".join(kept_chars)
    toks, starts = [], []
    for m in _WORDCHAR.finditer(text):
        toks.append(m.group(0))
        starts.append(kept_pos[m.start()])
    return toks, starts


@dataclass
class Boundary:
    gap_index: int
    word_pos: int
    char_pos: int
    depth: float
    block_sim: float
    left_unit: int | None = None
    right_unit: int | None = None


def run_texttiling(
    body: str,
    paras: Sequence[Paragraph],
    w: int = 20,
    k: int = 6,
    stopwords: Sequence[str] | None = None,
    smoothing_width: int = 2,
    smoothing_rounds: int = 1,
    cutoff_policy: str = "HC",
    suppress: int = 4,
    clip_edges: bool = True,
    tokenize_mode: str = "nltk",
    units: Sequence[Unit] | None = None,
    _return_internal: bool = False,
) -> dict:
    """Run TextTiling.  Returns curves, parameters and candidate boundaries."""
    sw = set(_NLTK_EN_STOPWORDS if stopwords is None else stopwords)

    toks, starts = _tokenise(body, tokenize_mode)
    n = len(toks)

    # ---- token sequences of length w --------------------------------------
    seqs: List[List[int]] = []          # word indices per sequence
    for i in range(0, n, w):
        seqs.append(list(range(i, min(i + w, n))))
    m = len(seqs)
    numgaps = m - 1
    if numgaps < 1:
        raise ValueError("text too short")

    # ---- block comparison at every gap -----------------------------------
    # Pre-compute the stopword-filtered word list per sequence.
    seq_words = [[toks[j] for j in idx if toks[j] not in sw] for idx in seqs]

    def blk_counter(lo: int, hi: int) -> Counter:
        c: Counter = Counter()
        for s in range(lo, hi):
            c.update(seq_words[s])
        return c

    gap_scores = np.zeros(numgaps, dtype=float)
    for g in range(numgaps):
        if g < k - 1:
            win = g + 1
        elif g > numgaps - k:
            win = numgaps - g
        else:
            win = k
        b1 = blk_counter(max(0, g - win + 1), g + 1)
        b2 = blk_counter(g + 1, min(m, g + 1 + win))
        if not b1 or not b2:
            gap_scores[g] = 0.0
            continue
        num = sum(b1[t] * b2[t] for t in (set(b1) & set(b2)))
        d1 = np.sqrt(sum(v * v for v in b1.values()))
        d2 = np.sqrt(sum(v * v for v in b2.values()))
        gap_scores[g] = float(num / (d1 * d2)) if d1 and d2 else 0.0

    # ---- smoothing --------------------------------------------------------
    smoothed = gap_scores.copy()
    for _ in range(max(0, smoothing_rounds)):
        smoothed = _smooth_flat(smoothed, smoothing_width + 1)

    # ---- depth scores -----------------------------------------------------
    scores = smoothed
    G = len(scores)
    depths = np.zeros(G, dtype=float)
    if clip_edges:
        clip = min(max(G // 10, 2), 5)
    else:
        clip = 0
    for idx in range(clip, G - clip):
        g = scores[idx]
        lpeak = g
        for s in scores[idx::-1]:
            if s >= lpeak:
                lpeak = s
            else:
                break
        rpeak = g
        for s in scores[idx:]:
            if s >= rpeak:
                rpeak = s
            else:
                break
        depths[idx] = lpeak + rpeak - 2 * g

    # ---- boundary identification -----------------------------------------
    avg, sd = float(depths.mean()), float(depths.std())
    cutoff = avg - (sd if cutoff_policy == "LC" else sd / 2.0)
    order = sorted(range(G), key=lambda i: -depths[i])
    hp = [i for i in order if depths[i] > cutoff]
    is_b = [0] * G
    for dt in hp:
        is_b[dt] = 1
        for dt2 in hp:
            if dt != dt2 and abs(dt2 - dt) < suppress and is_b[dt2] == 1:
                is_b[dt] = 0
                break

    # ---- map to char offsets & units --------------------------------------
    def char_of_gap(g: int) -> int:
        wj = min((g + 1) * w, n - 1)
        return starts[wj]

    def nearest_unit(pos: int, side: str) -> int | None:
        if not units:
            return None
        if side == "left":
            c = [u.index for u in units if u.end <= pos + 1]
            return c[-1] if c else None
        c = [u.index for u in units if u.start >= pos]
        return c[0] if c else None

    boundaries = []
    for g in range(G):
        if is_b[g]:
            cp = char_of_gap(g)
            boundaries.append(Boundary(
                gap_index=g, word_pos=(g + 1) * w, char_pos=cp,
                depth=float(depths[g]), block_sim=float(gap_scores[g]),
                left_unit=nearest_unit(cp, "left"),
                right_unit=nearest_unit(cp, "right"),
            ))

    out = {
        "params": {
            "w": w, "k": k, "smoothing_width": smoothing_width,
            "smoothing_rounds": smoothing_rounds,
            "cutoff_policy": cutoff_policy, "suppress": suppress,
            "tokenize_mode": tokenize_mode, "stopwords": len(sw),
            "clip_edges": clip_edges,
        },
        "n_words": n,
        "n_token_seqs": m,
        "gap_word_pos": [int((g + 1) * w) for g in range(G)],
        "gap_char_pos": [int(char_of_gap(g)) for g in range(G)],
        "block_sim": gap_scores.tolist(),
        "smoothed": smoothed.tolist(),
        "depth": depths.tolist(),
        "threshold": cutoff,
        "mean_depth": avg, "sd_depth": sd,
        "boundaries": [b.__dict__ for b in boundaries],
    }
    if _return_internal:
        out["_internal"] = {"gap_scores": gap_scores, "depths": depths,
                            "is_b": is_b}
    return out
