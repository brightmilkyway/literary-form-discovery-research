"""Where does the high baseline cosine come from?

Why this exists
  The v15 result left one thing unexplained. In the full scan the mean cosine
  between two 150-word units of *The Dead* is 0.534 (MiniLM) / 0.744 (bge),
  and the top-500 pairs only reach 0.577-0.773.  A dynamic range that narrow,
  sitting on a baseline that high, has two very different possible causes:

    (a) a property of the TEXT - every chunk of one story shares its cast,
        its props, its register, so all chunks resemble each other; or
    (b) a property of the EMBEDDING SPACE - sentence encoders are anisotropic,
        every vector lies in a narrow cone, so any two texts score ~0.5
        whatever they say.

  The two readings lead to opposite conclusions about the whole line of work,
  so it has to be settled before anything else is built on top.

Design
  Four populations, all cut into 150-word groups the same way (never splitting
  a paragraph, never crossing a story boundary):

    dead         78 units, *The Dead*
    other        the 14 other *Dubliners* stories, per story
    scram_dead   word-scrambled copies of a sample of `dead`
    scram_other  word-scrambled copies of a sample of `other`

  The scrambled copies replace every WORD with a random word of the same token
  length class, drawn from the encoder's own vocabulary, while every piece of
  punctuation, every paragraph break and the exact token count are preserved.
  They are what a unit looks like with all of its language removed but all of
  its shape kept.

  `other` is the decisive control: same author, same collection, same period,
  same register, DIFFERENT story.  If the baseline were a same-story effect it
  must drop when crossing stories; if it is geometric it must not.

  Reported for each population and each model:
    * mean pairwise cosine inside the population
    * mean cosine to the population centroid (anisotropy indicator: all
      vectors pointing one way pushes this towards 1)
    * participation ratio of the centred embeddings, i.e. the effective number
      of dimensions actually used out of 384

Writes ``data/derived/baseline_decomposition_<tag>.txt``.
"""

from __future__ import annotations

import argparse
import random
import re

import numpy as np

import common as C
from graded_content_distance import GradedReplacer
from ladder_principled import build_pooler

DERIVED = C.DATA_DIR / "derived"
SRC = C.DATA_DIR / "source"

STORIES = ["THE SISTERS", "AN ENCOUNTER", "ARABY", "EVELINE", "AFTER THE RACE",
           "TWO GALLANTS", "THE BOARDING HOUSE", "A LITTLE CLOUD",
           "COUNTERPARTS", "CLAY", "A PAINFUL CASE",
           "IVY DAY IN THE COMMITTEE ROOM", "A MOTHER", "GRACE"]

WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
MIN_GAP = 5


def load_other_stories():
    """Return [(story_name, [Paragraph, ...]), ...] from the donor file.

    Titles are matched IN ORDER, taking the first occurrence after the previous
    one: the donor text repeats some titles as sub-headings inside a story
    ('A PAINFUL CASE' appears again after its own 'DEATH OF A LADY AT SYDNEY
    PARADE'), and a plain set-membership test would split a story in two.
    """
    raw = (SRC / "dubliners_other_stories.txt").read_text(encoding="utf-8")
    paras = C.paragraphs(raw)
    marks = []
    cursor = 0
    for name in STORIES:
        for p in paras:
            if p.index < cursor:
                continue
            if re.sub(r"\s+", " ", p.text).strip() == name:
                marks.append((p.index, name))
                cursor = p.index + 1
                break
    if len(marks) != len(STORIES):
        raise RuntimeError(f"found {len(marks)} of {len(STORIES)} story titles")
    out = []
    for k, (idx, name) in enumerate(marks):
        lo = idx + 1
        hi = marks[k + 1][0] if k + 1 < len(marks) else len(paras)
        out.append((name, paras[lo:hi]))
    return out


def scramble(text: str, rep: GradedReplacer, rng: random.Random) -> str:
    """Replace every word with a random same-length word; keep everything else."""
    def sub(m):
        w = m.group(0)
        new = rep.pick_at(w, set(), rng, 0.00)
        return new if new else w
    return WORD_RE.sub(sub, text)


def stats(E: np.ndarray):
    """Mean pairwise cosine, mean cosine to the centroid, participation ratio."""
    S = E @ E.T
    n = len(E)
    iu = np.triu_indices(n, k=1)
    mp = float(S[iu].mean()) if n > 1 else float("nan")
    cen = E.mean(axis=0)
    cenc = cen / (np.linalg.norm(cen) + 1e-12)
    to_cen = float((E @ cenc).mean())
    X = E - cen
    cov = (X.T @ X) / max(1, len(X))
    lam = np.linalg.eigvalsh(cov)
    lam = lam[lam > 1e-12]
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    return mp, to_cen, pr


def cross(A: np.ndarray, B: np.ndarray) -> float:
    return float((A @ B.T).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="minilm")
    ap.add_argument("--sample", type=int, default=60)
    args = ap.parse_args()

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    dead_units = C.build_units(body, "group", target_words=150)

    stories = load_other_stories()
    other = []          # (story_name, Unit)
    for name, ps in stories:
        for u in C.units_by_group(ps, 150):
            if u.n_words >= 100:
                other.append((name, u))

    print(f"The Dead units: {len(dead_units)}")
    print(f"other-story units: {len(other)} across {len(stories)} stories")
    for name, ps in stories:
        k = sum(1 for s, _ in other if s == name)
        print(f"    {name:<34} {k:>3} units")

    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(args.model)
    m.eval()
    enc = build_pooler(m)
    rep = GradedReplacer(args.model, model=m)
    rng = random.Random(42)

    D = enc([u.text for u in dead_units])
    O = enc([u[1].text for u in other])
    names = [s for s, _ in other]

    n_d = min(args.sample, len(dead_units))
    n_o = min(args.sample, len(other))
    d_idx = rng.sample(range(len(dead_units)), n_d)
    o_idx = rng.sample(range(len(other)), n_o)
    SD = enc([scramble(dead_units[i].text, rep, rng) for i in d_idx])
    SO = enc([scramble(other[i][1].text, rep, rng) for i in o_idx])

    print(f"scrambled: {len(SD)} dead, {len(SO)} other", flush=True)

    # ---- within-population -------------------------------------------------
    def within_gap(E, gap=MIN_GAP):
        S = E @ E.T
        n = len(E)
        iu = np.triu_indices(n, k=1)
        keep = (iu[1] - iu[0]) >= gap
        return float(S[iu[0][keep], iu[1][keep]].mean())

    def within_story(E, names_, gap=MIN_GAP):
        S = E @ E.T
        vals = []
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                if j - i < gap:
                    continue
                if names_[i] == names_[j]:
                    vals.append(S[i, j])
        return float(np.mean(vals)), len(vals)

    def across_story(E, names_):
        S = E @ E.T
        vals = [S[i, j] for i in range(len(E)) for j in range(i + 1, len(E))
                if names_[i] != names_[j]]
        return float(np.mean(vals)), len(vals)

    mp_d, cen_d, pr_d = stats(D)
    mp_o, cen_o, pr_o = stats(O)
    mp_sd, cen_sd, pr_sd = stats(SD)
    mp_so, cen_so, pr_so = stats(SO)

    wd = within_gap(D)
    wo, n_wo = within_story(O, names)
    ao, n_ao = across_story(O, names)

    OUT = []
    def w(*a):
        OUT.append(" ".join(str(x) for x in a))

    w("=" * 78)
    w("WHERE DOES THE HIGH BASELINE COME FROM?")
    w("=" * 78)
    w(f"  model {args.model}")
    w(f"  all units cut the same way: 150-word groups, no paragraph split,")
    w(f"  no story boundary crossed")
    w("")
    w("  populations")
    w(f"    dead        {len(D):>4} units  (*The Dead*)")
    w(f"    other       {len(O):>4} units  (14 other *Dubliners* stories)")
    w(f"    scram_dead  {len(SD):>4} units  (all words replaced, shape kept)")
    w(f"    scram_other {len(SO):>4} units")
    w("")
    w("  word counts: dead min/median/max "
      f"{min(u.n_words for u in dead_units)}/"
      f"{int(np.median([u.n_words for u in dead_units]))}/"
      f"{max(u.n_words for u in dead_units)}"
      f"   other {min(u.n_words for _, u in other)}/"
      f"{int(np.median([u.n_words for _, u in other]))}/"
      f"{max(u.n_words for _, u in other)}")
    w("")
    w("  MEAN PAIRWISE COSINE")
    w(f"    {'population / comparison':<44}{'mean cos':>10}{'n pairs':>12}")
    w(f"    {'dead x dead (gap >= 5)':<44}{wd:>10.3f}{'-':>12}")
    w(f"    {'dead x dead (all pairs)':<44}{mp_d:>10.3f}{'-':>12}")
    w(f"    {'other x other, SAME story (gap >= 5)':<44}{wo:>10.3f}{n_wo:>12}")
    w(f"    {'other x other, DIFFERENT story':<44}{ao:>10.3f}{n_ao:>12}")
    w(f"    {'other x other (all pairs)':<44}{mp_o:>10.3f}{'-':>12}")
    w(f"    {'---':<44}")
    w(f"    {'dead x other  (cross-story)':<44}{cross(D, O):>10.3f}{len(D)*len(O):>12}")
    w(f"    {'---':<44}")
    w(f"    {'scram_dead x scram_dead':<44}{mp_sd:>10.3f}{'-':>12}")
    w(f"    {'scram_other x scram_other':<44}{mp_so:>10.3f}{'-':>12}")
    w(f"    {'scram_dead x scram_other':<44}{cross(SD, SO):>10.3f}{'-':>12}")
    w(f"    {'---':<44}")
    w(f"    {'dead x scram_dead (real vs own scramble)':<44}{cross(D, SD):>10.3f}{'-':>12}")
    w(f"    {'dead x scram_other':<44}{cross(D, SO):>10.3f}{'-':>12}")
    w(f"    {'other x scram_dead':<44}{cross(O, SD):>10.3f}{'-':>12}")
    w("")
    w("  ANISOTROPY INDICATORS  (all vectors pointing one way -> cos-to-centroid")
    w("  towards 1; participation ratio = effective dimensions used of 384)")
    w(f"    {'population':<16}{'cos to centroid':>18}{'participation ratio':>22}")
    for lbl, E in (("dead", D), ("other", O), ("scram_dead", SD),
                   ("scram_other", SO)):
        _, c, p = stats(E)
        w(f"    {lbl:<16}{c:>18.3f}{p:>22.1f}")
    w("")
    w("  DOES LENGTH DRIVE IT?  (correlation of cosine with length difference)")
    for lbl, E, lens in (("dead", D, [u.n_words for u in dead_units]),
                         ("other", O, [u.n_words for _, u in other])):
        S = E @ E.T
        n = len(E)
        iu = np.triu_indices(n, k=1)
        keep = (iu[1] - iu[0]) >= MIN_GAP
        a, b = iu[0][keep], iu[1][keep]
        dl = np.abs(np.array(lens)[a] - np.array(lens)[b]).astype(float)
        cs = S[a, b]
        r = float(np.corrcoef(dl, cs)[0, 1]) if dl.std() > 0 else float("nan")
        w(f"    {lbl:<16} r(cos, |len_i - len_j|) = {r:+.3f}   "
          f"(length sd {np.std(lens):.1f} words)")
    w("")

    out = DERIVED / f"baseline_decomposition_{args.tag}.txt"
    out.write_text("\n".join(OUT), encoding="utf-8")
    print("\n".join(OUT))
    print("wrote", out)


if __name__ == "__main__":
    main()
