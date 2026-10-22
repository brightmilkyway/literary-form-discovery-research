"""Scale adjudication, corrected: the permutation null is saturated at
sentence scale, so use a *matched word-set* null instead.

Why the first attempt failed
----------------------------
At sentence scale the permutation test (random sets of the same size) gives
~100% to almost every motif AND to sets defined by a single random content
word (median 99.2 / 95.4 / 100.0).  A word's occurrences already cluster into
scenes, so "random set" is far too weak a null, and the contiguous-block
sensitivity check scored *lower* than random word sets.  The test is
saturated and cannot discriminate.

Corrected design
----------------
For each motif and each distance band b (pairs at least b units apart):
  * obs  = mean within-set similarity of the motif
  * null = the same statistic for 300 sets defined by a single random content
           word, **matched in size** to the motif (within +/- 25%)
  * report obs, the null median, the percentile and the z-score
The band is what matters for the literary claim: the reading of "The Dead"
needs the weather to connect passages that are FAR apart, not adjacent ones.

Writes ``data/derived/scale_adjudication.txt`` (overwrites the failed first
version, whose numbers are kept in this docstring).
"""

from __future__ import annotations

import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

MOTIFS = {
    "weather": ["snow", "snowing", "snowy", "rain", "raining", "wet", "window",
                "goloshes", "shivering", "shiver", "flakes", "dew", "cold",
                "frost"],
    "music": ["song", "songs", "sing", "sings", "singing", "sang", "singer",
              "singers", "voice", "voices", "tune", "piano", "music", "choir",
              "concert", "soprano", "tenor", "ballad", "melody", "hummed",
              "humming", "note", "notes", "academy"],
    "religion": ["church", "chapel", "pope", "priest", "monk", "monks",
                 "monastery", "god", "religion", "protestant", "catholic",
                 "mass", "prayer", "prayers", "coffin", "christian", "heaven",
                 "sin", "sins", "soul"],
    "death": ["dead", "death", "died", "die", "dying", "grave", "buried",
              "burial", "churchyard", "corpse", "coffin", "tomb",
              "headstone", "headstones", "crosses", "mortal"],
    "food_drink": ["punch", "whisky", "lemonade", "stout", "port", "goose",
                   "ham", "pudding", "sultana", "cake", "wine", "tea",
                   "supper", "sirloin", "biscuits", "jelly", "carver"],
}
STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while yes o oh well said asked cried says say""".split())


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def sentence_units(body):
    import sentences as S
    return [{"text": s.text, "start": s.start, "end": s.end}
            for s in S.split_sentences(body)]


def para_units(body):
    return [{"text": p.text, "start": p.start, "end": p.end}
            for p in C.paragraphs(body)]


def word_set(text):
    return [t.lower().strip("'") for t in
            re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", text)]


def runs_of(idxs):
    if not idxs:
        return 0
    r, prev = 1, idxs[0]
    for i in idxs[1:]:
        if i != prev + 1:
            r += 1
        prev = i
    return r


def distance_norm(S, band):
    """Return a matrix of z-scores: each similarity minus the mean similarity
    of ALL pairs at the same index distance, divided by that distance's sd.

    This removes both the adjacency effect and the dispersion confound (a set
    spread over the whole text has all its pairs far apart, hence low raw
    similarity, which is not a property of the motif).
    """
    n = S.shape[0]
    ii, jj = np.mgrid[0:n, 0:n]
    d = np.abs(ii - jj)
    Z = np.zeros_like(S, dtype=float)
    for dd in range(band, n):
        m = d == dd
        if m.sum() < 2:
            continue
        vals = S[m]
        mu, sd = vals.mean(), vals.std()
        Z[m] = (S[m] - mu) / (sd + 1e-12)
    return Z


def score_set(Z, idxs, band):
    n = Z.shape[0]
    ii, jj = np.mgrid[0:n, 0:n]
    keep = np.abs(ii - jj) >= band
    idxs = np.asarray(idxs)
    sub = Z[np.ix_(idxs, idxs)]
    m = keep[np.ix_(idxs, idxs)] & np.triu(np.ones_like(sub, dtype=bool), k=1)
    if m.sum() == 0:
        return float("nan"), 0
    return float(sub[m].mean()), int(m.sum())


def within_mean(S, idxs, band):
    n = S.shape[0]
    ii, jj = np.mgrid[0:n, 0:n]
    keep = np.abs(ii - jj) >= band
    idxs = np.asarray(idxs)
    sub = S[np.ix_(idxs, idxs)]
    m = keep[np.ix_(idxs, idxs)]
    m = m & np.triu(np.ones_like(sub, dtype=bool), k=1)
    if m.sum() == 0:
        return float("nan"), 0
    return float(sub[m].mean()), int(m.sum())


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]

    import torch
    from sentence_transformers import SentenceTransformer

    m1 = SentenceTransformer("all-MiniLM-L6-v2"); m1.eval()
    m2 = SentenceTransformer("BAAI/bge-small-en-v1.5"); m2.eval()

    for scale_name, mk, bands in (("SENTENCE", sentence_units, (5, 30, 100)),
                                  ("PARAGRAPH", para_units, (3, 15, 40))):
        units = mk(body)
        texts = [u["text"] for u in units]
        n = len(units)
        w("=" * 78)
        w(f"SCALE: {scale_name}   n = {n} units, "
          f"{sum(len(t.split()) for t in texts)} words")
        w("=" * 78)
        w()

        mats = {}
        with torch.no_grad():
            e1 = m1.encode(texts, batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
            e2 = m2.encode(texts, batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        mats["MiniLM"] = e1 @ e1.T
        mats["bge"] = e2 @ e2.T
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True).fit_transform(
            [" ".join(word_set(t)) for t in texts]).toarray()
        nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1
        Xn = X / nrm
        mats["TF-IDF"] = Xn @ Xn.T

        toks = [set(word_set(t)) for t in texts]
        vocab = {}
        for i, ts in enumerate(toks):
            for t in ts:
                if t not in STOP and len(t) > 3:
                    vocab.setdefault(t, []).append(i)
        vocab = {k: v for k, v in vocab.items() if len(v) >= 4}

        for band in bands:
            Zs = {k: distance_norm(v, band) for k, v in mats.items()}
            w(f"   BAND = pairs at least {band} units apart")
            w("   score = mean over within-set pairs of "
              "(similarity - mean at that distance)/sd at that distance")
            w(f"   {'set':<14}{'n':>5}{'runs':>6}{'pairs':>7}" +
              "".join(f"{k:>26}" for k in mats))
            w(f"   {'':<14}{'':>5}{'':>6}{'':>7}" +
              "".join(f"{'score null_med  pct    z':>26}" for _ in mats))
            for name, lex in MOTIFS.items():
                ls = set(lex)
                idxs = [i for i in range(n) if toks[i] & ls]
                cells, paircnt = [], 0
                for mname, Z in Zs.items():
                    obs, paircnt = score_set(Z, idxs, band)
                    lo, hi = int(len(idxs) * 0.75), int(len(idxs) * 1.25)
                    pool = [k for k, v in vocab.items() if lo <= len(v) <= hi]
                    rng = random.Random(42)
                    rng.shuffle(pool)
                    pool = pool[:300]
                    null = np.array([score_set(Z, vocab[k], band)[0]
                                     for k in pool])
                    null = null[~np.isnan(null)]
                    med = float(np.median(null))
                    pct = float((null < obs).mean()) * 100
                    z = float((obs - null.mean()) / (null.std() + 1e-12))
                    cells.append(f"{obs:>7.2f}{med:>9.2f}{pct:>6.1f}{z:>5.1f}")
                w(f"   {name:<14}{len(idxs):>5}{runs_of(idxs):>6}{paircnt:>7}" +
                  "".join(f"{c:>26}" for c in cells))
            w()
            w("   shape-matched null: the SAME number of runs with the SAME run")
            w("   lengths, placed at random non-overlapping positions.  This")
            w("   controls for size, run count, run length and dispersion, so")
            w("   what is left is *which* units the motif picks.")
            for name, lex in MOTIFS.items():
                ls = set(lex)
                idxs = [i for i in range(n) if toks[i] & ls]
                runs = []
                cur = [idxs[0]]
                for i in idxs[1:]:
                    if i == cur[-1] + 1:
                        cur.append(i)
                    else:
                        runs.append(cur)
                        cur = [i]
                runs.append(cur)
                lengths = [len(r) for r in runs]
                cells = []
                for mname, Z in Zs.items():
                    obs, _ = score_set(Z, idxs, band)
                    rng = np.random.default_rng(11)
                    null = []
                    for _ in range(500):
                        placed, occ, ri, tries = [], set(), 0, 0
                        while ri < len(lengths) and tries < 2000:
                            tries += 1
                            st = int(rng.integers(0, max(1, n - 1)))
                            blk = list(range(st, min(n, st + lengths[ri])))
                            if any(b in occ for b in blk):
                                continue
                            placed += blk
                            occ.update(blk)
                            ri += 1
                        if ri < len(lengths):
                            continue
                        v, _ = score_set(Z, sorted(placed), band)
                        if not np.isnan(v):
                            null.append(v)
                    null = np.array(null)
                    if null.size == 0:
                        cells.append(f"{obs:>7.2f}{'--':>9}{'--':>6}")
                        continue
                    pct = float((null < obs).mean()) * 100
                    cells.append(f"{obs:>7.2f}{np.median(null):>9.2f}"
                                 f"{pct:>6.1f}")
                w(f"   {name:<14}{len(idxs):>5}{len(runs):>6}" +
                  "".join(f"{c:>22}" for c in cells))
            w(f"   {'':<14}{'':>5}{'runs':>6}" +
              "".join(f"{'score null_med  pct':>22}" for _ in mats))
            w()

        w("   null = 300 sets defined by a SINGLE random content word, matched")
        w("   in size to the motif (+/-25%).  z = (obs - null mean)/null sd.")
        w("   This null controls for the fact that any word's occurrences")
        w("   already cluster into scenes.")
        w()

    (DERIVED / "scale_adjudication.txt").write_text("\n".join(OUT),
                                                    encoding="utf-8")
    print("wrote", DERIVED / "scale_adjudication.txt")


if __name__ == "__main__":
    main()
