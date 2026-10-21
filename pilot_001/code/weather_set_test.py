"""Set-level test: does a 'weather' passage set hang together?

Individual-pair tests are fragile (a pair can fail mutual rank while the set is
real).  So define the weather set W *by content* — every unit in which snow,
rain, wet or window is salient — and ask, for each method, whether pairs
INSIDE W are more similar than pairs from W to everything else.

If the encoders are blind to the weather thread, within-W similarity should
not exceed the baseline.  If they see it, it should.

Writes ``data/derived/weather_set_test.txt``.
"""

from __future__ import annotations

import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    n = len(units)

    # ---- define the weather set by content, not by method ---------------
    WEATHER = ("snow", "rain", "wet", "window", "goloshes", "shivering")
    W = []
    for u in units:
        t = u.text.lower()
        hits = sum(t.count(k) for k in WEATHER)
        if hits >= 2:
            W.append(u.index)
    w("=" * 78)
    w("SET-LEVEL TEST: DOES A 'WEATHER' PASSAGE SET HANG TOGETHER?")
    w("=" * 78)
    w()
    w(f"W = units with >=2 hits of {WEATHER}")
    w(f"    n_W = {len(W)}: {W}")
    for i in W:
        w(f"      unit {i:>2} (paras {units[i].first_para}-"
          f"{units[i].last_para}): "
          f"{re.sub(r'\s+', ' ', units[i].text)[:110]}")
    w()

    mats = {
        "TF-IDF": np.load(DERIVED / "tfidf_group150.npy"),
        "Jaccard": np.load(DERIVED / "jaccard_group150.npy"),
        "MiniLM": np.load(DERIVED / "sbert_group150_pooled.npy"),
        "bge": np.load(DERIVED / "sbert_group150_bge.npy"),
    }
    NCD = 1.0 - np.load(DERIVED / "ncd_group150_zlib.npy")     # to similarity
    NCDr = -np.load(DERIVED / "ncd_group150_zlib_residual.npy")
    mats["NCD(1-d)"] = NCD
    mats["NCD-resid"] = NCDr

    w("   For each method: mean similarity for")
    w("     within-W pairs (i,j in W, i<j),")
    w("     W-to-rest pairs (i in W, j not in W),")
    w("     and the percentile of the within-W mean in a permutation null")
    w("     (10,000 random sets of the same size).")
    w()
    w(f"   {'method':<12}{'within-W':>10}{'W-to-rest':>11}{'diff':>8}"
      f"{'perm pct':>10}")
    rng = np.random.default_rng(42)
    results = {}
    for name, M in mats.items():
        iu, ju = np.triu_indices(n, k=1)
        vals = M[iu, ju]
        inW = np.isin(iu, W) & np.isin(ju, W)
        fromW = (np.isin(iu, W) ^ np.isin(ju, W))
        m_in = float(vals[inW].mean())
        m_out = float(vals[fromW].mean())
        # permutation null
        null = np.empty(10000)
        Wset = np.array(W)
        for k in range(10000):
            pick = rng.choice(n, size=len(W), replace=False)
            mask = np.isin(iu, pick) & np.isin(ju, pick)
            null[k] = vals[mask].mean()
        pct = float((null < m_in).mean()) * 100
        w(f"   {name:<12}{m_in:>10.3f}{m_out:>11.3f}{m_in - m_out:>8.3f}"
          f"{pct:>9.1f}%")
        results[name] = (m_in, m_out, pct)
    w()
    w("   (Percentile near 100 = the weather set really does hang together")
    w("    more than a random set of the same size; near 50 = no signal.)")
    w()

    # ---- how much of this is adjacency? ---------------------------------
    w("   CORE-SET VARIANT — only the units where weather is most salient")
    w("   (snow>=3, or goloshes>=5, or rain+wet+window+shivering>=4):")
    core = []
    for u in units:
        t = u.text.lower()
        sc = t.count("snow") + 5 * t.count("goloshes") + t.count("rain") \
            + t.count("wet") + t.count("window") + t.count("shivering")
        if t.count("snow") >= 3 or t.count("goloshes") >= 5 or \
                (t.count("rain") + t.count("wet") + t.count("window")
                 + t.count("shivering")) >= 4:
            core.append(u.index)
    w(f"      W_core = {core} (n={len(core)})")
    iu, ju = np.triu_indices(n, k=1)
    for name, M in mats.items():
        vals = M[iu, ju]
        mask = np.isin(iu, core) & np.isin(ju, core)
        m_in = float(vals[mask].mean())
        null = np.empty(5000)
        rng2 = np.random.default_rng(7)
        for k in range(5000):
            pick = rng2.choice(n, size=len(core), replace=False)
            m2 = np.isin(iu, pick) & np.isin(ju, pick)
            null[k] = vals[m2].mean()
        pct = float((null < m_in).mean()) * 100
        w(f"      {name:<12} within-core mean = {m_in:.3f}   "
          f"permutation percentile {pct:.1f}%")
    w()

    w("   CONTROL — the same statistic for a set defined by *position*: the")
    w("   last 5 units, which are adjacent and therefore trivially similar.")
    Wpos = list(range(n - 5, n))
    for name, M in mats.items():
        vals = M[iu, ju]
        mask = np.isin(iu, Wpos) & np.isin(ju, Wpos)
        w(f"      {name:<12} within-last-5 mean = {vals[mask].mean():.3f}")

    (DERIVED / "weather_set_test.txt").write_text("\n".join(OUT),
                                                  encoding="utf-8")
    print("wrote", DERIVED / "weather_set_test.txt")


if __name__ == "__main__":
    main()
