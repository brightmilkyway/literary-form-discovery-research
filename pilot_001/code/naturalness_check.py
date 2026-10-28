"""Does the v15 intervention move the text along a 'naturalness' direction?

Why this exists
  The baseline decomposition found that word-SCRAMBLED text forms its own
  cluster in both encoders: scrambled units are MORE similar to each other
  (0.658 / 0.856) than real units are (0.494 / 0.721), and real x scrambled is
  far apart (0.237 / 0.594).  So 'is this real language at all' is a direction
  in the embedding space, not a constant.

  The v15 intervention replaced content words with unrelated ones.  That
  changes the content, but it also makes the text less natural.  If the drop
  measured in v15 were mostly a shift along the naturalness direction, then the
  conclusion drawn from it - 'the link depends on content words being present'
  - would be confounded.

  The SHUF control in v15 already bounds this: shuffling the words in place
  destroys word order (very unnatural) while keeping the content, and it
  produced only about one third of the content effect.  This script measures
  the shift directly.

Design
  Build the two poles from real data:
    real pole    centroid of the 78 real *The Dead* units
    salad pole   centroid of word-scrambled copies (same token count and
                 punctuation, all words replaced)
  Then, for every variant that v15 already encoded (read back from the
  checkpoint, no re-encoding), report
    cos to the real pole, cos to the salad pole, and the naturalness
    coordinate = cos(v, real) - cos(v, salad)

  If the content variants sit near the real pole, v15 stands.  If they have
  slid towards the salad pole, part of the v15 effect is a naturalness shift
  and that has to be said.

Writes ``data/derived/naturalness_check_<tag>.txt``.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np

import common as C
from baseline_decomposition import scramble
from graded_content_distance import GradedReplacer, LEVELS
from ladder_principled import build_pooler

DERIVED = C.DATA_DIR / "derived"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="minilm")
    ap.add_argument("--scope", default="maximal")
    ap.add_argument("--sample", type=int, default=78)
    args = ap.parse_args()

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)

    ckdir = DERIVED / "_ckpt" / f"{args.tag}_{args.scope}"
    jobs = [json.loads(x) for x in
            (ckdir / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = json.loads((ckdir / "meta.json").read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(args.model)
    m.eval()
    enc = build_pooler(m)
    rep = GradedReplacer(args.model, model=m)
    rng = random.Random(42)

    base = enc([u.text for u in units])
    idx = list(range(len(units)))[:args.sample]
    salad = enc([scramble(units[i].text, rep, rng) for i in idx])

    real_pole = base.mean(axis=0)
    real_pole /= np.linalg.norm(real_pole) + 1e-12
    salad_pole = salad.mean(axis=0)
    salad_pole /= np.linalg.norm(salad_pole) + 1e-12
    print(f"pole cosine (real vs salad): {float(real_pole @ salad_pole):+.3f}")

    # read the variants back from the checkpoint
    starts = sorted(ckdir.glob("main_*.npy"))
    V = np.vstack([np.load(f) for f in starts])
    assert len(V) == len(jobs), f"{len(V)} embeddings vs {len(jobs)} jobs"
    print(f"variants read back: {len(V)}")

    n = len(units)
    A_of = np.array([meta[j[0]]["A"] for j in jobs])
    B_of = np.array([meta[j[0]]["B"] for j in jobs])
    Sv = V @ base.T
    row0 = np.array([base[a] @ base.T for a in A_of])
    ok = np.ones((len(jobs), n), dtype=bool)
    ok[np.arange(len(jobs)), A_of] = False
    delta = Sv - row0
    dm = np.array([delta[k][ok[k]].mean() for k in range(len(jobs))])
    spec = np.array([delta[k, B_of[k]] for k in range(len(jobs))]) - dm

    cos_real = V @ real_pole
    cos_salad = V @ salad_pole
    nat = cos_real - cos_salad

    base_real = base @ real_pole
    base_salad = base @ salad_pole

    OUT = []
    def w(*a):
        OUT.append(" ".join(str(x) for x in a))

    w("=" * 78)
    w("DID THE v15 INTERVENTION MOVE THE TEXT TOWARDS 'WORD SALAD'?")
    w("=" * 78)
    w(f"  model {args.model};  scope {args.scope}")
    w(f"  real pole  = centroid of {args.sample} real *The Dead* units")
    w(f"  salad pole = centroid of {args.sample} word-scrambled units")
    w(f"  cosine between the two poles: {float(real_pole @ salad_pole):+.3f}")
    w("")
    w("  where the ORIGINAL units sit:")
    w(f"    mean cos to real pole  : {base_real.mean():+.3f}")
    w(f"    mean cos to salad pole : {base_salad.mean():+.3f}")
    w(f"    naturalness coordinate : {(base_real - base_salad).mean():+.3f}")
    w("")
    w("  where each intervention level sits  (mean over all variants)")
    w(f"    {'level':<18}{'cos real':>10}{'cos salad':>11}"
      f"{'nat coord':>11}{'shift':>9}{'partner-specific':>18}")
    w(f"    {'(original)':<18}{base_real.mean():>10.3f}{base_salad.mean():>11.3f}"
      f"{(base_real - base_salad).mean():>11.3f}{0.0:>9.3f}"
      f"{0.0:>18.3f}")
    for lbl, _ in LEVELS:
        ks = [k for k, j in enumerate(jobs) if j[1] == lbl]
        if not ks:
            continue
        cr = float(cos_real[ks].mean())
        cs = float(cos_salad[ks].mean())
        shift = (cr - cs) - (base_real.mean() - base_salad.mean())
        w(f"    {lbl:<18}{cr:>10.3f}{cs:>11.3f}{cr - cs:>11.3f}"
          f"{shift:>+9.3f}{spec[ks].mean():>18.3f}")
    w("")
    w("  READING")
    w("    'shift' is how far the naturalness coordinate moved, negative =")
    w("    towards word salad.  Compare it with the partner-specific damage:")
    w("    if the damage scaled with the shift, the v15 effect would be a")
    w("    naturalness artefact.")
    w("")
    # the key comparison
    shuf = [k for k, j in enumerate(jobs) if j[1] == "SHUF order"]
    far = [k for k, j in enumerate(jobs) if j[1] == "q=0.00 furthest"]
    near = [k for k, j in enumerate(jobs) if j[1] == "q=0.99 near"]
    if shuf and far and near:
        s_sh = (cos_real[shuf] - cos_salad[shuf]).mean() - \
               (base_real.mean() - base_salad.mean())
        s_fa = (cos_real[far] - cos_salad[far]).mean() - \
               (base_real.mean() - base_salad.mean())
        s_ne = (cos_real[near] - cos_salad[near]).mean() - \
               (base_real.mean() - base_salad.mean())
        w("    SHUF   (order destroyed, content kept) :")
        w(f"        naturalness shift {s_sh:+.3f}   partner-specific "
          f"{spec[shuf].mean():.3f}")
        w("    q=0.00 (content replaced, order kept)  :")
        w(f"        naturalness shift {s_fa:+.3f}   partner-specific "
          f"{spec[far].mean():.3f}")
        w("    q=0.99 (near words, order kept)        :")
        w(f"        naturalness shift {s_ne:+.3f}   partner-specific "
          f"{spec[near].mean():.3f}")
        w("")
        w("    SHUF is the most unnatural of the three but does the least")
        w("    damage; if naturalness were the driver that ordering would be")
        w("    the other way round.")
    w("")

    out = DERIVED / f"naturalness_check_{args.tag}.txt"
    out.write_text("\n".join(OUT), encoding="utf-8")
    print("\n".join(OUT))
    print("wrote", out)


if __name__ == "__main__":
    main()
