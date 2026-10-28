"""Does the encoder's passage similarity respond to the SEMANTIC DISTANCE of
the content words it is given?

Why this test exists
  Two earlier measurements point in opposite directions.

    form_vs_content (v7)     over 120 random pairs, the original cosine
                             tracks the CONTENT-ONLY cosine at r = +0.782.
                             Content appears to drive the similarity.

    full scan (v14)          replacing the shared content words of the
                             target paragraph leaves the link intact
                             (mean z = -0.89 / -0.65).

  If word meaning reaches the pooled passage embedding, then how far the
  replacement word is from the original must matter: replacing 'snow' with
  'frost' (a near neighbour) should hurt the link less than replacing it with
  'sausage' (a distant word).  If the two hurt it equally, word identity is
  not reaching the passage vector.

Two corrections to the v14 probe, both needed before this question can be asked
  1. SCOPE.  The v14 probe replaced only the shared occurrences in the single
     most-shared paragraph - a median of 5 words out of a 150-word unit.  Too
     light to show a gradient.  The default here is MAXIMAL: every content word
     of every paragraph of A that shares any content word with B, which is
     ~58% of the tokens of the unit.
  2. WHAT IS REPLACED.  The v14 probe replaced EVERY word, function words
     included, so the syntax and the function-word profile were destroyed
     along with the content.  The conclusion drawn from it - that what
     survives is 'form' - was therefore not supported by its own design.
     Here only CONTENT words are replaced; function words, punctuation,
     word order, sentence breaks and paragraph breaks are all held fixed.

Levels
  q = 0.99 / 0.50 / 0.05 / 0.00
     the replacement is drawn at that SIMILARITY QUANTILE of the encoder's own
     input-embedding matrix, among candidates of the same token length class:
       0.99 nearest   (same semantic field - a rough paraphrase)
       0.00 furthest  (what the v14 probe used)
  SHUF
     control: the content words of the same positions are permuted among
     themselves.  The content multiset, and therefore the content overlap with
     B, is UNCHANGED; only the order is destroyed.  If the passage similarity
     drops under SHUF as much as under far replacement, the drop is about
     order and not about which words are present.

Four quantities per level
  z            the scan's measure: drop in similarity to B, relative to the
               drop to every other unit.  NOTE this is a SELECTIVITY measure,
               not a strength measure: its denominator is the spread of the
               other units' deltas, which grows with the strength of the
               intervention, so z can stay flat while the raw drop doubles.
               The raw drop is therefore reported next to it.
  cos to B     the raw drop.  This is what says how hard the link was hit.
  cos_content  cosine of the content-only strings (v7's definition).  The
               CHECK that the intervention did what it claims: high under
               SHUF (same words), collapsing towards the furthest level.
  rank of B    where B sits among all 78 units after the intervention, and
               its margin over the 2nd nearest.  This is the project's own
               judging standard (correction-free rank) and it is the most
               direct answer to 'did the encoder stop seeing B as the match'.

RESUME
  This run is long (CPU only) and the session has already lost one attempt to
  an interruption, so every stage is checkpointed under
  ``data/derived/_ckpt/<tag>_<scope>/``:

    jobs.jsonl        the variants, so they are not rebuilt
    main_<i>.npy      the passage embeddings, one file per chunk
    cont_<i>.npy      the content-only embeddings, one file per chunk

  Re-running the same command picks up where it stopped.  A fingerprint of the
  configuration and of the exact text list is stored in ``fingerprint.txt``;
  if it does not match, the checkpoint directory is discarded and rebuilt, so
  stale chunks can never be mixed with fresh ones.

Writes ``data/derived/graded_content_distance_<tag>_<scope>.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch

import common as C
from form_vs_content import content_only, is_content
from ladder_principled import Replacer, build_pooler, ctypes

DERIVED = C.DATA_DIR / "derived"
LEVELS = [("q=0.99 near", 0.99), ("q=0.50 mid", 0.50),
          ("q=0.05 far", 0.05), ("q=0.00 furthest", 0.00),
          ("SHUF order", None)]
CHUNK = 256


class GradedReplacer(Replacer):
    """Replacer that can be aimed at a chosen similarity quantile.

    The candidate order for a word is computed once and cached: the maximal
    scope asks for a replacement some 300,000 times, and a fresh matrix
    product per call is far too slow.
    """

    def __init__(self, model_name: str, model=None):
        super().__init__(model_name, model=model)
        self._order: dict[str, list[str] | None] = {}

    def _ordered(self, target: str) -> list[str] | None:
        t = target.lower()
        if t in self._order:
            return self._order[t]
        cands = []
        for L in (len(t), len(t) - 1, len(t) + 1):
            cands += self.by_len.get(L, [])
        if not cands:
            cands = list(range(len(self.words)))
        tid = self.idx.get(t)
        if tid is None:
            res = None
        else:
            tvec = self.W[tid]
            tvec = tvec / (np.linalg.norm(tvec) + 1e-9)
            sims = self.E[cands] @ tvec
            order = np.argsort(sims)          # ascending: least similar first
            res = [self.words[cands[int(k)]] for k in order]
        self._order[t] = res
        return res

    def pick_at(self, target: str, banned: set[str], rng: random.Random,
                q: float) -> str | None:
        res = self._ordered(target)
        if res is None:
            return None
        m = len(res)
        k0 = int(round(q * (m - 1)))
        tl = target.lower()
        for off in range(0, 500):
            for k in (k0 - off, k0 + off):
                if 0 <= k < m:
                    word = res[k]
                    if word not in banned and word != tl:
                        return word
        return None


def encode_ckpt(m, texts, ckdir, prefix):
    """Encode in chunks, one .npy per chunk; reuse chunks already on disk."""
    ckdir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(texts), CHUNK))
    for i, a in enumerate(starts):
        f = ckdir / f"{prefix}_{i:05d}.npy"
        if f.exists():
            continue
        with torch.no_grad():
            e = m.encode(texts[a:a + CHUNK], batch_size=CHUNK,
                         convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        np.save(f, e.astype(np.float32))
        print(f"    {prefix} chunk {i + 1}/{len(starts)}", flush=True)
    parts = [np.load(ckdir / f"{prefix}_{i:05d}.npy") for i in range(len(starts))]
    return np.vstack(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="minilm")
    ap.add_argument("--scope", choices=("maximal", "target"),
                    default="maximal")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)
    n = len(units)

    ckdir = DERIVED / "_ckpt" / f"{args.tag}_{args.scope}"
    ckdir.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: variants, checkpointed every BATCH pairs -------------
    #  Each pair draws from its own RNG (seeded 1000 + pair index), so a run
    #  that resumes at pair k produces exactly what an uninterrupted run
    #  would have produced at pair k.  A single shared RNG stream could not
    #  be resumed without replaying every earlier pair.
    BATCH = 25
    pairs_file = ckdir / "pairs.json"
    prog_file = ckdir / "build_progress.json"

    if prog_file.exists():
        done = json.loads(prog_file.read_text(encoding="utf-8"))["pairs_done"]
        pairs = [tuple(x) for x in json.loads(
            pairs_file.read_text(encoding="utf-8"))]
        jobs = [json.loads(x) for x in
                (ckdir / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]
        meta = json.loads((ckdir / "meta.json").read_text(encoding="utf-8"))
        ctexts = [json.loads(x) for x in
                  (ckdir / "ctexts_pairs.jsonl").read_text(
                      encoding="utf-8").splitlines()]
        print(f"build resumed at pair {done}/{len(pairs)} "
              f"({len(jobs)} variants so far)", flush=True)
    else:
        done = 0
        jobs, meta, ctexts = [], [], []
        from sentence_transformers import SentenceTransformer
        m0 = SentenceTransformer(args.model)
        m0.eval()
        enc0 = build_pooler(m0)
        base0 = enc0([u.text for u in units])
        rep = GradedReplacer(args.model, model=m0)

        iu, ju = np.triu_indices(n, k=1)
        S0 = base0 @ base0.T
        cand = sorted(((float(S0[i, j]), int(i), int(j))
                       for i, j in zip(iu, ju) if j - i >= 5),
                      key=lambda t: -t[0])
        pairs = cand[:args.pairs]
        pairs_file.write_text(json.dumps(pairs), encoding="utf-8")
        del m0, enc0

    if done < len(pairs):
        rep = GradedReplacer(args.model)
        t0 = time.time()
        for pi in range(done, len(pairs)):
            rng = random.Random(1000 + pi)
            cos0, A, B = pairs[pi]
            A, B = int(A), int(B)
            u = units[A]
            btypes = ctypes(units[B].text)
            best_p, best_s = None, -1
            for q_ in paras[u.first_para:u.last_para + 1]:
                s = len(ctypes(q_.text) & btypes)
                if s > best_s:
                    best_p, best_s = q_, s
            p = best_p
            pw = p.text.split()
            shared = [i for i, t in enumerate(pw)
                      if t.lower().strip("'\u2019") in btypes]

            if args.scope == "target":
                targets = {p.index: shared}
            else:
                targets = {}
                for q_ in paras[u.first_para:u.last_para + 1]:
                    if ctypes(q_.text) & btypes:
                        targets[q_.index] = [
                            i for i, t in enumerate(q_.text.split())
                            if is_content(t)]

            def rebuild(new_of):
                parts = [paras[k].text
                         for k in range(u.first_para, u.last_para + 1)]
                for idx, newp in new_of.items():
                    parts[idx - u.first_para] = newp
                return "\n\n".join(parts)

            n_tok = sum(len(paras[idx].text.split()) for idx in targets)
            n_rep = sum(len(v) for v in targets.values())

            if targets:
                for lvl, qq in LEVELS:
                    for _ in range(args.K):
                        new_of = {}
                        for idx, idxs in targets.items():
                            words = paras[idx].text.split()
                            if qq is None:
                                vals = [words[i] for i in idxs]
                                rng.shuffle(vals)
                                for i, v in zip(idxs, vals):
                                    words[i] = v
                            else:
                                for i in idxs:
                                    wn = rep.pick_at(words[i], btypes, rng, qq)
                                    if wn:
                                        words[i] = wn
                            new_of[idx] = " ".join(words)
                        jobs.append([pi, lvl, rebuild(new_of)])
            meta.append(dict(A=A, B=B, cos0=float(cos0), n_paras=len(targets),
                             n_rep=n_rep, n_tok=n_tok))
            ctexts.append(content_only(units[A].text))
            ctexts.append(content_only(units[B].text))

            if (pi + 1) % BATCH == 0 or pi + 1 == len(pairs):
                npairs_done = pi + 1
                (ckdir / "jobs.jsonl").write_text(
                    "\n".join(json.dumps(j) for j in jobs), encoding="utf-8")
                (ckdir / "meta.json").write_text(json.dumps(meta),
                                                 encoding="utf-8")
                (ckdir / "ctexts_pairs.jsonl").write_text(
                    "\n".join(json.dumps(t) for t in ctexts),
                    encoding="utf-8")
                (ckdir / "ctexts_jobs.jsonl").write_text(
                    "\n".join(json.dumps(content_only(t)) for _, _, t in jobs),
                    encoding="utf-8")
                prog_file.write_text(json.dumps(
                    {"pairs_done": npairs_done, "jobs": len(jobs)}),
                    encoding="utf-8")
                el = time.time() - t0
                rate = (npairs_done - done) / max(el, 1e-9)
                print(f"  built {npairs_done}/{len(pairs)} pairs "
                      f"({len(jobs)} variants, {rate:.1f} pairs/s, "
                      f"eta {int((len(pairs)-npairs_done)/max(rate,1e-9))}s)",
                      flush=True)

    # content-only strings: the per-pair ones first, then the per-variant ones
    ctexts = ctexts + [json.loads(x) for x in
                       (ckdir / "ctexts_jobs.jsonl").read_text(
                           encoding="utf-8").splitlines()]
    assert len(ctexts) == 2 * len(meta) + len(jobs), (
        f"content strings misaligned: {len(ctexts)} != "
        f"2*{len(meta)} + {len(jobs)}")
    print(f"variants {len(jobs)}; content strings {len(ctexts)}", flush=True)

    # ---- stage 2: fingerprint the text list ----------------------------
    h = hashlib.sha1()
    h.update(args.model.encode())
    h.update(str(len(jobs)).encode())
    for _, _, t in jobs:
        h.update(t.encode())
    h.update(str(len(ctexts)).encode())
    for t in ctexts:
        h.update(t.encode())
    fp = h.hexdigest()
    fp_file = ckdir / "fingerprint.txt"
    if fp_file.exists() and fp_file.read_text(encoding="utf-8").strip() != fp:
        print("fingerprint changed -> discarding stale chunks", flush=True)
        for f in ckdir.glob("*.npy"):
            f.unlink()
    fp_file.write_text(fp, encoding="utf-8")

    # ---- stage 3: encode ------------------------------------------------
    m = SentenceTransformer(args.model)
    m.eval()
    enc = build_pooler(m)
    base = enc([u.text for u in units])

    print("encoding passage variants ...", flush=True)
    V = encode_ckpt(m, [t for _, _, t in jobs], ckdir, "main")
    print("encoding content-only strings ...", flush=True)
    CV = encode_ckpt(m, ctexts, ckdir, "cont")

    n_pairs = len(meta)
    cA = CV[:2 * n_pairs:2]
    cB = CV[1:2 * n_pairs:2]
    cV = CV[2 * n_pairs:]

    A_of = np.array([meta[j[0]]["A"] for j in jobs])
    B_of = np.array([meta[j[0]]["B"] for j in jobs])
    Sv = V @ base.T
    row0 = np.array([base[a] @ base.T for a in A_of])
    ok = np.ones((len(jobs), n), dtype=bool)
    ok[np.arange(len(jobs)), A_of] = False
    delta = Sv - row0
    dm = np.array([delta[k][ok[k]].mean() for k in range(len(jobs))])
    ds = np.array([delta[k][ok[k]].std() for k in range(len(jobs))])
    zt = np.array([delta[k, B_of[k]] for k in range(len(jobs))])
    z = (zt - dm) / np.where(ds > 0, ds, np.nan)

    p_new = np.array([float(V[k] @ base[B_of[k]]) for k in range(len(jobs))])
    p_old = np.array([meta[j[0]]["cos0"] for j in jobs])
    c_new = np.array([float(cV[k] @ cB[j[0]]) for k, j in enumerate(jobs)])
    c_old = np.array([float(cA[j[0]] @ cB[j[0]]) for j in jobs])
    g_new = np.array([float((V[k] @ base.T)[ok[k]].mean())
                      for k in range(len(jobs))])
    g_old = np.array([float((base[a] @ base.T)[ok[k]].mean())
                      for k, a in enumerate(A_of)])

    def rank_of(sim_row, a, b):
        r = [x for x in np.argsort(-sim_row) if x != a]
        return r.index(b) + 1, float(sim_row[r[0]] - sim_row[b])

    rank_new, margin_new = [], []
    for k in range(len(jobs)):
        rk, mg = rank_of(Sv[k], A_of[k], B_of[k])
        rank_new.append(rk)
        margin_new.append(mg)
    S0 = base @ base.T                      # similarity rows of the originals
    rank_old, margin_old = [], []
    for a, b in zip(A_of, B_of):
        rk, mg = rank_of(S0[a], a, b)
        rank_old.append(rk)
        margin_old.append(mg)
    rank_old = np.array(rank_old)
    margin_old = np.array(margin_old)
    rank_new = np.array(rank_new)
    margin_new = np.array(margin_new)

    # ---- report ---------------------------------------------------------
    OUT = []
    def w(*a):
        OUT.append(" ".join(str(x) for x in a))

    ks_of = {lbl: [k for k, j in enumerate(jobs) if j[1] == lbl]
             for lbl, _ in LEVELS}

    w("=" * 78)
    w("GRADED CONTENT DISTANCE — does word meaning reach the passage vector?")
    w("=" * 78)
    w(f"  model {args.model};  scope {args.scope};  K = {args.K}")
    w(f"  population: top {n_pairs} pairs with gap >= 5;  units {n}")
    w("")
    w("  intervention: every content word of every paragraph of A that")
    w("                shares any content word with B is replaced.")
    w("                Function words, punctuation, word order, sentence")
    w("                and paragraph breaks are held fixed.")
    w(f"  replaced per pair: mean {np.mean([m_['n_rep'] for m_ in meta]):.0f} "
      f"content words of {np.mean([m_['n_tok'] for m_ in meta]):.0f} tokens, "
      f"across {np.mean([m_['n_paras'] for m_ in meta]):.1f} paragraphs")
    w("")
    w("  CHECK — was the shared content actually destroyed?")
    w("    (content-only cosine, v7's definition; lower = more destroyed)")
    w(f"    {'level':<18}{'before':>9}{'after':>9}{'drop':>9}")
    for lbl, _ in LEVELS:
        ks = ks_of[lbl]
        w(f"    {lbl:<18}{np.mean(c_old[ks]):>9.3f}{np.mean(c_new[ks]):>9.3f}"
          f"{np.mean(c_old[ks] - c_new[ks]):>9.3f}")
    w("")
    w("  RESULT — the passage-level link")
    w(f"    {'level':<18}{'z':>7}{'cos to B':>10}{'drop':>8}"
      f"{'rank of B':>11}{'margin':>8}")
    w(f"    {'(original)':<18}{0.0:>7.2f}{np.mean(p_old):>10.3f}"
      f"{0.0:>8.3f}{np.mean(rank_old):>11.1f}{np.mean(margin_old):>8.3f}")
    for lbl, _ in LEVELS:
        ks = ks_of[lbl]
        w(f"    {lbl:<18}{np.mean(z[ks]):>7.2f}{np.mean(p_new[ks]):>10.3f}"
          f"{np.mean(p_old[ks] - p_new[ks]):>8.3f}"
          f"{np.mean(rank_new[ks]):>11.1f}{np.mean(margin_new[ks]):>8.3f}")
    w("")
    w("  DIAGNOSTIC — does the variant become generic?")
    w("    (mean cosine to every other unit, excluding A)")
    w(f"    {'level':<18}{'before':>9}{'after':>9}{'shift':>9}")
    for lbl, _ in LEVELS:
        ks = ks_of[lbl]
        w(f"    {lbl:<18}{np.mean(g_old[ks]):>9.3f}{np.mean(g_new[ks]):>9.3f}"
          f"{np.mean(g_new[ks] - g_old[ks]):>+9.3f}")
    w("")
    w("  SUMMARY")
    w("    level          " + "".join(f"{lbl.split()[0]:>14}"
                                       for lbl, _ in LEVELS))
    w("    z              " + "".join(
        f"{np.mean(z[ks_of[lbl]]):>+14.2f}" for lbl, _ in LEVELS))
    w("    raw drop to B  " + "".join(
        f"{np.mean(p_old[ks_of[lbl]] - p_new[ks_of[lbl]]):>14.3f}"
        for lbl, _ in LEVELS))
    w("    content drop   " + "".join(
        f"{np.mean(c_old[ks_of[lbl]] - c_new[ks_of[lbl]]):>14.3f}"
        for lbl, _ in LEVELS))
    w("    mean rank of B " + "".join(
        f"{np.mean(rank_new[ks_of[lbl]]):>14.1f}" for lbl, _ in LEVELS))
    w(f"    rank of B with no intervention: {np.mean(rank_old):.1f}")
    w("")

    out = DERIVED / f"graded_content_distance_{args.tag}_{args.scope}.txt"
    out.write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
