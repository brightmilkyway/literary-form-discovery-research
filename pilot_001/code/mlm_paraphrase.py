"""Does the passage link depend on WHICH words are there, or on whether the
text stays coherent?

Why this exists
  v15 replaced content words with words drawn at controlled semantic distance
  from the encoder's own embedding matrix.  v16 then showed that this moved the
  text a long way towards the word-scrambled pole - at the furthest level the
  variants were closer to the scrambled pole than to the real one.  So the
  manipulation was not content-only: it changed word identity AND lexical
  coherence together.

  A masked language model separates them.  Fill each blank with the word the
  MLM finds most plausible GIVEN THE ORIGINAL CONTEXT: the word identity
  changes, the text stays coherent.

Arms
  MLM-top1    the most plausible non-original word at that position
  MLM-top10   a random choice among the ten most plausible
  MLM-tail    a random word from ranks 1000-3000 - plausible as a word, not as
              a continuation.  This is the matched "incoherent" control: same
              positions, same number of replacements, only the plausibility
              differs.

  All replacements are conditioned on the ORIGINAL text: each quarter of the
  positions is masked in turn, and the unmasked positions always show the
  original words.  So every replacement is "plausible in the original context"
  rather than "plausible in a half-rewritten text".

Reference numbers (already measured, v15, same population):
  q=0.99 near       partner-specific 0.039 / 0.019
  q=0.00 furthest   partner-specific 0.059 / 0.022
  SHUF              partner-specific 0.020 / 0.009

Reading
  * MLM-top10 as damaging as q=0.00 furthest  ->  word identity is what matters
  * MLM-top10 far less damaging               ->  coherence is what matters

Writes ``data/derived/mlm_paraphrase_<tag>.txt``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np

import common as C
from form_vs_content import content_only, is_content
from ladder_principled import build_pooler, ctypes

DERIVED = C.DATA_DIR / "derived"
CKPT = DERIVED / "_ckpt"
MLM = "distilroberta-base"
ARMS = ("MLM-top1", "MLM-top10", "MLM-tail")
MIN_GAP = 5
WORD_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z][A-Za-z'\u2019-]*)([^A-Za-z]*)$")


def split_punct(tok: str):
    m = WORD_RE.match(tok)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", tok, "")


def build_pairs(units, n_pairs):
    import torch  # noqa: F401  (build_pooler needs it loaded)
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("all-MiniLM-L6-v2")
    m.eval()
    enc = build_pooler(m)
    base = enc([u.text for u in units])
    iu, ju = np.triu_indices(len(units), k=1)
    S = base @ base.T
    cand = sorted(((float(S[i, j]), int(i), int(j))
                   for i, j in zip(iu, ju) if j - i >= MIN_GAP),
                  key=lambda t: -t[0])
    return cand[:n_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--quarters", type=int, default=4)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    paras = C.paragraphs(body)

    ckdir = CKPT / "mlm_paraphrase"
    ckdir.mkdir(parents=True, exist_ok=True)
    jobs_file = ckdir / "jobs.jsonl"

    if jobs_file.exists():
        jobs = [json.loads(x) for x in
                jobs_file.read_text(encoding="utf-8").splitlines()]
        print(f"resumed: {len(jobs)} variants")
    else:
        pairs = build_pairs(units, args.pairs)
        print(f"{len(pairs)} pairs selected")

        tok = AutoTokenizer.from_pretrained(MLM)
        model = AutoModelForMaskedLM.from_pretrained(MLM)
        model.eval()
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        print("MLM loaded:", MLM)

        # ---- plan: which words get replaced in each pair ----------------
        plan = []          # (pair_idx, para_idx, [word indices])
        meta = []
        for pi, (cos0, A, B) in enumerate(pairs):
            u = units[A]
            btypes = ctypes(units[B].text)
            hits = {}
            for q_ in paras[u.first_para:u.last_para + 1]:
                if ctypes(q_.text) & btypes:
                    idxs = [i for i, t in enumerate(q_.text.split())
                            if is_content(split_punct(t)[1])]
                    if idxs:
                        hits[q_.index] = idxs
            plan.append((pi, A, B, cos0, hits))
            meta.append(dict(A=A, B=B, cos0=float(cos0),
                             n_rep=sum(len(v) for v in hits.values())))

        # ---- generate the three arms -----------------------------------
        rng = random.Random(42)
        t0 = time.time()

        # current state of every paragraph being edited, per arm
        state = {a: {} for a in ARMS}
        for pi, A, B, cos0, hits in plan:
            for idx in hits:
                for a in ARMS:
                    state[a][(pi, idx)] = paras[idx].text.split()

        for qi in range(args.quarters):
            # for each (pair, para) collect the word indices of this quarter
            todo = []
            for pi, A, B, cos0, hits in plan:
                for idx, idxs in hits.items():
                    q = idxs[qi::args.quarters]
                    if q:
                        todo.append((pi, idx, q))
            if not todo:
                continue
            print(f"  quarter {qi + 1}/{args.quarters}: "
                  f"{len(todo)} paragraphs", flush=True)

            # build the masked texts once per (pair, para, quarter)
            texts, owners = [], []
            for pi, idx, q in todo:
                words = list(paras[idx].text.split())
                for i in q:
                    pre, _, post = split_punct(words[i])
                    words[i] = pre + "<mask>" + post
                texts.append(" ".join(words))
                owners.append((pi, idx, q))

            enc_b = tok(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
            # The LM head emits (batch, seq, 50265).  Keeping those for every
            # batch needs ~20 GB, so each batch is consumed and released
            # immediately: only the masked positions are ever looked at.
            with torch.no_grad():
                for a in range(0, len(texts), args.batch):
                    ids = enc_b["input_ids"][a:a + args.batch]
                    out = model(input_ids=ids,
                                attention_mask=enc_b["attention_mask"][
                                    a:a + args.batch]).logits
                    for k in range(out.shape[0]):
                        row_i = a + k
                        pi, idx, q = owners[row_i]
                        row = tok.convert_ids_to_tokens(ids[k].tolist())
                        mask_pos = [i for i, t in enumerate(row)
                                    if t == "<mask>"]
                        if len(mask_pos) != len(q):
                            print(f"    !! mask/word mismatch "
                                  f"{len(mask_pos)} vs {len(q)} at pair {pi} "
                                  f"para {idx}, skipping")
                            continue
                        for pos, wi in zip(mask_pos, q):
                            probs = torch.softmax(out[k, pos], dim=-1)
                            top = torch.topk(probs, 3000)
                            ranked = [tok.convert_ids_to_tokens(int(x))
                                      for x in top.indices.tolist()]
                            orig = split_punct(
                                paras[idx].text.split()[wi])[1].lower()
                            cands = []
                            for r, wv in enumerate(ranked):
                                ww = wv.replace("\u0120", "").strip()
                                if (re.fullmatch(
                                        r"[A-Za-z][A-Za-z'\u2019-]+", ww)
                                        and ww.lower() != orig):
                                    cands.append((r, ww))
                            if not cands:
                                continue
                            for arm in ARMS:
                                if arm == "MLM-top1":
                                    pick = cands[0][1]
                                elif arm == "MLM-top10":
                                    pick = rng.choice(cands[:10])[1]
                                else:
                                    tail = [c for c in cands
                                            if 1000 <= c[0] < 3000]
                                    pick = (rng.choice(tail)[1] if tail
                                            else cands[-1][1])
                                st = state[arm][(pi, idx)]
                                pre, _, post = split_punct(st[wi])
                                st[wi] = pre + pick + post
                    del out
            del enc_b
            print(f"    forward passes done ({time.time()-t0:.0f}s)", flush=True)

        # ---- assemble the variant texts ---------------------------------
        jobs, meta_out = [], []
        for pi, A, B, cos0, hits in plan:
            for arm in ARMS:
                for _ in range(args.K):
                    parts = [paras[k].text
                             for k in range(units[A].first_para,
                                            units[A].last_para + 1)]
                    for idx in hits:
                        parts[idx - units[A].first_para] = \
                            " ".join(state[arm][(pi, idx)])
                    jobs.append([pi, arm, "\n\n".join(parts)])
            meta_out.append(dict(A=A, B=B, cos0=float(cos0),
                                 n_rep=sum(len(v) for v in hits.values())))
        ctexts = []
        for m_ in meta_out:
            ctexts += [content_only(units[m_["A"]].text),
                       content_only(units[m_["B"]].text)]
        ctexts += [content_only(t) for _, _, t in jobs]

        jobs_file.write_text("\n".join(json.dumps(j) for j in jobs),
                             encoding="utf-8")
        (ckdir / "meta.json").write_text(json.dumps(meta_out),
                                         encoding="utf-8")
        (ckdir / "ctexts.jsonl").write_text(
            "\n".join(json.dumps(t) for t in ctexts), encoding="utf-8")
        print(f"built {len(jobs)} variants, {len(ctexts)} content strings, "
              f"{time.time()-t0:.0f}s")

    # ---- encode and score, one model at a time --------------------------
    from sentence_transformers import SentenceTransformer

    jobs = [json.loads(x) for x in
            jobs_file.read_text(encoding="utf-8").splitlines()]
    meta = json.loads((ckdir / "meta.json").read_text(encoding="utf-8"))
    ctexts = [json.loads(x) for x in
              (ckdir / "ctexts.jsonl").read_text(encoding="utf-8").splitlines()]
    n_pairs = len(meta)
    print(f"variants {len(jobs)}, content strings {len(ctexts)}")

    CH = 256

    def encode_ckpt(m, texts, prefix):
        starts = list(range(0, len(texts), CH))
        for i, a in enumerate(starts):
            f = ckdir / f"{prefix}_{i:05d}.npy"
            if f.exists():
                continue
            with torch.no_grad():
                e = m.encode(texts[a:a + CH], batch_size=CH,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
            np.save(f, e.astype(np.float32))
            print(f"    {prefix} {i+1}/{len(starts)}", flush=True)
        return np.vstack([np.load(ckdir / f"{prefix}_{i:05d}.npy")
                          for i in range(len(starts))])

    for tag, model_name in (("minilm", "all-MiniLM-L6-v2"),
                            ("bge", "BAAI/bge-small-en-v1.5")):
        m = SentenceTransformer(model_name)
        m.eval()
        enc = build_pooler(m)
        base = enc([u.text for u in units])
        n = len(units)
        V = encode_ckpt(m, [t for _, _, t in jobs], f"{tag}_main")
        CV = encode_ckpt(m, ctexts, f"{tag}_cont")

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
        spec = np.array([delta[k, B_of[k]] for k in range(len(jobs))]) - dm
        p_old = np.array([meta[j[0]]["cos0"] for j in jobs])
        p_new = np.array([float(V[k] @ base[B_of[k]])
                          for k in range(len(jobs))])
        c_old = np.array([float(cA[j[0]] @ cB[j[0]]) for j in jobs])
        c_new = np.array([float(cV[k] @ cB[j[0]]) for k, j in enumerate(jobs)])

        def rank_of(row, a, b):
            r = [x for x in np.argsort(-row) if x != a]
            return r.index(b) + 1

        rank_old = np.array([rank_of(base[a] @ base.T, a, b)
                             for a, b in zip(A_of, B_of)])
        rank_new = np.array([rank_of(Sv[k], A_of[k], B_of[k])
                             for k in range(len(jobs))])

        OUT = []
        def w(*a):
            OUT.append(" ".join(str(x) for x in a))

        w("=" * 78)
        w("MLM PARAPHRASE — word identity vs lexical coherence")
        w("=" * 78)
        w(f"  model {model_name};  population top {n_pairs} pairs (gap >= {MIN_GAP})")
        w(f"  MLM {MLM};  replacements per pair mean "
          f"{np.mean([m_['n_rep'] for m_ in meta]):.0f}")
        w("")
        w("  CHECK — was the content destroyed? (content-only cosine)")
        w(f"    {'arm':<14}{'before':>9}{'after':>9}{'drop':>9}")
        for arm in ARMS:
            ks = [k for k, j in enumerate(jobs) if j[1] == arm]
            w(f"    {arm:<14}{np.mean(c_old[ks]):>9.3f}"
              f"{np.mean(c_new[ks]):>9.3f}"
              f"{np.mean(c_old[ks] - c_new[ks]):>9.3f}")
        w("")
        w("  RESULT")
        w(f"    {'arm':<14}{'cos to B':>10}{'drop':>8}{'rank of B':>11}"
          f"{'partner-specific':>18}")
        w(f"    {'(original)':<14}{np.mean(p_old):>10.3f}{0.0:>8.3f}"
          f"{np.mean(rank_old):>11.1f}{0.0:>18.3f}")
        for arm in ARMS:
            ks = [k for k, j in enumerate(jobs) if j[1] == arm]
            w(f"    {arm:<14}{np.mean(p_new[ks]):>10.3f}"
              f"{np.mean(p_old[ks] - p_new[ks]):>8.3f}"
              f"{np.mean(rank_new[ks]):>11.1f}"
              f"{np.mean(spec[ks]):>18.3f}")
        w("")
        w("  REFERENCE (v15, same population, embedding-quantile replacements)")
        w("    q=0.99 near      partner-specific 0.039 / 0.019")
        w("    q=0.00 furthest  partner-specific 0.059 / 0.022")
        w("    SHUF (order)     partner-specific 0.020 / 0.009")
        w("")

        out = DERIVED / f"mlm_paraphrase_{tag}.txt"
        out.write_text("\n".join(OUT), encoding="utf-8")
        print("\n".join(OUT))
        print("wrote", out)


if __name__ == "__main__":
    main()
