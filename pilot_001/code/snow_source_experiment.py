"""Where does the final snow passage's peripherality come from?

Unit 77 (paras 454) is the most peripheral unit in BOTH encoders
(MiniLM rank 2/78, bge rank 1/78).  Three competing explanations:

  H1  lexical field   — the winter/west/death vocabulary
  H2  form/register   — pure narration, no dialogue, long periodic sentences
                        with anaphora ("falling ... falling ... falling")
  H3  length          — it is 160 words (SBERT/length r = 0.027, so unlikely)

Design: modify ONE unit, re-encode it, and measure its centrality against the
*fixed* embeddings of all other units.  Then compare with the same ablations
applied to control units (other pure-narration passages, and the most central
units) to see how much movement is normal.

Variants per unit
  original          the text as is
  no_topic          weather + west/death lexicon replaced by neutral fillers
  no_weather        only the weather lexicon replaced
  no_death_west     only the west/death lexicon replaced
  no_content        every content word replaced (form-only)
  bag_of_words      word order shuffled inside each sentence (syntax kept,
                    order destroyed)
  shuffled_sent     sentence order shuffled (local syntax kept)

Writes ``data/derived/snow_source_experiment.txt``.
"""

from __future__ import annotations

import json
import random
import re

import numpy as np

import common as C

DERIVED = C.DATA_DIR / "derived"
OUT = []

WEATHER = {"snow", "flakes", "falling", "fell", "pane", "window", "taps",
           "lamplight", "silver", "dark", "cold", "air"}
DEATH_WEST = {"journey", "westward", "west", "ireland", "plain", "hills",
              "bog", "allen", "shannon", "waves", "churchyard", "michael",
              "furey", "buried", "crosses", "headstones", "spears", "gate",
              "thorns", "soul", "swooned", "universe", "descent", "dead",
              "living", "newspapers", "treeless", "barren", "obliquely"}
STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while yes o oh well said asked cried says say""".split())

TARGET = 77
CONTROLS = [63, 57, 59, 61, 0, 17, 74, 75, 76]      # pure-narration units
CENTRAL = [39, 72, 24, 48, 14]                       # most central units


def w(*a):
    OUT.append(" ".join(str(x) for x in a))


def prev(t, n=110):
    s = re.sub(r"\s+", " ", t).strip()
    return s[:n] + ("…" if len(s) > n else "")


def types_of(text):
    return {m.group(0) for m in re.finditer(r"[A-Za-z][A-Za-z'\u2019-]*", text)}


def replace(text, kill, filler="thing"):
    ks = {k.lower() for k in kill}
    return re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                  lambda m: filler if m.group(0).lower() in ks else m.group(0),
                  text)


def content_types(text):
    return {t for t in types_of(text) if t.lower().strip("'") not in STOP}


def variant(text, kind, rng):
    if kind == "original":
        return text
    if kind == "no_topic":
        return replace(text, WEATHER | DEATH_WEST)
    if kind == "no_weather":
        return replace(text, WEATHER)
    if kind == "no_death_west":
        return replace(text, DEATH_WEST)
    if kind == "no_content":
        return replace(text, content_types(text))
    if kind == "bag_of_words":
        out = []
        for sent in re.split(r"(?<=[.!?…])\s+", text):
            toks = sent.split()
            rng.shuffle(toks)
            out.append(" ".join(toks))
        return " ".join(out)
    if kind == "shuffled_sent":
        sents = [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
        rng.shuffle(sents)
        return " ".join(sents)
    raise ValueError(kind)


KINDS = ["original", "no_weather", "no_death_west", "no_topic", "no_content",
         "bag_of_words", "shuffled_sent"]


def main():
    C.set_seed(42)
    body = C.strip_title(C.load_raw())[0]
    units = C.build_units(body, "group", target_words=150)
    n = len(units)
    paras = C.paragraphs(body)

    import torch
    from sentence_transformers import SentenceTransformer

    def pooled_encoder(model_name, max_words=130):
        """Paragraph-aware pooling — identical to fix_representations.py."""
        m = SentenceTransformer(model_name)
        m.eval()

        def split_paras(u):
            buf, cnt, out = [], 0, []
            for p in paras[u.first_para:u.last_para + 1]:
                if p.n_words > max_words:
                    toks = p.text.split()
                    for k in range(0, len(toks), max_words):
                        out.append(" ".join(toks[k:k + max_words]))
                    continue
                if cnt + p.n_words > max_words and buf:
                    out.append("\n\n".join(buf))
                    buf, cnt = [], 0
                buf.append(p.text)
                cnt += p.n_words
            if buf:
                out.append("\n\n".join(buf))
            return out

        def enc(texts, owners):
            sub, owner = [], []
            for k, t in enumerate(texts):
                if owners is not None and owners[k] is not None:
                    for piece in split_paras(owners[k]):
                        sub.append(piece)
                        owner.append(k)
                else:
                    words = t.split()
                    for i in range(0, len(words), max_words):
                        sub.append(" ".join(words[i:i + max_words]))
                        owner.append(k)
            with torch.no_grad():
                e = m.encode(sub, batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
            out = np.zeros((len(texts), e.shape[1]))
            for k, o in enumerate(owner):
                out[o] += e[k]
            out /= np.linalg.norm(out, axis=1, keepdims=True)
            return out
        return enc

    models = {
        "MiniLM-pooled": pooled_encoder("all-MiniLM-L6-v2"),
        "bge-small": pooled_encoder("BAAI/bge-small-en-v1.5"),
    }

    w("=" * 78)
    w("WHERE DOES THE FINAL SNOW PASSAGE'S PERIPHERALITY COME FROM?")
    w("=" * 78)
    w()
    w(f"target unit {TARGET} (paras {units[TARGET].first_para}-"
      f"{units[TARGET].last_para}, {units[TARGET].n_words} words):")
    w(f"   {prev(units[TARGET].text, 320)}")
    w()
    w("Note: peripherality is measured as the MEAN cosine to all other units.")
    w("Lower = more peripheral.  Ranks are out of 78.")
    w()

    for mname, enc in models.items():
        base = enc([u.text for u in units], units)
        S = base @ base.T
        np.fill_diagonal(S, np.nan)
        cent0 = np.nanmean(S, axis=1)
        order = np.argsort(cent0)
        rank0 = {int(i): int(np.where(order == i)[0][0]) + 1 for i in range(n)}
        w("-" * 74)
        w(f"MODEL: {mname}   (global mean centrality {np.nanmean(cent0):.3f})")
        w("-" * 74)
        w(f"   original centrality of unit {TARGET}: {cent0[TARGET]:.3f}  "
          f"rank {rank0[TARGET]}/78")
        w()

        def central(text):
            e = enc([text], [None])
            return float((e @ base.T).ravel()[np.arange(n) != TARGET].mean())

        for label, group in (("TARGET (final snow)", [TARGET]),
                             ("CONTROLS (other pure narration)",
                              [c for c in CONTROLS if c != TARGET]),
                             ("MOST CENTRAL UNITS", CENTRAL)):
            w(f"   --- {label} ---")
            w(f"   {'unit':>5} {'original':>9} " +
              "".join(f"{k:>14}" for k in KINDS[1:]))
            for u in group:
                rng = random.Random(42)
                row = [f"{cent0[u]:>9.3f}"]
                for kind in KINDS[1:]:
                    t = variant(units[u].text, kind, rng)
                    c = float((enc([t], [None]) @ base.T).ravel()[
                        np.arange(n) != u].mean())
                    row.append(f"{c:>14.3f}")
                w(f"   {u:>5} " + "".join(row))
            w()

        # ---- MATCHED CONTROL: same NUMBER of ablated content-word types ----
        w("   --- MATCHED RANDOM-ABLATION CONTROL for unit "
          f"{TARGET} ---")
        n_topic = len({t.lower() for t in types_of(units[TARGET].text)
                       if t.lower() in (WEATHER | DEATH_WEST)})
        w(f"   the topic lexicon hits {n_topic} word types in unit {TARGET};")
        w(f"   control: ablate {n_topic} RANDOM content-word types, 8 reps")
        cts = sorted(content_types(units[TARGET].text))
        drops = []
        for rep in range(8):
            rng = random.Random(100 + rep)
            pick = rng.sample(cts, min(n_topic, len(cts)))
            c = central(replace(units[TARGET].text, pick))
            drops.append(c)
        drops = np.array(drops)
        c_topic = central(variant(units[TARGET].text, "no_topic",
                                  random.Random(42)))
        w(f"      random-ablation centrality: mean {drops.mean():.3f}, "
          f"sd {drops.std():.3f}, range [{drops.min():.3f}, {drops.max():.3f}]")
        w(f"      topic-lexicon ablation   : {c_topic:.3f}")
        pct = float((drops < c_topic).mean()) * 100
        w(f"      -> topic ablation sits at the {pct:.0f}th percentile of the")
        w(f"         random-ablation distribution.")
        w(f"      original centrality: {cent0[TARGET]:.3f}")
        w()

        # ---- TRANSPLANT: does the lexicon make a CENTRAL unit peripheral? --
        w("   --- TRANSPLANT CONTROL: does the snow/west/death lexicon make a")
        w("       central passage peripheral? ---")
        lexicon = sorted({t.lower() for t in types_of(units[TARGET].text)
                          if t.lower() in (WEATHER | DEATH_WEST)})
        for donor in (63, 57, 59):
            cts_d = sorted(content_types(units[donor].text))
            rng = random.Random(7)
            rng.shuffle(cts_d)
            n_swap = min(len(lexicon), len(cts_d))
            mapping = {cts_d[k].lower(): lexicon[k] for k in range(n_swap)}
            txt = re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                         lambda m: mapping.get(m.group(0).lower(), m.group(0)),
                         units[donor].text)
            c_tr = float((enc([txt], [None]) @ base.T).ravel()[
                np.arange(n) != donor].mean())
            # matched control: replace the same number of types with each other
            mapping2 = {cts_d[k].lower(): cts_d[(k + n_swap) % len(cts_d)].lower()
                        for k in range(n_swap)}
            txt2 = re.sub(r"[A-Za-z][A-Za-z'\u2019-]*",
                          lambda m: mapping2.get(m.group(0).lower(), m.group(0)),
                          units[donor].text)
            c_ct = float((enc([txt2], [None]) @ base.T).ravel()[
                np.arange(n) != donor].mean())
            w(f"      unit {donor}: original {cent0[donor]:.3f} | "
              f"+snow lexicon {c_tr:.3f} | shuffled-content control {c_ct:.3f}")
        w("      (If the lexicon caused peripherality, '+snow lexicon' should")
        w("       be clearly BELOW both the original and the control.)")
        w()
        w(f"   RANK of unit {TARGET} under each variant:")
        for kind in KINDS:
            rng = random.Random(42)
            t = variant(units[TARGET].text, kind, rng)
            c = central(t)
            r = int((cent0 < c).sum()) + 1
            w(f"      {kind:<16} centrality {c:.3f}   rank {r}/78")
        w()

    (DERIVED / "snow_source_experiment.txt").write_text("\n".join(OUT),
                                                        encoding="utf-8")
    print("wrote", DERIVED / "snow_source_experiment.txt")


if __name__ == "__main__":
    main()
