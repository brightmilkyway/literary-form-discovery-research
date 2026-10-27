"""Check the two replacements the review asked for.

1. Proper nouns by a real NER model instead of a capitalisation heuristic.
2. Replacement words drawn from the encoder's own embedding matrix instead of
   a hand-picked filler list.
"""

from __future__ import annotations

import re
from collections import Counter

import common as C

body = C.strip_title(C.load_raw())[0]

# ---------------------------------------------------------------- 1. NER
import spacy

nlp = spacy.load("en_core_web_sm")
print(f"spaCy {spacy.__version__}, model en_core_web_sm, "
      f"pipes: {nlp.pipe_names}")

# spaCy handles ~1M chars; the story is 86k so one pass is fine
doc = nlp(body)
print(f"tokens: {len(doc)}, ents: {len(doc.ents)}")

c = Counter((e.label_, e.text) for e in doc.ents)
print("\ntop PERSON entities:")
for (lab, txt), n in c.most_common(60):
    if lab == "PERSON":
        print(f"   {txt!r:<22} x{n}")

print("\nother frequent entity labels:")
for lab in ("GPE", "LOC", "ORG", "FAC", "NORP", "WORK_OF_ART", "DATE", "TIME"):
    items = [(t, n) for (l, t), n in c.items() if l == lab]
    items.sort(key=lambda x: -x[1])
    if items:
        print(f"   {lab}: {items[:8]}")

# POS-based proper nouns as a cross-check
propn = Counter(t.text for t in doc if t.pos_ == "PROPN")
print(f"\nPROPN tokens: {len(propn)} types; top 25: "
      f"{[t for t, _ in propn.most_common(25)]}")

# compare with the heuristic the review rejected
STOP = set("""a an the and or but if of to in on at for with by from as is was
were be been being am are he she it they them his her their its i you we me my
your yours that this these those not no nor so do did does done had has have
having will would shall should can could may might must there here then than
when where which who whom what how all any both each few more most other some
such only own same too very just now up down out off over under again further
once about into through during before after above below between against because
until while yes o oh well""".split())
cap: Counter = Counter()
tot: Counter = Counter()
for tok in re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", body):
    low = tok.lower()
    if low in STOP or len(low) < 2:
        continue
    tot[low] += 1
    if tok[:1].isupper():
        cap[low] += 1
heur = {t for t in tot if tot[t] >= 2 and cap.get(t, 0) / tot[t] >= 0.8}
ner = {e.text.lower() for e in doc.ents if e.label_ == "PERSON"}
print(f"\nheuristic names: {len(heur)}   NER PERSON types: {len(ner)}")
print(f"  in NER but not heuristic: {sorted(ner - heur)[:15]}")
print(f"  in heuristic but not NER: {sorted(heur - ner)[:25]}")

# ------------------------------------------------- 2. embedding matrix
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("all-MiniLM-L6-v2")
emb_layer = m[0].auto_model.get_input_embeddings()
W = emb_layer.weight.detach().cpu().numpy()
print(f"\nembedding matrix: {W.shape}")
tok = m.tokenizer
vocab = tok.get_vocab()
words = [w for w in vocab
         if re.fullmatch(r"[a-z]{3,}", w) and not w.startswith("##")]
print(f"whole alphabetic words of length>=3 in the tokenizer vocab: "
      f"{len(words)}")
print(f"  examples: {words[:20]}")
