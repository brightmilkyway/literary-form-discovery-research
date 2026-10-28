"""Show, concretely, what the v14 probe did to a passage versus what the
corrected design does.

Used only to document the correction: the v14 probe replaced every word,
function words included, so the conclusion drawn from it ('what survives is
form') was not supported by its own design.

Writes ``data/derived/probe_design_correction.txt``.
"""

import random

import common as C
from form_vs_content import is_content
from ladder_principled import Replacer

OUT = []
def w(*a):
    OUT.append(" ".join(str(x) for x in a))

C.set_seed(42)
body = C.strip_title(C.load_raw())[0]
paras = C.paragraphs(body)

# a paragraph with plenty of dialogue and function words
p = paras[40]
words = p.text.split()
w("=" * 78)
w("WHAT THE v14 PROBE ACTUALLY DID TO A PASSAGE")
w("=" * 78)
w("")
w("ORIGINAL")
w("-" * 78)
w(" ".join(words))
w("")

rep = Replacer("all-MiniLM-L6-v2")

# --- what the v14 probe did: every word ---------------------------------
rng = random.Random(42)
v14 = [rep.pick(x, set(), rng) for x in words]
w("v14 PROBE — every word replaced (function words included)")
w("-" * 78)
w(" ".join(v14))
w("")
kept = [x for x in words if not is_content(x)]
w(f"  function words surviving : "
  f"{sum(1 for a, b in zip(words, v14) if not is_content(a) and a == b)}"
  f" of {len(kept)}")
w("")

# --- what the corrected design does: content words only -----------------
rng = random.Random(42)
new = list(words)
n_rep = 0
for i, x in enumerate(words):
    if is_content(x):
        wn = rep.pick(x, set(), rng)
        if wn:
            new[i] = wn
            n_rep += 1
w("CORRECTED — content words only (function words held fixed)")
w("-" * 78)
w(" ".join(new))
w("")
w(f"  content words replaced   : {n_rep} of "
  f"{sum(1 for x in words if is_content(x))}")
w(f"  function words surviving : "
  f"{sum(1 for a, b in zip(words, new) if not is_content(a) and a == b)}"
  f" of {len(kept)}")
w("")
w("Reading: the v14 probe left no grammar at all, so the claim that what")
w("survives its intervention is 'form - syntax, function words, register'")
w("was not supported by its own design.  The corrected run holds the")
w("function words fixed and moves only the content words, which is what")
w("the question requires.")
w("")

path = C.DATA_DIR / "derived" / "probe_design_correction.txt"
path.write_text("\n".join(OUT), encoding="utf-8")
print("\n".join(OUT))
print("wrote", path)

