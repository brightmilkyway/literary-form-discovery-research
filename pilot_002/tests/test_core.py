from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import common as C
import geometry as G


def test_partition_grid_and_roundtrip():
    body, _ = C.load_body()
    partitions = C.build_partitions(body)
    assert len(partitions) == 22
    checks = C.validate_partitions(body, partitions)
    assert all(row["ok"] for row in checks)
    assert len(partitions["sentence"]) > len(partitions["paragraph"])
    assert partitions["w100_p25"][0].word_start == 25


def test_percentiles_and_stripe_detection():
    units = [C.Unit(i, "synthetic", "w050", 0, i * 10, i * 10 + 9,
                    i * 50, i * 50 + 50, "word " * 50) for i in range(12)]
    S = np.zeros((12, 12), dtype=float)
    np.fill_diagonal(S, 1)
    # A three-cell off-diagonal stripe at offset 6.
    for i in range(3):
        S[i, i + 6] = S[i + 6, i] = 1.0
    valid = np.triu(np.ones((12, 12), dtype=bool), k=1)
    P = G.percentile_matrix(S, valid)
    stripes = G.stripe_events(P, valid, units, "synthetic", "word_tfidf")
    assert stripes
    assert max(row["cells"] for row in stripes) >= 3


def test_remote_mask_uses_word_coordinates():
    units = [C.Unit(i, "s", "w050", 0, i, i + 1, i * 100,
                    i * 100 + 50, "x") for i in range(10)]
    mask = G.valid_remote_mask(units, 1000)
    assert not mask[0, 3]  # gap 250 < 300
    assert mask[0, 4]      # gap 350 >= 300

