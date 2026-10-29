from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import common as C
import interpretive as I
import run_pipeline as P
import predictive as PR


def test_checkpoint_grid_is_prefix_only_and_covers_source():
    body, _ = C.load_body()
    paras = C.paragraphs(body)
    checkpoints = C.build_checkpoints(body, paras, 40)
    assert len(checkpoints) == 40
    assert checkpoints[0]["paragraph_index"] == 0
    assert checkpoints[-1]["paragraph_index"] == len(paras) - 1
    assert all(a["char_end"] < b["char_end"] for a, b in zip(checkpoints, checkpoints[1:]))
    for checkpoint in checkpoints:
        prefix = body[:checkpoint["char_end"]]
        assert hashlib.sha256(prefix.encode("utf-8")).hexdigest() == checkpoint["prefix_sha256"]


def test_fixed_hash_representation_is_deterministic():
    texts = ["alpha beta gamma", "alpha beta delta"]
    first = P.vectorizer("word_hash", 1024).transform(texts).toarray()
    second = P.vectorizer("word_hash", 1024).transform(texts).toarray()
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_scales_end_at_checkpoint_and_do_not_include_future_text():
    body, _ = C.load_body()
    paras = C.paragraphs(body)
    checkpoints = C.build_checkpoints(body, paras, 40)
    for scale in ("p1", "p4", "cumulative"):
        texts = P.texts_at_scale(body, paras, checkpoints, scale)
        for text, checkpoint in zip(texts, checkpoints):
            assert text.endswith(paras[checkpoint["paragraph_index"]].text)
            assert text in body[:checkpoint["char_end"]]


def test_interpretive_requests_contain_exact_prefix_and_frozen_prompt():
    path = I.prepare()
    rows = C.read_jsonl(path)
    body, _ = C.load_body()
    checkpoints = C.build_checkpoints(body, C.paragraphs(body), 40)
    assert len(rows) == len(checkpoints)
    assert len({row["prompt_sha256"] for row in rows}) == 1
    for row, checkpoint in zip(rows, checkpoints):
        prefix = body[:checkpoint["char_end"]]
        assert row["input"]["user"] == I.USER_TEMPLATE.format(
            checkpoint_id=checkpoint["checkpoint_id"], prefix=prefix)
        assert row["status"] == "prepared_not_run"


def test_predictive_chunks_cover_every_token_once_with_bounded_context():
    plan = PR.chunk_plan(2501, maximum=1024, stride=512)
    targets = [i for _, start, end in plan for i in range(start, end)]
    assert targets == list(range(1, 2501))
    assert all(end - context <= 1024 for context, _, end in plan)
    assert all(context < start < end for context, start, end in plan)
