"""The lock queue and the restart-safe partial file, without a GPU."""

from __future__ import annotations

import json

from pairaudit.generate import queue
from pairaudit.generate.items import load_spec


def test_claim_is_exclusive_and_done_models_are_skipped(tmp_path):
    assert queue.claim(tmp_path, "m", host="a")
    assert not queue.claim(tmp_path, "m", host="b")
    assert queue.lock_owner(tmp_path, "m") == "a"
    queue.release(tmp_path, "m")
    queue.write_json(queue.summary_path(tmp_path, "m"), {})
    assert not queue.claim(tmp_path, "m", host="a")


def test_partial_resume_drops_cut_lines_and_stale_ids(tmp_path):
    items = [{"id": "p0"}, {"id": "p1"}, {"id": "p2"}]
    p = queue.partial_path(tmp_path, "m")
    good0 = {"i": 0, "id": "p0", "samples": [{}, {}]}
    stale = {"i": 1, "id": "OLD", "samples": [{}, {}]}
    short = {"i": 2, "id": "p2", "samples": [{}]}
    p.write_text(json.dumps(good0) + "\n" + json.dumps(stale) + "\n" + json.dumps(short) + "\n" + '{"i": 2, "id": "p2", "sam')
    done = queue.load_partial(p, items, n=2)
    assert list(done) == [0]
    assert p.read_text() == json.dumps(good0) + "\n"


def test_spec_defaults_and_json_string():
    spec = load_spec('{"A": [1.0, 1.0, 49152]}', default=(0.6, 0.95, 16384))
    assert spec["A"] == (1.0, 1.0, 49152)
    assert spec["B"] == (0.6, 0.95, 16384)
