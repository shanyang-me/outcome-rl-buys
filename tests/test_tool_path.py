"""The tool path end to end on synthetic outputs: score -> rows -> analyze, with a minimal config and no paper-specific keys."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from pairaudit.cli import main


def _write_outputs(h, rng, n_items: int, p_fin: float, p_right: float, kind: str, prefix: str) -> None:
    """Synthetic outputs in the ``pairaudit score`` input shape: a boxed answer, a finish reason and a token length."""
    for i in range(n_items):
        for k in range(8):
            fin = rng.random() < p_fin
            right = fin and rng.random() < p_right
            h.write(json.dumps({"id": f"{prefix}{i}", "kind": kind, "k": k, "text": "\\boxed{1}" if right else "\\boxed{2}",
                                "finish_reason": "stop" if fin else "length", "len": int(rng.integers(500, 6000)) if fin else 8192}) + "\n")


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    rng = np.random.default_rng(0)
    items = tmp_path / "items.jsonl"
    with open(items, "w") as h:
        for i in range(60):
            h.write(json.dumps({"id": f"p{i}", "gt": json.dumps({"equiv_mode": "numeric", "golds": ["1"]}), "kind": "phys"}) + "\n")
        for i in range(40):
            h.write(json.dumps({"id": f"m{i}", "gt": json.dumps({"equiv_mode": "numeric", "golds": ["1"]}), "kind": "math"}) + "\n")
    outputs = {}
    for model, p_fin, p_right in (("base", 0.7, 0.5), ("rl", 0.9, 0.6)):
        f = tmp_path / f"{model}.jsonl.gz"
        with gzip.open(f, "wt") as h:
            _write_outputs(h, rng, 60, p_fin, p_right, "phys", "p")
            _write_outputs(h, rng, 40, p_fin, p_right, "math", "m")
        outputs[model] = f
    config = tmp_path / "pairs.json"
    config.write_text(json.dumps({
        "bases": {"base": {"label": "Base-7B"}},
        "pairs": {"rl": {"model": "rl", "base": "base", "train_cap": 4096}},
        "budgets": {"ladder": [2048, 4096, 8192]},
    }))
    return {"tmp": tmp_path, "items": items, "outputs": outputs, "config": config, "rows": tmp_path / "rows"}


def test_score_then_analyze(workspace):
    w = workspace
    for model, f in w["outputs"].items():
        assert main(["--config", str(w["config"]), "--rows", str(w["rows"]), "score", "--texts", str(f), "--items", str(w["items"]),
                     "--model", model, "--run", "myrun", "--cap", "8192"]) == 0
    manifest = json.loads((w["rows"] / "MANIFEST.json").read_text())
    assert manifest["myrun"]["nested"] and manifest["myrun"]["models"]["rl"]["caps"] == [2048, 4096, 8192]
    rows = [json.loads(l) for l in gzip.open(w["rows"] / "myrun" / "rl_cap8192.jsonl.gz", "rt")]
    assert len(rows) == 100 * 8 and all(set(r) == {"id", "kind", "k", "finished", "score", "len"} for r in rows)
    assert all(r["score"] == 0.0 for r in rows if not r["finished"])

    out = w["tmp"] / "out"
    assert main(["--config", str(w["config"]), "--rows", str(w["rows"]), "analyze", "--out", str(out), "--boot", "50", "--cell-boot", "200"]) == 0
    for name in ("cap_split", "ledger", "cells", "cap_cells", "stats", "report"):
        assert (out / f"{name}.{'md' if name == 'report' else 'json'}").exists()
    led = json.loads((out / "ledger.json").read_text())
    assert led["kinds"] == ["phys", "math"]           # first appearance in the rows, since the config names none
    assert led["fixed_budgets"] == [2048, 4096, 8192]   # every shared budget, since the config names none
    assert led["cap"]["at"]["recipes"]["rl|phys"]["cap"] == 4096
    assert led["cap"]["beyond"]["recipes"]["rl|phys"]["cap"] == 8192
    split = json.loads((out / "cap_split.json").read_text())
    assert split["kind"] == "phys" and split["recipes"]["rl"]["at"]["F"] > 0  # the RL model finishes more, by construction
    pooled = led["cap"]["at"]["pooled"]
    assert abs(pooled["F"] + pooled["P"] + pooled["reach"] - 1) < 1e-9  # size shares sum to one
    report = (out / "report.md").read_text()
    assert "Ledger" in report and "Base-7B" in report


def test_custom_scorer(workspace, tmp_path):
    w = workspace
    plug = tmp_path / "myscorer.py"
    plug.write_text("def exact(text, gt):\n    return 1.0 if text.strip() == '\\\\boxed{1}' else 0.0\n")
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        assert main(["--rows", str(w["rows"]), "--config", str(w["config"]), "score", "--texts", str(w["outputs"]["rl"]), "--items", str(w["items"]),
                     "--model", "rl", "--run", "plug", "--scorer", "myscorer:exact"]) == 0
    finally:
        sys.path.remove(str(tmp_path))
    rows = [json.loads(l) for l in gzip.open(w["rows"] / "plug" / "rl_cap8192.jsonl.gz", "rt")]
    assert {r["score"] for r in rows} <= {0.0, 1.0}
