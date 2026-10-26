"""Fixed-budget cells: point estimates reproduce the paper's table exactly; intervals closely (different random stream)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pairaudit.analysis.cells import fixed_budget_cells
from pairaudit.analysis.registry import Registry
from pairaudit.analysis.rows import RowStore

ROOT = Path(__file__).resolve().parents[1]
GOLD = Path(__file__).resolve().parent / "golden" / "pairs_ci_all.json"


def test_fixed_budget_points_match_golden():
    reg = Registry.load()
    gold = json.loads(GOLD.read_text())
    out = fixed_budget_cells(RowStore(ROOT / "data/rows", reg.views["fixed"]), reg, n_boot=300)
    assert set(out) == set(gold)
    for key, g in gold.items():
        o = out[key]
        assert o["n_problems"] == g["n_problems"] and o["len_ratio"] == pytest.approx(g["len_ratio"])
        for q, gq in (("mean", "mean"), ("finished", "finished"), ("sgf", "sgf"), ("any", "any8")):
            assert o[q]["base"] == pytest.approx(g[gq]["base"], abs=1e-9), (key, q)
            assert o[q]["rl"] == pytest.approx(g[gq]["rl"], abs=1e-9), (key, q)
            assert o[q]["ci"][0] == pytest.approx(g[gq]["ci"][0], abs=0.02) and o[q]["ci"][1] == pytest.approx(g[gq]["ci"][1], abs=0.02), (key, q)
        assert o["finishing_term"] == pytest.approx(g["finishing_term"], abs=1e-9)
        assert o["matched"]["n"] == g["matched"]["n"]
        if "delta" in g["matched"]:
            assert o["matched"]["delta"] == pytest.approx(g["matched"]["delta"], abs=1e-9), key
