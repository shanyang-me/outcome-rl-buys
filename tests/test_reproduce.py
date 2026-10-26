"""Reproduction tests: the refactored pipeline reproduces the numbers the paper's scripts produced from the same rows.

The golden files under ``tests/golden`` were written by the research scripts on 2026-10-19.
Point estimates must agree to floating-point precision; bootstrap intervals, which depend
on the random stream, must agree closely (the slow tests re-run them with the same seeds).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pairaudit.analysis.budget import cap_split
from pairaudit.analysis.ledger import budget_cells, cap_cells, crossed_bootstrap, point
from pairaudit.analysis.registry import Registry
from pairaudit.analysis.rows import RowStore

ROOT = Path(__file__).resolve().parents[1]
GOLD = Path(__file__).resolve().parent / "golden"


@pytest.fixture(scope="module")
def reg():
    return Registry.load()


@pytest.fixture(scope="module")
def stores(reg):
    return RowStore(ROOT / "data/rows", reg.views["fixed"]), RowStore(ROOT / "data/rows", reg.views["cap"])


def test_cap_split_matches_golden(reg, stores):
    gold = json.loads((GOLD / "cap_split_phys.json").read_text())
    out = cap_split(stores[1], reg)
    assert set(out["recipes"]) == set(gold["recipes"])
    for name, g in gold["recipes"].items():
        o = out["recipes"][name]
        for key in ("at_cap", "beyond_cap", "at_label", "beyond_label", "below_only", "train_cap", "base"):
            assert o[key] == g[key], (name, key)
        for which in ("at", "beyond"):
            if which in g:
                for k, v in g[which].items():
                    assert o[which][k] == pytest.approx(v, abs=1e-9), (name, which, k)
    for which, p in gold["pooled"].items():
        for k, v in p.items():
            assert out["pooled"][which][k] == pytest.approx(v, abs=1e-9), (which, k)


def test_ledger_points_match_golden(reg, stores):
    story = json.loads((GOLD / "fig1_story.json").read_text())
    ci = json.loads((GOLD / "fig1_ci.json").read_text())
    fixed = reg.select(["registered"])
    for cap, g in story["budget"].items():
        pt = point(budget_cells(stores[0], reg, fixed, int(cap)))
        for k in ("F", "P", "reach", "reach_robust", "n", "cells"):
            assert pt["pooled"][k] == pytest.approx(g["pooled"][k], abs=1e-9), (cap, k)
        for key, cell in g["recipes"].items():
            for k in ("d", "F", "mirror", "n", "n_never", "n_never_rl"):
                assert pt["recipes"][key][k if k not in ("n_never", "n_never_rl") else {"n_never": "U", "n_never_rl": "Up"}[k]] == pytest.approx(cell[k], abs=1e-9), (cap, key, k)
            assert pt["recipes"][key]["P"] == pytest.approx(cell["P"] + cell["mirror"], abs=1e-9)
            assert pt["recipes"][key]["reach"] == pytest.approx(cell["reach"] - cell["mirror"], abs=1e-9)
    for which in ("at", "beyond"):
        cells, _ = cap_cells(stores[1], reg, reg.select(), which)
        pt = point(cells)
        for k in ("F", "P", "reach", "reach_robust", "n", "cells"):
            assert pt["pooled"][k] == pytest.approx(story["cap"][which]["pooled"][k], abs=1e-9), (which, k)
        for key in ("size", "signed", "size_sym", "signed_sym", "size_m", "signed_m", "diff"):
            assert list(pt["shares"][key]) == pytest.approx(ci[which]["point"][key], abs=1e-9), (which, key)


@pytest.mark.slow
def test_ledger_intervals_match_golden(reg, stores):
    """Same seeds, same draw order (one stream across the at and beyond rows): the intervals come back to the digit."""
    import numpy as np

    ci = json.loads((GOLD / "fig1_ci.json").read_text())
    rng_recipes, rng_bases = np.random.default_rng(0), np.random.default_rng(1)
    for which in ("at", "beyond"):
        cells, _ = cap_cells(stores[1], reg, reg.select(), which)
        got = crossed_bootstrap(cells, n_boot=2000, rng=rng_recipes)
        for key in ("size", "signed", "size_sym", "signed_sym", "size_m", "signed_m", "diff"):
            assert [x for pair in got[key] for x in pair] == pytest.approx([x for pair in ci[which]["ci"][key] for x in pair], abs=1e-9), (which, key)
        base_of = {r: reg.recipes[r].base_label for r, _ in cells}
        got_b = crossed_bootstrap(cells, n_boot=2000, cluster=base_of, rng=rng_bases)
        for key in ("size", "signed", "size_m", "signed_m"):
            assert [x for pair in got_b[key] for x in pair] == pytest.approx([x for pair in ci[which]["ci_base"][key] for x in pair], abs=1e-9), (which, key)
