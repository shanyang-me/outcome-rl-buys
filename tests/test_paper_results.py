"""The paper's own analysis scripts, run on the shipped rows, rebuild the result files the paper was written from.

paper/materialize.py writes every model x budget of data/rows in the layout the scripts read; each script then runs
unchanged from paper/analysis and its output is compared with paper/results (written by the research runs, Oct 2026).
Points and the seeded bootstrap intervals must agree to 1e-9.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "paper" / "analysis"
RESULTS = ROOT / "paper" / "results"
# separate 4k/8k/16k generations first, then the long reads; the order the paper's cells were read in
RUNS = ["pair_decomp3_a", "pair_decomp3_b", "pair_decomp3_e", "pair_decomp3_f", "pair_decomp2", "pair_decomp4",
        "pair_decomp6_cap64", "pair_decomp10_native", "pair_decomp9_32b", "pair_decomp5c_cap48", "pair_decomp5_big"]
CAP_RUNS = ["pair_decomp6_cap64", "pair_decomp10_native", "pair_decomp9_32b", "pair_decomp5c_cap48"]

pytestmark = pytest.mark.slow


def run(script, *args, env=None):
    subprocess.run([sys.executable, str(ANALYSIS / script), *map(str, args)], cwd=ANALYSIS, check=True,
                   capture_output=True, env=env)


def diffs(got, want, path=""):
    if isinstance(want, dict):
        for k, v in want.items():
            yield from diffs(got.get(k) if isinstance(got, dict) else None, v, f"{path}/{k}")
    elif isinstance(want, list):
        for i, v in enumerate(want):
            yield from diffs(got[i] if isinstance(got, list) and i < len(got) else None, v, f"{path}[{i}]")
    elif isinstance(want, (int, float)) and not isinstance(want, bool):
        if got is None or abs(got - want) > 1e-9:
            yield path, got, want


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("rows")
    subprocess.run([sys.executable, str(ROOT / "paper" / "materialize.py"), str(out)], check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def cap_split(rows, tmp_path_factory):
    out = tmp_path_factory.mktemp("res") / "cap_split_phys.json"
    run("cap_split.py", out, *(rows / r for r in CAP_RUNS))
    return out


def test_cap_split(cap_split):
    got, want = json.loads(cap_split.read_text()), json.loads((RESULTS / "cap_split_phys.json").read_text())
    assert list(diffs(got["recipes"], want["recipes"])) == []


def test_cap_cells(rows, cap_split, tmp_path):
    run("cells_table.py", tmp_path / "capcells.tex", tmp_path / "cap_cells.json", cap_split, *(rows / r for r in RUNS))
    got, want = json.loads((tmp_path / "cap_cells.json").read_text()), json.loads((RESULTS / "cap_cells.json").read_text())
    assert list(diffs(got, want)) == []


def test_split_sensitivity(rows, cap_split, tmp_path):
    import os
    run("fig1_ci.py", tmp_path / "fig1_ci.json", cap_split, *(rows / r for r in RUNS), env={**os.environ, "KINDS": "phys,math"})
    got, want = json.loads((tmp_path / "fig1_ci.json").read_text()), json.loads((RESULTS / "fig1_ci.json").read_text())
    assert list(diffs(got, want)) == []


def test_pooled_shares():
    out = subprocess.run([sys.executable, "-I", str(ANALYSIS / "pooled_shares.py"), str(ROOT / "paper" / "tables" / "capcells.csv")],
                         check=True, capture_output=True, text=True).stdout
    assert out == (RESULTS / "pooled_shares_out.txt").read_text()
