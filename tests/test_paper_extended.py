"""The paper's appendix reads, rebuilt from the shipped rows by the paper's own scripts.

Harder physics set, competition math, authors' temperatures, both judge regrades and reach at k=32. Each script runs unchanged
from paper/analysis on data/rows and data/extra; its output is compared with the result file the paper used (paper/results,
to 1e-9) or, where that file was not kept, with the paper's printed table (paper/tables, cell by cell).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "paper" / "analysis"
RES = ROOT / "paper" / "results"
TAB = ROOT / "paper" / "tables"

pytestmark = pytest.mark.slow

TEMP_PAIRS = {  # recipe: (RL, base, the 0.6 read it is compared with)
    "Skywork-32B": ("Skywork-OR1-32B", "DeepSeek-R1-Distill-Qwen-32B", "pair_decomp9_32b"),
    "JustRL-N": ("JustRL-Nemotron-1.5B", "OpenMath-Nemotron-1.5B", "pair_regen64"),
    "JustRL-D": ("JustRL-DeepSeek-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B", "pair_regen64"),
    "Polaris-1.7B": ("Polaris-1.7B-Preview", "Qwen3-1.7B", "pair_decomp10_native"),
    "Polaris-4B": ("Polaris-4B-Preview", "Qwen3-4B", "pair_decomp10_native"),
    "QuestA": ("QuestA-Nemotron-1.5B", "OpenMath-Nemotron-1.5B", "pair_regen64"),
    "Skywork-7B": ("Skywork-OR1-7B", "DeepSeek-R1-Distill-Qwen-7B", "pair_regen64"),
    "Skywork-Math": ("Skywork-OR1-Math-7B", "DeepSeek-R1-Distill-Qwen-7B", "pair_regen64"),
}


def py(script, *args, cwd=A, **env):
    subprocess.run([sys.executable, str(script), *map(str, args)], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, **{k: str(v) for k, v in env.items()}})


def same_json(got, want):
    py(ROOT / "paper" / "compare_json.py", got, want, cwd=ROOT)


def same_table(tex, csv):
    py(ROOT / "paper" / "compare_table.py", tex, csv, cwd=ROOT)


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    """build dir: rows/<run>/<model>_cap<C>.jsonl and extra/<run>/... unpacked."""
    out = tmp_path_factory.mktemp("build")
    py(ROOT / "paper" / "materialize.py", out / "rows", cwd=ROOT)
    for f in (ROOT / "data" / "extra").rglob("*"):
        if f.is_file():
            dst = out / "extra" / f.relative_to(ROOT / "data" / "extra")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if f.suffix == ".gz":
                with gzip.open(f, "rb") as a, open(dst.with_suffix(""), "wb") as z:
                    shutil.copyfileobj(a, z)
            else:
                shutil.copy(f, dst)
    for run, name in (("pair_hard", "hard671"), ("pair_math", "math276")):  # the scripts read ids, sources and golds, not texts
        with open(out / "rows" / run / "problems.jsonl", "w") as h:
            for line in open(ROOT / "sets" / f"{name}.jsonl"):
                r = json.loads(line)
                h.write(json.dumps({**{k: v for k, v in r.items() if k != "q_sha256"}, "q": "", "gt": json.dumps(r["gt"]),
                                    "scorable": r["scorable"] in (True, "True"), "diff": None}) + "\n")
    return out


def test_harder_set(b):
    j = b / "extra" / "pair_hard" / "judged"
    py(A / "hard_reach.py", b / "hj.json", b / "rows" / "pair_hard", JUDGED=j, NBOOT=2000)
    same_json(b / "hj.json", RES / "harder_set" / "hard_reach_judged.json")
    py(A / "hard_reach.py", b / "hm.json", b / "rows" / "pair_hard", NBOOT=2000)
    same_json(b / "hm.json", RES / "harder_set" / "hard_reach_matcher.json")
    py(A / "hard_by_source.py", b / "bs.json", b / "rows" / "pair_hard", JUDGED=j, NBOOT=2000)
    same_json(b / "bs.json", RES / "harder_set" / "by_source_judged.json")
    py(A / "hard_table.py", b / "hj.json", RES / "cap_cells.json", b / "hardcells.tex", b / "filter_j.tex")
    same_table(b / "hardcells.tex", TAB / "hardcells.csv")
    py(A / "hard_table.py", b / "hm.json", RES / "cap_cells.json", b / "cells_m.tex", b / "filter.tex")
    same_table(b / "filter.tex", TAB / "filter.csv")
    e = b / "extra" / "pair_hard32"
    py(A / "hard32_analysis.py", e / "targets.json", e, j, b / "rows" / "pair_hard", b / "h32.json", JUDGED32=e / "judged")
    py(A / "hard32_table.py", b / "h32.json", b / "hard32.tex")
    same_table(b / "hard32.tex", TAB / "hard32.csv")


def test_competition_math(b):
    m, j, e = b / "rows" / "pair_math", b / "extra" / "pair_math" / "judged", b / "extra" / "pair_math32"
    py(A / "hard_reach.py", b / "mr.json", m, JUDGED=j, KINDS="pre,post", NBOOT=5000)
    py(A / "math_p4.py", b / "p4.json", m, JUDGED=j, NBOOT=5000)
    py(ROOT / "paper" / "math" / "p5_harness.py", m, b / "p5.json", cwd=ROOT / "paper" / "math", JUDGED=j)
    py(A / "math_table.py", b / "mr.json", b / "p5.json", b / "mathcells.tex", b / "mathyears.tex", b / "mathharness.tex",
       P4=b / "p4.json", SCORER="judge")
    for t in ("mathcells", "mathyears", "mathharness"):
        same_table(b / f"{t}.tex", TAB / f"{t}.csv")
    py(A / "hard32_analysis.py", e / "targets.json", e, j, m, b / "m32.json", JUDGED32=e / "judged")
    py(A / "hard32_table.py", b / "m32.json", b / "math32.tex")
    same_table(b / "math32.tex", TAB / "math32.csv")


def test_authors_temperature(b):
    for name, (rl, base, run06) in TEMP_PAIRS.items():
        py(A / "temp_cells.py", rl, base, b / "rows" / run06, b / "rows" / "pair_temp", b / f"temp_{name}.json")
        same_json(b / f"temp_{name}.json", RES / "analysis_results" / f"temp_cells_{name}.json")


def test_judge_regrades(b):
    py(A / "regrade_cells.py", b / "extra" / "pair_regrade", RES / "matcher_audit", b / "rg.json")
    same_json(b / "rg.json", RES / "regrade_cells.json")
    py(A / "regrade2_table.py", b / "extra" / "pair_regrade2" / "pair_regen64", b / "rg2.json", b / "regrade2.tex")
    same_json(b / "rg2.json", RES / "analysis_results" / "regrade2_cells.json")
    same_table(b / "regrade2.tex", TAB / "regrade2.csv")


def test_reach_at_32(b):
    py(A / "reach32_table.py", RES / "resample" / "reach32_targets.json", b / "extra" / "pair_reach32", b / "r32.json", b / "r32.tex")
    same_json(b / "r32.json", RES / "resample" / "reach32_table.json")
    same_table(b / "r32.tex", TAB / "reach32.csv")
