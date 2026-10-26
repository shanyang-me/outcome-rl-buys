# Reproducing the paper's numbers

Everything below runs on CPU from `data/rows`.

| what | command | checked by |
|---|---|---|
| at-cap / beyond-cap cells per recipe (two-part split) | `pairaudit budget --out out/cap_split.json` | `tests/test_reproduce.py::test_cap_split_matches_golden` (1e-9) |
| the ledger: pooled shares at 4k/8k/16k, at the cap, beyond, with crossed-bootstrap intervals | `pairaudit ledger --out out/ledger.json --tex out/ledger.tex` | `test_ledger_points_match_golden` (1e-9), `test_ledger_intervals_match_golden` (1e-9, slow) |
| every fixed-budget cell with problem-clustered intervals (the paper's main table) | `pairaudit cells --out out/cells.json` | `tests/test_cells_golden.py` (points 1e-9, intervals 0.02) |
| every cap cell with intervals (appendix table) | `pairaudit captable --out out/cap_cells.json --tex out/cap_cells.tex` | run it; compare with `tests/golden/cap_cells.json` |
| recipe-level statistics and the pre-registered hypotheses | `pairaudit stats --flags out/cells.json --out out/stats.json` | — |
| a pair at T=0.6 and at its authors' temperature | `pairaudit temperature Skywork-32B` | — |
| one model's cell summary | `pairaudit summary Qwen3-4B 16384 --view fixed` | — |

`pairaudit ledger --kinds phys` pools physics only (the paper's main text pools physics and
MATH-500; the physics-only shares are in its appendix).

## Golden files

`tests/golden/*.json` are the files the research scripts wrote on 2026-10-19 from the same
rows. The reproduction tests rebuild them. Point estimates agree to floating point. The
ledger's crossed-bootstrap intervals agree to floating point too (same seeds, same draw
order). The per-cell intervals agree within 0.02: the original per-cell script drew from
Python's `random` and the package draws from NumPy's generator.

## Regenerating rows (GPU)

```bash
pip install -e ".[generate]"
# every model at max(16k, its cap) on a problem file, two pods sharing one queue
bash scripts/pod/chain.sh generate forward          # pod A
bash scripts/pod/chain.sh generate reverse no32b    # pod B (H100)
# 24 more samples on each model's never-solved problems
bash scripts/pod/chain.sh reach
# judge regrade of the stored texts
PAIRAUDIT_SRC=/workspace/audit/pair_hard bash scripts/pod/chain.sh regrade
```

Add a finished run to the data by copying its `<model>_cap<generation cap>.jsonl` files
(gzipped) into `data/rows/<run>/` and listing the run in `data/rows/MANIFEST.json` with
`nested: true` and the budget ladder it materialised; then name it in the view it belongs to
in `configs/recipes.json`.
