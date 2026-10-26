# What Outcome RL Buys: Finishing, Reliability, Rarely Reach

Code, metrics, test sets and row-level data for the paper (Shan Yang, 2026; arXiv: TODO).
Project page: https://shanyang.me/what-outcome-rl-buys/

We read 20 released outcome-RL models (1.5B to 32B, seven bases) against the thinking models they started from at fixed token
budgets, and split each change in pass@1 into three parts that add up exactly:

| part | what it measures |
|---|---|
| **finishing** | answering before the budget runs out |
| **reliability** | being right once finished |
| **reach** | solving problems the base answered wrongly in all eight tries, minus the problems the base solves that RL answered wrongly in all eight |

Where the base often runs out of budget, most of the change is finishing (82% at 4k and 8k). Where the base has enough tokens,
the gain is reliability. Reach adds at most 2.1 percentage points on any of the four physics and math sets.

## Reproduce

```bash
pip install -e ".[dev]"            # NumPy, SciPy, SymPy; no GPU
pytest -m "not slow"               # 85 tests: the split's identities, scorer cases, the first release's numbers (~20 s)
pytest -m slow                     # the paper's own scripts rebuild its numbers from the rows (~4 min): cap split,
                                   # per-cell intervals, pooled shares, harder set, competition math, authors' temperatures,
                                   # judge regrades, reach at k=32; each matches the paper's result file or printed table
python paper/materialize.py        # every model x budget as plain JSON lines in build/rows, for the scripts in paper/analysis
```

`docs/REPRODUCE.md` and `paper/TABLES.md` say which script writes each table and from which rows.

## Layout

```
pairaudit/        the metrics: rows and nested budgets, the exact three-part split with the mirror control, cap selection,
                  problem-clustered and crossed bootstraps, recipe statistics; the frozen answer matcher and judge protocol v2;
                  the sampler and GPU work queue used for the reads (generation needs `pip install -e ".[generate]"`)
paper/analysis/   the scripts that wrote the paper's numbers, unchanged
paper/results/    their outputs, as the paper used them
paper/tables/     every table in the paper as CSV
sets/             test sets: ids, gold answers and a hash of each problem text; phys300.jsonl also ships the texts;
                  build_sets.py rebuilds the other sets' texts from their sources at pinned revisions; SOURCES.md has licences
data/rows/        one line per sample {id, kind, k, finished, score, len} for every model and run of the main study
data/extra/       24-sample rounds, judged scores and judge verdicts, summaries of each read
data/meta/        model snapshot, GPU, vLLM version, sampling settings and timings of each read
configs/          the 20 pairs: Hugging Face ids, verified training caps, authors' sampling settings
docs/             METHOD.md (the split), PROTOCOL.md (sampling, scoring, judging)
```

## Test sets

| set | problems | texts |
|---|---|---|
| physics | 300 (Physics-R1 corpus from UGPhysics, TextbookReasoning, PHYSICS, PhysReason) | shipped in `sets/phys300.jsonl` (CC BY-NC-SA 4.0, see `sets/SOURCES.md`) |
| harder physics | 671 (OlympiadBench, PHYBench, PHYSICS, FrontierScience, UGPhysics) | `python sets/build_sets.py` |
| competition math | 276 (MathArena: 2025 contests, AIME and HMMT February 2026) | `python sets/build_sets.py` |
| MATH-500 | 500 | the public release |

For the harder and competition-math sets the repository ships hashes and golds, not text (UGPhysics and MathArena are
CC BY-NC-SA 4.0); the builder fails loudly if any rebuilt text differs from what the models saw.

## Licence and citation

Code and data are Apache-2.0, except `sets/phys300.jsonl`, which is CC BY-NC-SA 4.0 (`sets/SOURCES.md`). The rows are derived measurements and carry no third-party text. `CITATION.cff` has the
metadata.
