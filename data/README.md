# Data

## Rows

`rows/<run>/<model>_cap<C>.jsonl.gz`, one JSON line per sample:

| field | meaning |
|---|---|
| `id` | problem id, shared across models |
| `kind` | `phys` (the 300 physics problems) or `math` (MATH-500) |
| `k` | sample index (0–7; reach reads continue at 8) |
| `finished` | stopped on its own within the generation cap |
| `score` | the frozen matcher's partial-credit score (0 when unfinished) |
| `len` | token length, clipped to the cap |
| `think_closed` | whether a `</think>` was emitted (informational) |

This row schema is the contract between scoring and analysis: `pairaudit score` writes it
from any model's outputs (`{id, k, text, finish_reason, len}` lines) with the frozen matcher
or a scorer of your own, and everything under `pairaudit/analysis` reads nothing else. `kind`
is any benchmark name; the first one in the config (or in the rows) is the primary one the cap
study selects budgets on.

`MANIFEST.json` lists every run, whether it is *nested* (one file per model at the generation
cap; tighter budgets are derived on load) or holds separate generations per budget, and the
budget ladder each model was materialised at. `configs/recipes.json` names which runs each
analysis view reads, in priority order.

| run | what | models |
|---|---|---|
| `pair_decomp3_a`, `3_b`, `3_e`, `3_f`, `pair_decomp2` | the first eight pairs at 4k/8k/16k, separate generations per budget | 12 |
| `pair_decomp4` | the ten pre-registered pairs and their bases, one 16k generation | 12 |
| `pair_decomp6_cap64` | the 1.5B/7B families and e3/Qwen3-1.7B at 64k (36k for the Qwen3-1.7B pair) | 20 |
| `pair_decomp10_native` | both Polaris pairs under native configs at 64k, physics only | 4 |
| `pair_decomp5_big`, `5c_cap48`, `9_32b` | the 14B and 32B pairs at 16k and 48k | 6 |
| `pair_temp` | the 32B pair at its authors' temperature (1.0, top-p 1.0) at 48k, physics only; more pairs land as the reads finish | 2 |
| `pair_decomp` | Open-Reasoner-Zero-7B vs Qwen2.5-7B at 2k/8k: the mirror regime (a non-thinking base), not part of the main study | 2 |

`meta/<run>/meta_<model>.json` records the Hugging Face snapshot, GPU, vLLM version, dtype,
overrides, settings and timings of each generation.

## Problems

The problem texts are not in this repository. They come from third-party benchmarks whose
licences allow research use but which this repository does not redistribute until each one
has been checked (TODO(shan): check and either add the texts or a builder script).

* `problems/phys300_ids.json`: the 300 text-only physics problems every pair is read on, drawn
  from the audit's corpus-v2 pools (closed-form golds under the frozen matcher). Id prefixes:
  `physcorp_` (150, UGPhysics-derived corpus), `tbr_` (99), `physicsb_` (32, the PHYSICS
  benchmark), `cal_problem_` (17, PhysReason), `comp_` (2).
* `problems/hard671_ids.json`: the harder unfiltered read (471 from OlympiadBench physics,
  PHYBench, the PHYSICS olympiad tier and FrontierScience, plus 200 refills from the audit's
  own pools with the base filter removed), with each problem's source and dataset tolerance.
* MATH-500: `tests.jsonl` from the MATH-500 release (`generate.items.load_math500`).

An items file for `pairaudit generate` is JSON lines `{id, q, gt, kind}` with `gt` a JSON
string `{"equiv_mode": "numeric" | "sympy", "golds": [...], "tol": optional}`.
