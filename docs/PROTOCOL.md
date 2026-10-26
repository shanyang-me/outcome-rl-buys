# Sampling, scoring and judging

## Sampling

* Prompt: the problem text plus `\n\nPlease reason step by step, and put your final answer within \boxed{}.`,
  rendered through the model's own chat template with the generation prompt added.
* 8 samples per problem, temperature 0.6, top-p 0.95, seed 0, bfloat16 (models that ship
  float16 weights are read in bfloat16 like every other).
* Native configs: a model that ships its own `rope_scaling` (Polaris: YaRN 1.5 over 65,536)
  is loaded unchanged; a model without rope scaling and too few positions for the generation
  (the Qwen3 bases, 40,960) gets only `max_position_embeddings` raised to the cap plus a margin.
  Applying static YaRN to both models of a pair was tried and rejected: it changed both
  models (Qwen3-4B finished 0.33 of samples at 16k against 0.88 natively).
* One generation per model at its largest budget. Finished at budget C iff
  `finish_reason == "stop"` and length ≤ C. The nested read was checked against separate
  generations at 4k/8k/16k with the same seed: 99% of samples identical in the
  R1-Distill-1.5B family, cell means within 0.01 elsewhere.
* Authors' temperature: for recipes whose authors evaluate at a different temperature
  (Skywork 1.0, Polaris 1.4, the OpenMath-Nemotron family 0.7), the pair is re-read at that
  temperature with top-p 1.0 (`configs/specs/authors_T.json`), and the split is reported
  side by side (`pairaudit temperature <recipe>`).

## Scoring (the frozen matcher)

`pairaudit.scoring.matcher.score(text, gt)` is the fraction of a problem's gold answers
matched by any balanced `\boxed{}` value in the text. `gt` is a JSON string
`{"equiv_mode": "numeric" | "sympy", "golds": [...]}`.

* Numeric golds: parsed with units; 1% relative tolerance after the unit is stripped; a
  different unit of the same dimension fails closed (values are compared, never converted).
* Symbolic golds: LaTeX to SymPy, equivalence by simplification with a sampled check as
  fallback; equation sides may be retained, reversed or dropped.
* Multi-component golds credit the fraction of components matched; prose golds are excluded
  from the denominator; unparseable golds stay in it and score 0.
* Malformed output never raises; it fails closed. A box cut off by the budget is skipped.

The module is frozen (SHA-256 `200d5aa0141268dce366cda8c38c5ec2e4640738d7bafbd2fb50a07a04830ae5`).
Every number in the audit was produced with it. Its 29 behavioural tests are in
`tests/test_answer_equiv.py`.

## Judging (protocol v2)

Matcher misses are measured, not patched. For every finished physics sample the matcher
scored below full credit, each gold part is graded:

1. `answer_norm.compare_any` decides the mechanical cases: a bare number matches a gold with
   a unit when magnitudes agree; a different unit matches only after conversion; sign is
   significant; a missing direction word is tolerated, a contradictory one is not; dataset
   tolerances (`tol` in the gold record) are honoured.
2. Symbolic, multi-valued and otherwise undecidable parts are deferred to a pinned open judge
   (Qwen3-32B, greedy) with the same rules written out and the parsed forms attached.
3. A sample's judged score is the share of its gold parts judged right, never below the
   matcher's score.

Why: the same 526 problems scored 40.1 under one judge and 24.0 under another, and the
disagreement concentrated on bare numbers against golds with units; left to a model's
discretion that call moved scores by more than the effects being measured, and unevenly
across systems. The cases the two graders disagreed on are pinned in `tests/test_answer_norm.py`.
`pairaudit regrade` runs the protocol over stored texts; `analysis/regrade.py` recomputes the
split under the judge beside the matcher's.
