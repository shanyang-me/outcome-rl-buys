# Pre-registration: the competition-math read (drafted Oct 21, 2026, before any output exists)

## Why
Round 11 of the review loop judged the paper's reliability claim as established only on physics, which is out of domain for
19 of 20 recipes, because MATH-500 is at ceiling (the bases are right 89 to 92 percent of the time when they finish). This read
puts the same split on hard, in-domain math at each recipe's own cap, where the recipes report their gains.

## Problems
276 final-answer competition problems from MathArena (problems_math.jsonl, raw copies in raw/):
- after every model's release (kind "post", 63): AIME 2026 (30), HMMT February 2026 (33);
- 2025 contests (kind "pre", 213): AIME 2025 (30), HMMT February 2025 (30), HMMT November 2025 (30), BRUMO 2025 (30),
  CMIMC 2025 (40), SMT 2025 (53).
The two kinds are the contamination strata, as OlympiadBench against FrontierScience on the harder physics set. Proof-based
contests (USAMO, IMO, Putnam, IMC) are excluded.

## Protocol (identical to the harder physics set except scoring)
Every one of the 27 models: eight samples per problem, temperature 0.6, top-p 0.95, seed 0, native configs, one generation at
the larger of 16k tokens and the training cap; smaller budgets read from the same samples by truncation; a base read at the
largest cap among its recipes. Scoring: the frozen matcher with PAIR_EXACT=1 (a numeric gold must be matched exactly; the
matcher's 1 percent tolerance is for physics), then the judge protocol v2 on every finished sample the matcher rejects. Known leniency left in place: a symbolic gold (a radical or an expression) is still credited when the model boxes a decimal within 1 percent of its value, since the exact mode only covers golds that parse to a number; the judge pass does not see those accepts, so they are counted after the read from the stored rows. Then
the 24-sample round on every one-sided problem of every pair, both sides, as on the harder set. Reads then analyses with
hard_reach.py, hard_by_source.py, hard32_targets.py and hard32_analysis.py unchanged.

## Predictions (the paper's framework applied to this set; to be scored after the read)
P1. Finishing. At each recipe's cap, the base's finishing on these problems is below its value on the paper's physics set for
    every pair, since the problems are harder, and the four recipes whose base runs out of tokens at the cap on physics
    (Open-RS3, ProRL v2, JustRL-DeepSeek, e3) have the finishing term as the larger part here too.
P2. Reliability. For every recipe whose base finishes at least 0.90 of its derivations at the cap, pass@1 on finished
    derivations rises (P > 0 with the interval excluding zero in at least twelve of the sixteen), and reliability is the larger
    part on the point estimate in a majority of them.
P3. Reach. Net reach at the cap is within [-0.02, +0.03] for every pair; its interval is clear of zero for at most half of the
    pairs; robust reach (wins in at least four of eight, net of the mirror) is at most 0.02 everywhere; the mirror is of the
    same size as the RL side's count on every pair (ratio between 0.5 and 2 where both exceed ten problems).
P4. Contamination. Net reach on the 2026 contests is no larger than on the 2025 contests (difference within the paired interval).
P5. Harness. Our read of each RL model's AIME 2025 pass@1 at its cap is within 0.08 of the number its authors report, where
    they report one at a comparable budget and temperature.
Any prediction that fails is reported as failing, as H2 was on physics.

## Cost
About a quarter of the harder physics set's generation: roughly one day on four or five pods, or two to three days on two.
Run: bash math_chain.sh [reverse|forward] on each pod after copying problems_math.jsonl to /workspace/w3/pair_math/ and
deploying the patched pair_hard.py; judge with judge_hard.py on /workspace/w3/pair_math; the +24 round with hard32_targets.py
and the reach32 reader as for pair_hard32.
