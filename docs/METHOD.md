# The accounting

A *cell* is one released recipe against its base on one benchmark at one budget B, with n
problems and k = 8 samples each. A sample is **finished** when it stopped on its own within
B tokens; an unfinished sample scores 0. A sample is **solved** when it is finished and the
matcher gives full credit.

## Two parts: the capped score

Let `m` be the mean capped score (partial credit), `fin` the finished rate, and
`acc = m / fin` the mean score over finished samples. Then, exactly,

    m_RL − m_base = F + P
    F = (fin_RL − fin_base) · acc_RL            the finishing term
    P = fin_base · (acc_RL − acc_base)          the reliability term

F is the change that would be observed if the RL model's finished derivations were as
accurate as they are but it finished as often as the base; P is the change on the base's
finishing rate from the RL model's higher or lower accuracy per finished derivation. The cross
term `(fin_RL − fin_base)(acc_RL − acc_base)` sits inside F. `budget.cap_split` and
`cells.fixed_budget_cells` report this split; the *finishing share* of a cell is F over the
change.

## Three parts: pass@1 with reach

On full credit, define per problem the solved counts `s_base`, `s_RL` and the finished counts
`f_base`, `f_RL` (out of k). Let

    U  = problems with f_base = k and s_base = 0        (the base finished every sample and never solved)
    U' = problems with f_RL = k and s_RL = 0            (the mirror set)

Then, with the two-part split taken over the problems outside U (weighted by their share of
all problems),

    Δpass@1 = F + P + reach
    reach   = Σ_{U} s_RL / (k n)                         (the RL model's wins on nothing the base's samples contain)
    mirror  = Σ_{U'} s_base / (k n)

The mirror's loss sits inside P, so the reported parts are `P' = P + mirror` and
`reach' = reach − mirror`; `F + P' + reach' = Δpass@1` still holds exactly. **Robust reach**
counts only problems of U that the RL model solves in at least four of eight samples.

Two further exact splits bound where the cross term goes:

* symmetric: `F_s = (fin_RL − fin_base) · (acc_RL + acc_base)/2`, `P_s = (fin_RL + fin_base)/2 · (acc_RL − acc_base)`;
* matched: `P_m = fin_base · (acc_RL − acc_base)` with both accuracies taken only on the
  problems both models finish in at least seven of eight samples, `F_m` the remainder, so
  selection into the finished set cannot enter P.

`decomp.three_part` returns all of them from one pass over the per-problem arrays.

## Budgets

The fixed-budget study reads every registered recipe at 4k, 8k and 16k tokens. The cap study
reads each recipe at the smallest ladder budget at or above its final training response cap
(`at`) and at the next ladder budget (`beyond`); the ladder is 4k, 8k, 16k, 24k, 28k, 32k,
36k, 48k, 52k, 64k. Training caps are verified against each recipe's primary source and
recorded with a note in `configs/recipes.json`.

## Pooling and intervals

Pooled shares over cells: `size = (Σ|F|, Σ|P'|, max(0, Σ reach')) / total` and
`signed = (ΣF, ΣP', Σ reach') / Σ(F + P' + reach')`. Sizes are what the figures show, because
some recipes finish less than their base while gaining reliability; signed shares are reported
beside them.

* Per cell: paired bootstrap over problems, 20,000 resamples, percentile 95% intervals.
* Pooled: crossed bootstrap, 2,000 replicates, each resampling the recipes with replacement
  and, in the same replicate, the shared problems of each benchmark (one draw of problem
  counts weights every cell, since every model answers the same problems).
* Base-clustered: the same with the bases resampled, each with all its recipes.
* A change "counts" when its interval, rounded to the printed precision, excludes zero
  (the registered rule).

## Recipe-level statistics (`analysis/stats.py`)

Binding-cap dominance `D = Σ|F| / (Σ|F| + Σ|P|)` over a recipe's 4k and 8k cells; a one-sided
sign test over the changing recipes with D > 1/2; pooled dominance with the crossed bootstrap;
the matched change pooled as the mean of recipe means and by DerSimonian–Laird random effects
with a Hartung–Knapp–Sidik–Jonkman interval; Spearman's correlation between a recipe's median
log length ratio and its mean finishing change, with an exact permutation p up to eight
recipes and Monte Carlo beyond.

## Pre-registration

Six pairs on two new bases were registered on 2026-10-11 (hypotheses, statistic, matched
rule, predicted directions) before any output existed; four more on an already-read base
were added the same day as an amendment. H1 (every added recipe that changes its score where
the budget binds has D > 1/2) held 10 of 10, p = 0.001; H2 (matched changes stay small) held.
`stats.recipe_stats` scores the hypotheses from the cell table.
