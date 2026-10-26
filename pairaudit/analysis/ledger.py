"""The ledger: pooled shares of finishing, reliability and reach over recipe x benchmark cells, with intervals.

Rows of the ledger are budgets: the fixed budgets (every registered recipe) and, from the
cap study, each recipe at its training cap and one budget beyond. A row pools the
three-part split (``decomp.three_part``) over its cells and reports the parts' sizes as
shares of their total (``size``) and the signed parts as shares of the net change
(``signed``), under the paper's split and the symmetric and matched variants.

Intervals come from a crossed bootstrap: each replicate resamples the recipes with
replacement and, in the same replicate, the shared problems of each benchmark (every
model answers the same problems, so one draw of problem counts weights every cell). A
second bootstrap resamples the bases, each with all its recipes, instead of the recipes.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from pairaudit.analysis.budget import cap_budgets
from pairaudit.analysis.decomp import Cell, arrays, pool_shares, three_part
from pairaudit.analysis.registry import Recipe, Registry
from pairaudit.analysis.rows import RowStore, resolve_kinds

SPLITS = ("", "sym", "matched")
KEYS = ("size", "signed", "size_sym", "signed_sym", "size_m", "signed_m", "diff")


def _pool(parts: list[dict]) -> dict[str, tuple]:
    out = {}
    for split, suffix in zip(SPLITS, ("", "_sym", "_m")):
        p = pool_shares(parts, split)
        out["size" + suffix] = p["size"]
        out["signed" + suffix] = p["signed"]
        if not split:
            out["reach_robust"] = (p["reach_robust"],)
    # reliability minus finishing, as a share, so the ordering of the two terms gets its own interval
    out["diff"] = (out["size"][1] - out["size"][0], out["signed"][1] - out["signed"][0], out["size_sym"][1] - out["size_sym"][0])
    return out


def budget_cells(store: RowStore, reg: Registry, recipes: list[Recipe], cap: int, kinds=None) -> dict[tuple[str, str], Cell]:
    """Cells for every recipe at one fixed budget, keyed by (recipe, kind), in family order."""
    kinds = resolve_kinds(store, reg, recipes, kinds)
    cells = {}
    for base, recs in reg.families(recipes).items():
        B = store.load(base, cap)
        for rec in recs:
            Rl = store.load(rec.model, cap)
            for kind in kinds:
                c = arrays(B, Rl, kind)
                if c is not None:
                    cells[(rec.name, kind)] = c
    return cells


def cap_cells(store: RowStore, reg: Registry, recipes: list[Recipe], which: str, kinds=None) -> tuple[dict[tuple[str, str], Cell], dict[str, int]]:
    """Cells for every recipe at its cap ('at') or one beyond ('beyond'); recipes read below their cap are skipped."""
    kinds = resolve_kinds(store, reg, recipes, kinds)
    cells, caps = {}, {}
    for rec in recipes:
        at, beyond, below_only = cap_budgets(store, rec)
        cap = {"at": at, "beyond": beyond}[which]
        if below_only or cap is None:
            continue
        B, Rl = store.load(rec.base, cap), store.load(rec.model, cap)
        for kind in kinds:
            c = arrays(B, Rl, kind)
            if c is not None:
                cells[(rec.name, kind)] = c
        caps[rec.name] = cap
    return cells, caps


def point(cells: dict[tuple[str, str], Cell]) -> dict:
    """Per-cell parts and the pooled shares, unweighted."""
    per = {f"{r}|{k}": three_part(c) for (r, k), c in cells.items()}
    pooled = _pool(list(per.values()))
    return {"recipes": per, "pooled": {"F": pooled["size"][0], "P": pooled["size"][1], "reach": pooled["size"][2],
                                       "reach_robust": pooled["reach_robust"][0], "signed": pooled["signed"],
                                       "n": len({r for r, _ in cells}), "cells": len(cells)}, "shares": pooled}


def crossed_bootstrap(cells: dict[tuple[str, str], Cell], kinds=None, n_boot: int = 2000, seed: int = 0,
                      cluster: dict[str, str] | None = None, rng: np.random.Generator | None = None) -> dict[str, list[list[float]]]:
    """95% intervals of the pooled shares; ``cluster`` maps recipe -> base to resample bases instead of recipes.

    Pass one ``rng`` to several calls to draw them from one stream (the ledger does, so its rows share a stream).
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    kinds = tuple(kinds) if kinds else tuple(dict.fromkeys(k for _, k in cells))
    recs = sorted({r for r, _ in cells})
    ids = {kind: sorted({i for (r, k), c in cells.items() if k == kind for i in c.ids}) for kind in kinds}
    units = sorted(set(cluster.values())) if cluster else recs
    boot: list[dict] = []
    for _ in range(n_boot):
        drawn = list(rng.choice(units, len(units)))
        pick = [r for b in drawn for r in recs if cluster[r] == b] if cluster else drawn
        cnt = {kind: Counter(rng.choice(ids[kind], len(ids[kind]))) for kind in kinds}
        vals = []
        for r in pick:
            for kind in kinds:
                c = cells.get((r, kind))
                if c is None:
                    continue
                w = np.array([cnt[kind][i] for i in c.ids], float)
                if w.sum() and (w * ~c.never).sum():
                    vals.append(three_part(c, w))
        boot.append(_pool(vals))
    return {k: [[float(np.percentile([b[k][j] for b in boot], q)) for q in (2.5, 97.5)] for j in range(len(boot[0][k]))] for k in KEYS}


def ledger(store_fixed: RowStore, store_cap: RowStore, reg: Registry, kinds=None, n_boot: int = 2000,
           fixed_groups=None, cap_groups=None, with_ci: bool = True, fixed_budgets=None) -> dict:
    """The whole ledger: fixed-budget rows for the registered recipes, cap rows for all, with intervals.

    Defaults: ``kinds`` from the config or the rows; ``fixed_groups`` the registered pairs;
    ``fixed_budgets`` the config's, else every budget all fixed pairs share.
    """
    fixed = reg.select(list(fixed_groups)) if fixed_groups else reg.registered
    kinds = resolve_kinds(store_fixed, reg, fixed, kinds)
    budgets = list(fixed_budgets or reg.fixed_budgets or store_fixed.common_caps({m for r in fixed for m in (r.base, r.model)}))
    out = {"kinds": list(kinds), "fixed_budgets": budgets, "budget": {}, "cap": {}}
    rng_recipes, rng_bases = np.random.default_rng(0), np.random.default_rng(1)
    for cap in budgets:
        cells = budget_cells(store_fixed, reg, fixed, cap, kinds)
        out["budget"][str(cap)] = point(cells)
    for which in ("at", "beyond"):
        cells, caps = cap_cells(store_cap, reg, reg.select(cap_groups), which, kinds)
        entry = point(cells)
        for name, cap in caps.items():
            for kind in kinds:
                if f"{name}|{kind}" in entry["recipes"]:
                    entry["recipes"][f"{name}|{kind}"]["cap"] = cap
        if with_ci:
            entry["ci"] = crossed_bootstrap(cells, kinds, n_boot, rng=rng_recipes)
            base_of = {r: reg.recipes[r].base_label for r, _ in cells}
            entry["ci_base"] = crossed_bootstrap(cells, kinds, n_boot, cluster=base_of, rng=rng_bases)
            entry["bases"] = len(set(base_of.values()))
        out["cap"][which] = entry
    return out
