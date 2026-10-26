"""Per-cell statistics with problem-clustered bootstrap intervals.

``fixed_budget_cells``: for every recipe and fixed budget, base -> RL in capped score,
finished rate, score given finished and any-solve, each with a paired-by-problem
percentile interval; the exact two-part split of the capped change; the matched change
(score given finished on problems both models finish >= 7 of 8); and the printed-interval
flags the registered analysis uses (a change counts when its interval, rounded to the
printed precision, excludes zero).

``cap_cell_table``: for every recipe at its cap and one beyond, the three-part split with
paired bootstrap intervals over problems.
"""

from __future__ import annotations

import numpy as np

from pairaudit.analysis.budget import cap_budgets
from pairaudit.analysis.decomp import Cell, arrays, matched_sgf_delta, rates, three_part
from pairaudit.analysis.registry import Recipe, Registry
from pairaudit.analysis.rows import RowStore, resolve_kinds

MATCHED_MIN_PROBLEMS = 15


def _ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _resample_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def fixed_budget_cell(c: Cell, n_boot: int = 20000, seed: int = 0) -> dict:
    """One cell: rates base -> RL with intervals, the two-part split, the matched change and the flags."""
    rng = np.random.default_rng(seed)
    pt = rates(c)
    deltas = {q: [] for q in ("mean", "finished", "sgf", "any")}
    shares = []
    for _ in range(n_boot):
        w = _resample_weights(rng, c.n)
        r = rates(c, w)
        for q in deltas:
            deltas[q].append(r["rl"][q] - r["base"][q])
        dm = r["rl"]["mean"] - r["base"]["mean"]
        if abs(dm) > 1e-9:
            shares.append((r["rl"]["finished"] - r["base"]["finished"]) * r["rl"]["sgf"] / dm)
    cell = {"n_problems": c.n, "len_ratio": pt["rl"]["len"] / pt["base"]["len"]}
    for q in deltas:
        cell[q] = {"base": pt["base"][q], "rl": pt["rl"][q], "delta": pt["rl"][q] - pt["base"][q], "ci": _ci(np.array(deltas[q]))}
    lo, hi = (lambda q: cell[q]["ci"][0]), (lambda q: cell[q]["ci"][1])
    cell["capped_gain"], cell["capped_loss"] = round(lo("mean"), 3) > 0, round(hi("mean"), 3) < 0
    cell["finishing_gain"], cell["finishing_loss"] = round(lo("finished"), 2) > 0, round(hi("finished"), 2) < 0
    cell["perfinished_gain"], cell["perfinished_loss"] = round(lo("sgf"), 3) > 0, round(hi("sgf"), 3) < 0
    m = matched_sgf_delta(c)
    if m is not None and m[1] >= MATCHED_MIN_PROBLEMS:
        idx = np.flatnonzero(c.matched)
        bs = []
        for _ in range(n_boot):
            w = np.zeros(c.n)
            np.add.at(w, idx[rng.integers(0, len(idx), len(idx))], 1.0)
            bs.append(matched_sgf_delta(c, w)[0])
        cell["matched"] = {"n": m[1], "delta": m[0], "ci": _ci(np.array(bs))}
        cell["matched"]["gain"], cell["matched"]["loss"] = round(cell["matched"]["ci"][0], 3) > 0, round(cell["matched"]["ci"][1], 3) < 0
    else:
        cell["matched"] = {"n": m[1] if m else 0}
    cell["finishing_term"] = cell["finished"]["delta"] * cell["sgf"]["rl"]
    cell["perfinished_term"] = cell["finished"]["base"] * cell["sgf"]["delta"]
    dm = cell["mean"]["delta"]
    cell["finishing_share"] = cell["finishing_term"] / dm if abs(dm) > 1e-9 else None
    cell["finishing_share_ci"] = _ci(np.array(shares)) if len(shares) > 100 else None
    cell["few_finished"] = min(int(c.bf.astype(bool).sum()), int(c.rf.astype(bool).sum())) < MATCHED_MIN_PROBLEMS
    return cell


def fixed_budget_cells(store: RowStore, reg: Registry, recipes: list[Recipe] | None = None, caps: list[int] | None = None,
                       kinds=None, n_boot: int = 20000, seed: int = 0) -> dict[str, dict]:
    """Every (recipe, benchmark, fixed budget) cell that was read, keyed ``"<model> cap<cap> <kind>"``."""
    recipes = recipes or reg.registered
    kinds = resolve_kinds(store, reg, recipes, kinds)
    caps = caps or reg.fixed_budgets or store.common_caps({m for r in recipes for m in (r.base, r.model)})
    out = {}
    for base, recs in reg.families(recipes).items():
        for cap in caps:
            if not store.has(base, cap):
                continue
            B = store.load(base, cap)
            for rec in recs:
                if not store.has(rec.model, cap):
                    continue
                Rl = store.load(rec.model, cap)
                for kind in kinds:
                    c = arrays(B, Rl, kind)
                    if c is None or c.n < 20:
                        continue
                    out[f"{rec.model} cap{cap} {kind}"] = {"base": base, "rl": rec.model, "recipe": rec.name, "cap": cap, "set": kind,
                                                            **fixed_budget_cell(c, n_boot, seed)}
    return out


def cap_cell_stats(c: Cell, n_boot: int = 20000, seed: int = 0) -> dict:
    """The three-part split with paired bootstrap intervals over problems."""
    rng = np.random.default_rng(seed)
    pt = three_part(c)
    boots = {k: [] for k in ("F", "P", "reach", "d")}
    for _ in range(n_boot):
        b = three_part(c, _resample_weights(rng, c.n))
        for k in boots:
            boots[k].append(b[k])
    out = {k: pt[k] for k in ("n", "fin_b", "fin_r", "acc_b", "acc_r", "U", "Up")}
    for k in boots:
        out[k] = [pt[k], *_ci(np.array(boots[k]))]
    return out


def cap_cell_table(store: RowStore, reg: Registry, recipes: list[Recipe] | None = None, kinds=None,
                   n_boot: int = 20000, seed: int = 0) -> dict[str, dict]:
    """Every cell behind the cap rows of the ledger, ordered as the figure: by the base's finished rate at the cap, lowest first."""
    rng = np.random.default_rng(seed)
    recs = recipes or reg.select()
    kinds = resolve_kinds(store, reg, recs, kinds)
    rows = {}
    for rec in recs:
        at, beyond, below_only = cap_budgets(store, rec)
        if at is None:
            continue
        rows[rec.name] = {"base": rec.base_label, "cells": [], "below_only": below_only}
        for which, cap in (("at", at), ("beyond", beyond)):
            if cap is None:
                continue
            B, Rl = store.load(rec.base, cap), store.load(rec.model, cap)
            for kind in kinds:
                c = arrays(B, Rl, kind)
                if c is None:
                    continue
                s = cap_cell_stats(c, n_boot, seed=int(rng.integers(0, 2**31)))
                s.update(which=which, kind=kind, cap=cap, label=reg.label(cap) + (" (below cap)" if below_only and which == "at" else ""))
                rows[rec.name]["cells"].append(s)
    order = sorted(rows, key=lambda n: next(c["fin_b"] for c in rows[n]["cells"] if c["which"] == "at" and c["kind"] == kinds[0]))
    return {n: rows[n] for n in order}
