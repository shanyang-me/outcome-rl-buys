"""Recipe-level inference for the fixed-budget study: the recipe is the unit of replication.

* Binding-cap dominance of a recipe: D = sum|F| / (sum|F| + sum|P|) over its cells at the
  budgets where the cap binds (4k and 8k by default). A recipe *changes where the budget
  binds* when one of those cells' capped change has a printed interval excluding zero.
* Sign test: one-sided binomial over the changing recipes with D > 1/2.
* Pooled dominance with a crossed bootstrap (recipes and problems resampled together).
* Matched change (score given finished on problems both models finish >= 7 of 8), pooled
  as the mean of recipe means and by DerSimonian-Laird random effects with an HKSJ interval.
* Length vs finishing: Spearman of median log length ratio against mean dfinished at the
  binding budgets, with an exact permutation p (Monte Carlo beyond eight recipes).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as sps

from pairaudit.analysis.decomp import Cell, arrays, matched_sgf_delta, rates
from pairaudit.analysis.registry import Recipe, Registry
from pairaudit.analysis.rows import RowStore, resolve_kinds

MATCHED_MIN_PROBLEMS = 15


def _cell_stats(c: Cell, w: np.ndarray | None) -> tuple[float, float, float, float, float | None]:
    r = rates(c, w)
    fb, fr, ab, ar = r["base"]["finished"], r["rl"]["finished"], r["base"]["sgf"], r["rl"]["sgf"]
    m = matched_sgf_delta(c, w)
    return r["rl"]["mean"] - r["base"]["mean"], (fr - fb) * ar, fb * (ar - ab), fr - fb, (m[0] if m else None)


def collect_cells(store: RowStore, reg: Registry, recipes: list[Recipe], caps=None, kinds=None):
    kinds = resolve_kinds(store, reg, recipes, kinds)
    caps = caps or reg.fixed_budgets or store.common_caps({m for r in recipes for m in (r.base, r.model)})
    cells = []
    ids: dict[str, list[str]] = {}
    for base, recs in reg.families(recipes).items():
        for cap in caps:
            if not store.has(base, cap):
                continue
            B = store.load(base, cap)
            for rec in recs:
                if not store.has(rec.model, cap):
                    continue
                for kind in kinds:
                    c = arrays(B, store.load(rec.model, cap), kind)
                    if c is None:
                        continue
                    ids.setdefault(kind, c.ids)
                    if c.ids != ids[kind]:
                        raise ValueError(f"{rec.name} {cap} {kind}: problem set differs from the shared one")
                    cells.append({"rec": rec.name, "kind": kind, "cap": cap, "cell": c})
    return cells, ids


def perm_p(x, y, rho, n_mc: int = 100000, seed: int = 1) -> tuple[float, str]:
    """Two-sided permutation p for Spearman's rho: exact up to 8 points, Monte Carlo beyond."""
    n = len(x)
    if n <= 8:
        vals = [sps.spearmanr(x, [y[i] for i in p]).statistic for p in itertools.permutations(range(n))]
        how = "exact"
    else:
        g = np.random.default_rng(seed)
        vals = [sps.spearmanr(x, [y[i] for i in g.permutation(n)]).statistic for _ in range(n_mc)]
        how = f"monte carlo {n_mc}"
    return float(np.mean([abs(v) >= abs(rho) - 1e-12 for v in vals])), how


def recipe_stats(store: RowStore, reg: Registry, recipes: list[Recipe], flags: dict | None = None, binding=None,
                 caps=None, kinds=None, n_boot: int = 4000, seed: int = 0) -> dict:
    """Dominance, sign test, pooled matched change and the length association for a set of recipes.

    ``flags`` is the fixed-budget cell table (``cells.fixed_budget_cells``) and supplies the
    'changes where the budget binds' rule and the length ratios; without it every recipe counts as changing.
    """
    cells, ids = collect_cells(store, reg, recipes, caps, kinds)
    caps = sorted({c["cap"] for c in cells})
    binding = list(binding or reg.binding or caps[:-1])
    names = sorted({c["rec"] for c in cells})
    R = len(names)
    ones = {k: np.ones(len(v)) for k, v in ids.items()}

    def summarise(st, mult=None, capset=None):
        mult = mult or dict.fromkeys(names, 1)
        aF = dict.fromkeys(names, 0.0)
        aP = dict.fromkeys(names, 0.0)
        mm: dict[str, list[float]] = {n: [] for n in names}
        for c, (_d, F, P, _df, m) in zip(cells, st):
            if capset is None or c["cap"] in capset:
                aF[c["rec"]] += abs(F)
                aP[c["rec"]] += abs(P)
            if capset is None and m is not None and int(c["cell"].matched.sum()) >= MATCHED_MIN_PROBLEMS:
                mm[c["rec"]].append(m)
        dom = {n: aF[n] / (aF[n] + aP[n]) for n in names if aF[n] + aP[n] > 0}
        tf = sum(mult[n] * aF[n] for n in dom)
        tp = sum(mult[n] * aP[n] for n in dom)
        mmr = {n: (sum(v) / len(v) if v else None) for n, v in mm.items()}
        have = [n for n in names if mmr[n] is not None]
        pooled_mm = (sum(mult[n] * mmr[n] for n in have) / sum(mult[n] for n in have)) if have and capset is None else None
        return dom, (tf / (tf + tp) if tf + tp > 0 else float("nan")), mmr, pooled_mm

    point = [_cell_stats(c["cell"], ones[c["kind"]]) for c in cells]
    for _c, (d, F, P, _, _) in zip(cells, point):
        assert abs(d - F - P) < 1e-9
    dom_b, pd_b, mmr, pmm = summarise(point, capset=set(binding))
    _, _, mmr, pmm = summarise(point)
    rng = np.random.default_rng(seed)
    boot_pd, boot_mm = [], []
    boot_dom = {n: [] for n in names}
    boot_mmr = {n: [] for n in names}
    for _ in range(n_boot):
        w = {k: np.bincount(rng.integers(0, len(v), len(v)), minlength=len(v)).astype(float) for k, v in ids.items()}
        mult = dict.fromkeys(names, 0)
        for i in rng.integers(0, R, R):
            mult[names[i]] += 1
        st = [_cell_stats(c["cell"], w[c["kind"]]) for c in cells]
        _, pdb, _, _ = summarise(st, mult, set(binding))
        _, _, _, pm = summarise(st, mult)
        boot_pd.append(pdb)
        boot_mm.append(pm)
        d_p, _, _, _ = summarise(st, capset=set(binding))
        _, _, mmr_p, _ = summarise(st)
        for n in names:
            if n in d_p:
                boot_dom[n].append(d_p[n])
            if mmr_p[n] is not None:
                boot_mmr[n].append(mmr_p[n])
    def ci(x):
        x = [v for v in x if v is not None and not np.isnan(v)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x else None

    def changing(n):
        if flags is None:
            return True
        return any(v["recipe"] == n and v["cap"] in binding and (v["capped_gain"] or v["capped_loss"]) for v in flags.values())

    ch = [n for n in dom_b if changing(n)]
    k = sum(dom_b[n] > 0.5 for n in ch)
    res = {
        "n_cells": len(cells), "n_recipes": R, "binding_budgets": list(binding),
        "dominance": {n: {"point": dom_b[n], "ci_problems": ci(boot_dom[n]), "changing": changing(n)} for n in dom_b},
        "pooled_dominance": {"point": pd_b, "ci_crossed": ci(boot_pd), "p_le_half": float(np.mean(np.array(boot_pd) <= 0.5))},
        "sign_test": {"k_above_half": k, "n_changing": len(ch), "p_one_sided": float(sps.binomtest(k, len(ch), 0.5, alternative="greater").pvalue) if ch else None},
        "matched": {n: {"mean": mmr[n], "ci_problems": ci(boot_mmr[n])} for n in names},
        "pooled_matched": {"mean_of_recipe_means": pmm, "ci_crossed": ci(boot_mm)},
    }
    y = np.array([mmr[n] for n in names if mmr[n] is not None and boot_mmr[n]])
    se = np.array([np.std(boot_mmr[n]) for n in names if mmr[n] is not None and boot_mmr[n]])
    if len(y) >= 2:
        wv = 1 / se**2
        ybar = (wv * y).sum() / wv.sum()
        Q = float((wv * (y - ybar) ** 2).sum())
        kk = len(y)
        tau2 = max(0.0, (Q - (kk - 1)) / (wv.sum() - (wv**2).sum() / wv.sum()))
        ws = 1 / (se**2 + tau2)
        mu = float((ws * y).sum() / ws.sum())
        q_hk = float((ws * (y - mu) ** 2).sum() / (kk - 1))
        se_hk = math.sqrt(q_hk / ws.sum())
        tq = sps.t.ppf(0.975, kk - 1)
        res["random_effects_matched"] = {"mu": mu, "tau2": tau2, "Q": Q, "k": kk, "I2": max(0.0, (Q - (kk - 1)) / Q) if Q > 0 else 0.0,
                                         "hksj_ci": [mu - tq * se_hk, mu + tq * se_hk], "hksj_p_two_sided": float(2 * sps.t.sf(abs(mu) / se_hk, kk - 1))}
    if flags is not None:
        x, yy, used = [], [], []
        for n in names:
            lr = [math.log(v["len_ratio"]) for v in flags.values() if v["recipe"] == n]
            df = [v["finished"]["delta"] for v in flags.values() if v["recipe"] == n and v["cap"] in binding]
            if lr and df:
                x.append(float(np.median(lr)))
                yy.append(float(np.mean(df)))
                used.append(n)
        if len(used) >= 4:
            rho = float(sps.spearmanr(x, yy).statistic)
            pv, how = perm_p(x, yy, rho)
            res["length_vs_finishing"] = {"recipes": used, "median_log_len_ratio": x, "mean_dfinished_binding": yy, "spearman": rho, "p_two_sided": pv, "p_method": how}
    return res
