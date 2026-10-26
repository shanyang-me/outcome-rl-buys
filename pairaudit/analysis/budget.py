"""Which budget is 'at the cap' and 'beyond' for each recipe, and the two-part split there (the cap study's cell table)."""

from __future__ import annotations

import statistics as st

from pairaudit.analysis.decomp import arrays, two_part
from pairaudit.analysis.registry import Recipe, Registry
from pairaudit.analysis.rows import RowStore, resolve_kinds


def cap_budgets(store: RowStore, recipe: Recipe) -> tuple[int | None, int | None, bool]:
    """(at, beyond, below_only): the smallest common budget >= train cap, the next one up; or the highest read if none reaches the cap."""
    common = sorted(set(store.caps_for(recipe.model)) & set(store.caps_for(recipe.base)))
    if not common:
        return None, None, False
    at = min((c for c in common if c >= recipe.train_cap), default=None)
    if at is None:
        return max(common), None, True
    beyond = min((c for c in common if c > at), default=None)
    return at, beyond, False


def cap_split(store: RowStore, reg: Registry, recipes: list[Recipe] | None = None, kind: str | None = None) -> dict:
    """Per recipe: the at/beyond budgets and the two-part split at each; pooled means over recipes read at their cap.

    ``kind`` is the benchmark the budgets are selected on (default: the primary one).
    """
    recipes = recipes or reg.select()
    kind = kind or resolve_kinds(store, reg, recipes)[0]
    out = {"recipes": {}, "pooled": {}, "kind": kind}
    for rec in recipes:
        at, beyond, below_only = cap_budgets(store, rec)
        if at is None:
            continue
        entry = {
            "base": rec.base_label, "cap_label": rec.cap_label, "train_cap": rec.train_cap,
            "at_cap": at, "beyond_cap": beyond, "at_label": reg.label(at) + (" (below cap)" if below_only else ""),
            "beyond_label": reg.label(beyond) if beyond else None, "below_only": below_only,
        }
        entry["at"] = two_part(arrays(store.load(rec.base, at), store.load(rec.model, at), kind))
        if beyond:
            entry["beyond"] = two_part(arrays(store.load(rec.base, beyond), store.load(rec.model, beyond), kind))
        out["recipes"][rec.name] = entry
    for which in ("at", "beyond"):
        rs = [r[which] for r in out["recipes"].values() if which in r and not r.get("below_only")]
        if rs:
            out["pooled"][which] = {
                "n": len(rs), "F_mean": st.mean(r["F"] for r in rs), "P_mean": st.mean(r["P"] for r in rs),
                "d_mean": st.mean(r["d"] for r in rs),
                "dominance": sum(abs(r["F"]) for r in rs) / max(1e-9, sum(abs(r["F"]) + abs(r["P"]) for r in rs)),
                "finish_more": sum(1 for r in rs if r["F"] > 0), "finishing_larger": sum(1 for r in rs if abs(r["F"]) > abs(r["P"])),
                "rl_only_nb": sum(r["rl_only_base_all_finished"] for r in rs), "base_only_nb": sum(r["base_only_rl_all_finished"] for r in rs),
            }
    return out
