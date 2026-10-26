"""Figure 1 ledger data. For each row (budget rows over the registered recipes at 4k/8k/16k; cap rows at each recipe's training cap and
one budget beyond, from the 64k read), a recipe's change in physics score vs its base is split exactly into
  reach  = the RL model's score on problems the base finished all eight times and solved none (within budget, wins alone),
  F, P   = finishing and reliability terms (aggregate rates, as in the paper) on the remaining problems,
with the mirror of reach (the base's score on problems the RL model finished all eight times and solved none) as the control.
Pooled shares use sum|F|, sum|P + mirror| and max(0, sum(reach - mirror)) over recipe x benchmark cells.
usage: KINDS=phys,math python fig1_story.py out.json cap_split.json rows_root [rows_root ...]"""
import json, glob, sys, collections, os
OUT, CAPJ, ROOTS = sys.argv[1], sys.argv[2], sys.argv[3:]
KINDS = tuple(os.environ.get("KINDS", "phys,math").split(","))  # benchmarks pooled; the paper's headline shares pool physics and MATH-500
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "overlap_per_recipe.py")).read(); exec(src[src.index("BASES = "):src.index("def load")])
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
LATE = ("pair_decomp9_32b", "pair_decomp10_native")  # reads past the cap added Oct 16: used only where no earlier read has the file
late = lambda r: any(t in r for t in LATE)
def load(model, cap, roots):
    g = lambda keep: sorted(f for r in roots if keep(r) for f in glob.glob(f"{r}/**/{model}_cap{cap}.jsonl", recursive=True))
    fs = g(lambda r: not late(r)) or g(late); assert fs, (model, cap)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip():
            r = json.loads(l)
            if r["kind"] in KINDS: by[(r["kind"], r["id"])].append(r)
    return by
ok = lambda r: r["score"] >= 1 - 1e-9
def cell(B, R, kind):
    keys = [k for k in B if k in R and k[0] == kind]; n = len(keys)
    N = [k for k in keys if all(r["finished"] for r in B[k]) and not any(ok(r) for r in B[k])]
    Nr = [k for k in keys if all(r["finished"] for r in R[k]) and not any(ok(r) for r in R[k])]
    C = [k for k in keys if k not in set(N)]
    reach = sum(ok(r) for k in N for r in R[k]) / (8 * n); mirror = sum(ok(r) for k in Nr for r in B[k]) / (8 * n)
    reach_robust = sum(sum(ok(r) for r in R[k]) for k in N if sum(ok(r) for r in R[k]) >= 4) / (8 * n); mirror_robust = sum(sum(ok(r) for r in B[k]) for k in Nr if sum(ok(r) for r in B[k]) >= 4) / (8 * n)
    def rates(M, ks):
        rows = [r for k in ks for r in M[k]]; f = sum(r["finished"] for r in rows); s = sum(ok(r) for r in rows)
        return f / len(rows), (s / f if f else 0.0)
    fb, ab = rates(B, C); fr, ar = rates(R, C); w = len(C) / n
    F = (fr - fb) * ar * w; P = fb * (ar - ab) * w
    dd = sum(ok(r) for k in keys for r in R[k]) / (8 * n) - sum(ok(r) for k in keys for r in B[k]) / (8 * n)
    assert abs(F + P + reach - dd) < 1e-9, (F + P + reach, dd)
    return {"d": dd, "F": F, "P": P, "reach": reach, "mirror": mirror, "reach_robust": reach_robust, "mirror_robust": mirror_robust, "n": n, "n_never": len(N), "n_never_rl": len(Nr),
            "rl_solved_on_never": sum(any(ok(r) for r in R[k]) for k in N), "base_solved_on_never_rl": sum(any(ok(r) for r in B[k]) for k in Nr)}
def pool(per):
    # Net reach: reach minus its mirror (the base's score on problems the RL model finished eight times and never solved). The
    # mirror's loss sits inside the reliability term, so it is moved there: P' = P + mirror, reach' = reach - mirror, and
    # F + P' + reach' is still the change exactly. Shares: sum|F|, sum|P'| and max(0, sum reach') over their total.
    sF = sum(abs(v["F"]) for v in per.values()); sP = sum(abs(v["P"] + v["mirror"]) for v in per.values())
    net = max(0.0, sum(v["reach"] - v["mirror"] for v in per.values())); tot = sF + sP + net
    rob = min(net, max(0.0, sum(v["reach_robust"] - v["mirror_robust"] for v in per.values())))
    return {"F": sF / tot, "P": sP / tot, "reach": net / tot, "reach_robust": rob / tot,
            "n": len({k.split("|")[0] for k in per}), "cells": len(per), "reach_raw": sum(v["reach"] for v in per.values()) / len(per),
            "mirror_raw": sum(v["mirror"] for v in per.values()) / len(per), "rl_cells": sum(v["rl_solved_on_never"] for v in per.values()),
            "base_cells": sum(v["base_solved_on_never_rl"] for v in per.values())}
out = {"budget": {}, "cap": {}}
for cap in (4096, 8192, 16384):
    per = {}
    for base, recipes in BASES.items():
        B = load(base, cap, ROOTS)
        for rec in recipes:
            R = load(rec, cap, ROOTS)
            for kind in KINDS: per[f"{SHORT[rec]}|{kind}"] = cell(B, R, kind)
    out["budget"][cap] = {"recipes": per, "pooled": pool(per)}
    print(cap, {k: round(v, 3) for k, v in out["budget"][cap]["pooled"].items()})
cs = json.load(open(CAPJ))["recipes"]; R64 = [r for r in ROOTS if "cap64" in r or "pair_decomp5" in r or late(r)] or ROOTS
for which in ("at", "beyond"):
    per = {}
    for rec, c in cs.items():
        if c.get("below_only"): continue  # read below its training cap: not an at-cap cell
        rl, base = PAIRS[rec][0], PAIRS[rec][1]; cap = c[f"{which}_cap"]
        B, R = load(base, cap, R64), load(rl, cap, R64)
        for kind in KINDS:
            if not any(k[0] == kind and k in R for k in B): continue  # read on one benchmark only (the Polaris pairs past 32k: physics)
            per[f"{rec}|{kind}"] = cell(B, R, kind) | {"cap": cap}
    out["cap"][which] = {"recipes": per, "pooled": pool(per)}
    print(which, {k: round(v, 3) for k, v in out["cap"][which]["pooled"].items()}, {k: (round(v["F"], 3), round(v["P"], 3), round(v["reach"], 3), round(v["mirror"], 3)) for k, v in per.items()})
out["kinds"] = list(KINDS); json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)
