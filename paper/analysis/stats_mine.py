"""Recipe-level significance for the released-model claim.
Unit of replication: the recipe (released RL model). Crossed bootstrap: each replicate resamples recipes with replacement AND problems
with replacement (one draw of the 300 physics and 500 MATH ids shared by all cells, since every model answers the same problems).
Exact decomposition per cell: dmean = F + P, F = dfinished * acc_RL, P = finished_base * dacc (acc = accuracy given finished).
usage: python stats_mine.py rows_dir out.json"""
import json, glob, sys, collections, itertools, math, os
import numpy as np
from scipy import stats
D = sys.argv[1]; OUT = sys.argv[2]
from families import selected, perm_p
FAM = selected()
CAPS = (4096, 8192, 16384); KINDS = ("phys", "math")
def load(model, cap):
    fs = glob.glob(f"{D}/*/{model}_cap{cap}.jsonl")
    if not fs: return None
    assert len(fs) == 1, fs
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return by
ids = {}; cells = []   # cell: (recipe, kind, cap, arrays)
for base, recs in FAM.items():
    for cap in CAPS:
        B0 = load(base, cap)
        if not B0: continue
        for rec in recs:
            R0 = load(rec, cap)
            if not R0: continue
            for kind in KINDS:
                key = sorted(k for k in B0 if k[0] == kind and k in R0)
                ids.setdefault(kind, key)
                assert key == ids[kind], (rec, cap, kind, "problem set differs")
                arr = lambda S, f: np.array([f(S[k]) for k in key], dtype=float)
                c = {"rec": rec, "kind": kind, "cap": cap}
                for tag, S in (("b", B0), ("r", R0)):
                    c[tag + "n"] = arr(S, len); c[tag + "f"] = arr(S, lambda v: sum(x["finished"] for x in v))
                    c[tag + "s"] = arr(S, lambda v: sum(x["score"] for x in v)); c[tag + "sf"] = arr(S, lambda v: sum(x["score"] for x in v if x["finished"]))
                c["mask"] = (c["bf"] >= 7) & (c["rf"] >= 7)
                cells.append(c)
recipes = sorted({c["rec"] for c in cells}); R = len(recipes); print(f"{len(cells)} cells, {R} recipes, problems {[(k, len(v)) for k, v in ids.items()]}")
def cell_stats(c, w):
    W = lambda x: float((w * x).sum())
    nb, nr = W(c["bn"]), W(c["rn"]); fb, fr = W(c["bf"]) / nb, W(c["rf"]) / nr
    ab, ar = W(c["bsf"]) / max(W(c["bf"]), 1e-9), W(c["rsf"]) / max(W(c["rf"]), 1e-9)
    mb, mr = W(c["bs"]) / nb, W(c["rs"]) / nr
    F = (fr - fb) * ar; P = fb * (ar - ab)
    m = c["mask"]; mm = None
    if (w * m).sum() >= 1 and W(m * c["bf"]) > 0 and W(m * c["rf"]) > 0:
        mm = W(m * c["rsf"]) / W(m * c["rf"]) - W(m * c["bsf"]) / W(m * c["bf"])
    return mr - mb, F, P, fr - fb, mm
ones = {k: np.ones(len(v)) for k, v in ids.items()}
point = [cell_stats(c, ones[c["kind"]]) for c in cells]
for c, (d, F, P, df, mm) in zip(cells, point): assert abs(d - F - P) < 1e-9
_MS, _MC = os.environ.get("MATCH_SET"), os.environ.get("MATCH_CAPS")   # optional: pool matched changes over one set / some caps only
matched_ok = [c["mask"].sum() >= 15 and (not _MS or c["kind"] == _MS) and (not _MC or str(c["cap"]) in _MC.split(",")) for c in cells]   # same reporting rule as the paper (n >= 15 at the point estimate)
lenr = {}
for c in cells:
    lb = (c["bn"] * 0).sum()
def recipe_summ(stats_list, mult=None):
    """stats_list aligned with cells; mult: recipe -> multiplicity (crossed bootstrap)."""
    mult = mult or {r: 1 for r in recipes}
    per = {r: {"absF": 0.0, "absP": 0.0, "P2": 0.0, "D2": 0.0, "mm": []} for r in recipes}
    for c, (d, F, P, df, mm), ok in zip(cells, stats_list, matched_ok):
        q = per[c["rec"]]; q["absF"] += abs(F); q["absP"] += abs(P); q["P2"] += P * P; q["D2"] += d * d
        if ok and mm is not None: q["mm"].append(mm)
    dom = {r: q["absF"] / (q["absF"] + q["absP"]) if q["absF"] + q["absP"] > 0 else float("nan") for r, q in per.items()}
    tot = lambda key: sum(mult[r] * per[r][key] for r in recipes)
    pooled_dom = tot("absF") / (tot("absF") + tot("absP"))
    r2 = 1 - tot("P2") / tot("D2")
    mmr = {r: (sum(q["mm"]) / len(q["mm"]) if q["mm"] else None) for r, q in per.items()}
    pooled_mm = sum(mult[r] * mmr[r] for r in recipes if mmr[r] is not None) / sum(mult[r] for r in recipes if mmr[r] is not None)
    return dom, pooled_dom, r2, mmr, pooled_mm
CAPSETS = {"4k": {4096}, "8k": {8192}, "16k": {16384}, "4k+8k": {4096, 8192}}
_DS = os.environ.get("DOM_SET")   # optional: dominance over one set only (phys | math); the registered analysis pools both
def capset_dom(stats_list, capset, mult=None):
    mult = mult or {r: 1 for r in recipes}
    aF = {r: 0.0 for r in recipes}; aP = {r: 0.0 for r in recipes}
    for c, (d, F, P, df, mm) in zip(cells, stats_list):
        if c["cap"] in capset and (not _DS or c["kind"] == _DS): aF[c["rec"]] += abs(F); aP[c["rec"]] += abs(P)
    per = {r: aF[r] / (aF[r] + aP[r]) for r in recipes if aF[r] + aP[r] > 0}
    tf = sum(mult[r] * aF[r] for r in per); tp = sum(mult[r] * aP[r] for r in per)
    return per, (tf / (tf + tp) if tf + tp > 0 else float("nan"))
cap_point = {k: capset_dom(point, v) for k, v in CAPSETS.items()}
cap_boot = {k: [] for k in CAPSETS}; cap_boot_r = {k: {r: [] for r in recipes} for k in CAPSETS}
dom, pdom, r2, mmr, pmm = recipe_summ(point)
print("per-recipe dominance |F|/(|F|+|P|):", {r[:22]: round(v, 3) for r, v in dom.items()})
print(f"pooled dominance {pdom:.3f}; uncentred R2 of dmean on F alone {r2:.3f}; pooled matched change (mean of recipe means) {pmm:+.4f}")
# crossed bootstrap
rng = np.random.default_rng(0); B = 4000
bd, br2, bmm = [], [], []; bdom_r = {r: [] for r in recipes}; bmm_r = {r: [] for r in recipes}
for b in range(B):
    w = {k: np.bincount(rng.integers(0, len(v), len(v)), minlength=len(v)).astype(float) for k, v in ids.items()}
    draw = rng.integers(0, R, R); mult = {r: 0 for r in recipes}
    for i in draw: mult[recipes[i]] += 1
    st_b = [cell_stats(c, w[c["kind"]]) for c in cells]
    d_b, pd_b, r2_b, mmr_b, pmm_b = recipe_summ(st_b, mult)
    bd.append(pd_b); br2.append(r2_b); bmm.append(pmm_b)
    for k, v in CAPSETS.items():
        cap_boot[k].append(capset_dom(st_b, v, mult)[1])
        for r, x in capset_dom(st_b, v)[0].items(): cap_boot_r[k][r].append(x)
    d_p, _, _, mmr_p, _ = recipe_summ(st_b)       # problem-only bootstrap for per-recipe CIs
    for r in recipes:
        bdom_r[r].append(d_p[r]);
        if mmr_p[r] is not None: bmm_r[r].append(mmr_p[r])
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
res = {"n_cells": len(cells), "n_recipes": R,
       "pooled_dominance": pdom, "pooled_dominance_ci_crossed": ci(bd), "pooled_dominance_p_le_half": float(np.mean(np.array(bd) <= 0.5)),
       "r2_dmean_on_F": r2, "r2_ci_crossed": ci(br2),
       "pooled_matched_change": pmm, "pooled_matched_ci_crossed": ci(bmm),
       "per_recipe": {r: {"dominance": dom[r], "dominance_ci_problems": ci(bdom_r[r]), "matched_mean": mmr[r], "matched_ci_problems": ci(bmm_r[r]) if bmm_r[r] else None} for r in recipes}}
# recipe-level sign test: dominance > 1/2 with the problem-bootstrap CI excluding 1/2
k_point = sum(dom[r] > 0.5 for r in recipes); k_sig = sum(ci(bdom_r[r])[0] > 0.5 for r in recipes)
res["sign_test_point"] = {"k": k_point, "n": R, "p_one_sided": float(stats.binomtest(k_point, R, 0.5, alternative="greater").pvalue)}
res["recipes_dominance_ci_above_half"] = k_sig
# random-effects (DerSimonian-Laird) pooling of the recipe matched means, SE from the problem bootstrap
y = np.array([mmr[r] for r in recipes if mmr[r] is not None]); se = np.array([np.std(bmm_r[r]) for r in recipes if mmr[r] is not None])
wv = 1 / se**2; ybar = (wv * y).sum() / wv.sum(); Q = (wv * (y - ybar) ** 2).sum(); k = len(y)
tau2 = max(0.0, (Q - (k - 1)) / (wv.sum() - (wv**2).sum() / wv.sum())); ws = 1 / (se**2 + tau2); mu = (ws * y).sum() / ws.sum(); semu = math.sqrt(1 / ws.sum())
res["re_meta_matched"] = {"mu": mu, "ci": [mu - 1.96 * semu, mu + 1.96 * semu], "tau2": tau2, "Q": Q, "k": k, "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0}
q_hk = float((ws * (y - mu) ** 2).sum() / (k - 1)); se_hk = math.sqrt(q_hk / ws.sum()); tq = stats.t.ppf(0.975, k - 1)
res["re_meta_matched"]["hksj_ci"] = [mu - tq * se_hk, mu + tq * se_hk]
res["re_meta_matched"]["hksj_p_two_sided"] = float(2 * stats.t.sf(abs(mu) / se_hk, k - 1))
FLAGS = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
def changing_in(capset):
    return {r for r in recipes if any(v["rl"] == r and v["cap"] in capset and (v["capped_gain"] or v["capped_loss"]) for v in FLAGS.values())}
res["dominance_by_capset"] = {}
for kk, (per, pooled) in cap_point.items():
    # a recipe changes in this cap set if one of its cells there has a capped change whose printed interval excludes zero (pairs_ci flags)
    changing = [r for r in per if r in changing_in(CAPSETS[kk])] if FLAGS else [r for r in per if r != "Open-RS3"]
    res["dominance_by_capset"][kk] = {"pooled": pooled, "ci_crossed": ci(cap_boot[kk]), "per_recipe": per,
        "per_recipe_ci_problems": {r: ci(cap_boot_r[kk][r]) for r in per},
        "k_above_half_changing": sum(per[r] > 0.5 for r in changing), "n_changing": len(changing),
        "sign_p_changing": float(stats.binomtest(sum(per[r] > 0.5 for r in changing), len(changing), 0.5, alternative="greater").pvalue)}
# length vs finishing, recipe level: median log length ratio vs mean dfinished at 4k and 8k
x, yy = [], []
for r in recipes:
    cs = [(c, p) for c, p in zip(cells, point) if c["rec"] == r]
    lr = [math.log((c["rn"] * 0 + 1).sum()) for c, p in cs]   # placeholder replaced below
lens = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
if lens:
    for r in recipes:
        lr = [math.log(v["len_ratio"]) for v in lens.values() if v["rl"] == r]
        df = [v["finished"]["delta"] for v in lens.values() if v["rl"] == r and v["cap"] in (4096, 8192)]
        x.append(float(np.median(lr))); yy.append(float(np.mean(df)))
    rho = stats.spearmanr(x, yy).statistic
    pv, how = perm_p(x, yy, rho)
    res["length_vs_finishing"] = {"x_median_log_len_ratio": dict(zip(recipes, x)), "y_mean_dfinished_4k8k": dict(zip(recipes, yy)), "spearman": rho,
                                  "p_exact_two_sided": pv, "p_method": how}
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res.items():
    if k != "per_recipe": print(k, ":", v)
for r, v in res["per_recipe"].items(): print(f"  {r:40s} dominance {v['dominance']:.3f} [{v['dominance_ci_problems'][0]:.3f},{v['dominance_ci_problems'][1]:.3f}]  matched {('%+.4f' % v['matched_mean']) if v['matched_mean'] is not None else '-'} {('[%+.4f,%+.4f]' % tuple(v['matched_ci_problems'])) if v['matched_ci_problems'] else ''}")
