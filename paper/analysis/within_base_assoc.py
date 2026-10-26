"""Length-finishing association within bases (exploratory, not registered): base-demeaned Spearman and a permutation test that
shuffles the finishing change only within each base, so the contrast between bases is held fixed.
usage: python within_base_assoc.py stats_all_final.json   (reads length_vs_finishing_matched written by length_matched.py)"""
import json, sys, math, collections
import numpy as np
from scipy import stats
from families import FAMILIES
lv = json.load(open(sys.argv[1]))["length_vs_finishing_matched"]
base_of = {r: b for b, rs in FAMILIES.items() for r in rs}
recs = sorted(lv["x_median_matched_finished_len_ratio"])
x = np.array([math.log(lv["x_median_matched_finished_len_ratio"][r]) for r in recs]); y = np.array([lv["y_mean_dfinished_4k8k"][r] for r in recs])
groups = collections.defaultdict(list)
for i, r in enumerate(recs): groups[base_of[r]].append(i)
rho = stats.spearmanr(x, y).statistic
g = np.random.default_rng(7); N = 100000; hits = 0
for _ in range(N):
    yy = y.copy()
    for idx in groups.values(): yy[idx] = y[g.permutation(idx)]
    hits += abs(stats.spearmanr(x, yy).statistic) >= abs(rho) - 1e-12
xd, yd = x.copy(), y.copy()
for idx in groups.values(): xd[idx] -= x[idx].mean(); yd[idx] -= y[idx].mean()
print(f"Spearman over {len(recs)} recipes {rho:+.3f}; within-base permutation p {hits / N:.4f}; base-demeaned Spearman {stats.spearmanr(xd, yd).statistic:+.3f}")
