"""Length change measured without the finishing confound: mean token length of FINISHED samples on problems both models finish
>= 7 of 8 (n >= 15), per cell; recipe = median log ratio over its cells. Spearman over recipes against the mean finishing change at
4k and 8k, exact permutation p. usage: python length_matched.py rows_dir pairs_ci.json stats.json"""
import json, glob, sys, collections, itertools, math
import numpy as np
from scipy import stats
D, J, OUTS = sys.argv[1], sys.argv[2], sys.argv[3]
from families import selected, perm_p
FAM = selected()
def load(m, cap):
    fs = glob.glob(f"{D}/*/{m}_cap{cap}.jsonl")
    if not fs: return None
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return by
cellr = collections.defaultdict(list); detail = []
for base, recs in FAM.items():
    for cap in (4096, 8192, 16384):
        Bm = load(base, cap)
        if not Bm: continue
        for rec in recs:
            Rm = load(rec, cap)
            if not Rm: continue
            for kind in ("phys", "math"):
                keys = [k for k in Bm if k[0] == kind and k in Rm and sum(x["finished"] for x in Bm[k]) >= 7 and sum(x["finished"] for x in Rm[k]) >= 7]
                if len(keys) < 15: continue
                lb = np.mean([x["len"] for k in keys for x in Bm[k] if x["finished"]]); lr = np.mean([x["len"] for k in keys for x in Rm[k] if x["finished"]])
                cellr[rec].append(math.log(lr / lb)); detail.append((rec, kind, cap, len(keys), lr / lb))
d = json.load(open(J))
recs = sorted(cellr); x = [float(np.median(cellr[r])) for r in recs]
y = [float(np.mean([v["finished"]["delta"] for v in d.values() if v["rl"] == r and v["cap"] in (4096, 8192)])) for r in recs]
rho = stats.spearmanr(x, y).statistic
pval, how = perm_p(x, y, rho)
for r, a, b in zip(recs, x, y): print(f"  {r:40s} matched finished-length ratio {math.exp(a):.2f}x  dfinished(4k,8k) {b:+.3f}  cells {[ (k, c, n, round(q, 2)) for (rr, k, c, n, q) in detail if rr == r]}")
print(f"Spearman {rho:+.3f}, p {pval:.4f} ({how}), n={len(recs)}")
st = json.load(open(OUTS)); st["length_vs_finishing_matched"] = {"x_median_matched_finished_len_ratio": {r: math.exp(a) for r, a in zip(recs, x)}, "y_mean_dfinished_4k8k": dict(zip(recs, y)), "spearman": rho, "p_exact_two_sided": pval, "p_method": how}
json.dump(st, open(OUTS, "w"), indent=1)
