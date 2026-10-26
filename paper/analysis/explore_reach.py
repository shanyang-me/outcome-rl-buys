"""Does reach follow exploration? For each recipe at its training cap (physics + MATH-500, 64k read), on the shared region (problems
in neither N nor N', so reach cannot enter), compare the RL model with its base on mean score and on coverage (solved at least once
in eight). Sharpening raises the mean and lowers coverage; a recipe that keeps exploring holds coverage. breadth = d(coverage) - d(mean).
Then rank-correlate breadth with net reach (reach - mirror, Eq. 2) across recipes.
usage: python explore_reach.py out.json cap_split.json rows_root..."""
import json, glob, sys, collections, math, os
OUT, CAPJ, ROOTS = sys.argv[1], sys.argv[2], sys.argv[3:]
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def load(model, cap):
    fs = sorted(f for r in ROOTS for f in glob.glob(f"{r}/**/{model}_cap{cap}.jsonl", recursive=True)); assert fs, (model, cap)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return by
ok = lambda r: r["score"] >= 1 - 1e-9
cs = json.load(open(CAPJ))["recipes"]; out = {}
for rec, c in cs.items():
    if c.get("below_only"): continue
    rl, base, cap = PAIRS[rec][0], PAIRS[rec][1], c["at_cap"]
    try: B, R = load(base, cap), load(rl, cap)
    except AssertionError: continue
    keys = [k for k in B if k in R]
    never = lambda M, k: all(r["finished"] for r in M[k]) and not any(ok(r) for r in M[k])
    N = [k for k in keys if never(B, k)]; Nr = [k for k in keys if never(R, k)]
    C = [k for k in keys if k not in set(N) | set(Nr)]
    mean = lambda M, ks: sum(r["score"] for k in ks for r in M[k]) / sum(len(M[k]) for k in ks)
    cov = lambda M, ks: sum(any(ok(r) for r in M[k]) for k in ks) / len(ks)
    reach = sum(ok(r) for k in N for r in R[k]) / (8 * len(keys)); mirror = sum(ok(r) for k in Nr for r in B[k]) / (8 * len(keys))
    out[rec] = {"d_mean": mean(R, C) - mean(B, C), "d_cov": cov(R, C) - cov(B, C), "net_reach": reach - mirror, "n": len(keys), "cap": cap}
    out[rec]["breadth"] = out[rec]["d_cov"] - out[rec]["d_mean"]
def rank(v): s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); [r.__setitem__(i, k) for k, i in enumerate(s)]; return r
def sp(x, y):
    a, b = rank(x), rank(y); n = len(a); ma, mb = sum(a) / n, sum(b) / n
    return sum((i - ma) * (j - mb) for i, j in zip(a, b)) / math.sqrt(sum((i - ma) ** 2 for i in a) * sum((j - mb) ** 2 for j in b))
ks = sorted(out, key=lambda k: -out[k]["net_reach"])
for k in ks: v = out[k]; print(f"{k:13s} net reach {100*v['net_reach']:+.2f} pts | shared region: d mean {v['d_mean']:+.3f} d coverage {v['d_cov']:+.3f} breadth {v['breadth']:+.3f}")
rho = sp([out[k]["breadth"] for k in ks], [out[k]["net_reach"] for k in ks]); rho2 = sp([out[k]["d_cov"] for k in ks], [out[k]["net_reach"] for k in ks])
# permutation p for rho
import random; random.seed(0); xs = [out[k]["breadth"] for k in ks]; ys = [out[k]["net_reach"] for k in ks]
p = sum(abs(sp(random.sample(xs, len(xs)), ys)) >= abs(rho) for _ in range(5000)) / 5000
print(f"spearman(breadth, net reach) = {rho:.2f} (perm p {p:.3f}); spearman(d coverage, net reach) = {rho2:.2f}; n = {len(ks)}")
json.dump({"recipes": out, "rho_breadth": rho, "p": p, "rho_cov": rho2}, open(OUT, "w"), indent=1)
