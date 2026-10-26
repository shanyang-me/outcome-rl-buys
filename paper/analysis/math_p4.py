"""Prediction P4 of the competition-math read: net reach on the 2026 contests (kind post, problems that postdate every model's
release) is no larger than on the 2025 contests (kind pre). For each recipe at its cap this bootstraps the difference
reach(post) - reach(pre), resampling problems within each stratum independently (the strata are different problems, so the
interval is a two-sample one), and the same for robust reach and the change in pass@1. P4 holds for a recipe when the
difference's interval contains zero or lies below it. Same loading and scoring as hard_reach.py (JUDGED switches the judged
scores on, on both sides of a pair or not at all).
usage: [JUDGED=dir] [NBOOT=n] python math_p4.py out.json rows_root"""
import json, glob, sys, os, collections
import numpy as np
OUT, ROOT = sys.argv[1], sys.argv[2]; NB = int(os.environ.get("NBOOT", "5000"))
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def caps_for(model): return sorted({int(os.path.basename(f).split("_cap")[1].split(".")[0]) for f in glob.glob(f"{ROOT}/**/{model}_cap*.jsonl", recursive=True)})
def load(model, cap):
    fs = sorted(glob.glob(f"{ROOT}/**/{model}_cap{cap}.jsonl", recursive=True)); assert fs, (model, cap)
    by = collections.defaultdict(list); jm = None
    if os.environ.get("JUDGED") and os.path.exists(f"{os.environ['JUDGED']}/{model}.jsonl"):
        jm = {(r["id"], r["k"]): r["score_j"] for r in (json.loads(l) for l in open(f"{os.environ['JUDGED']}/{model}.jsonl") if l.strip())}
    miss = 0
    for l in open(fs[0]):
        if l.strip():
            r = json.loads(l)
            if jm is not None:
                if r["finished"] and (r["id"], r["k"]) not in jm: miss += 1
                r["score"] = jm.get((r["id"], r["k"]), r["score"]) if r["finished"] else 0.0
            by[(r["kind"], r["id"])].append(r)
    if miss: raise SystemExit(f"{model}: the judged file lacks {miss} finished samples; refusing to mix judged and matcher scores")
    return by
def arrays(B, R, kind):
    keys = sorted(k for k in B if k in R and k[0] == kind); ok = lambda r: r["score"] >= 1 - 1e-9
    a = lambda M, f: np.array([sum(f(r) for r in M[k]) for k in keys], float)
    bf, rf, bs, rs = a(B, lambda r: r["finished"]), a(R, lambda r: r["finished"]), a(B, ok), a(R, ok); nb, nr = a(B, lambda r: 1), a(R, lambda r: 1)
    return dict(n=len(keys), bs=bs, rs=rs, never=(bf == nb) & (bs == 0), never_rl=(rf == nr) & (rs == 0))
def cell(c, w):
    n = w.sum(); W = lambda x, m: float((w * x * m).sum())
    reach = (W(c["rs"], c["never"]) - W(c["bs"], c["never_rl"])) / (8 * n)
    rob = (W(c["rs"] >= 4, c["never"]) - W(c["bs"] >= 4, c["never_rl"])) / n
    dd = float((w * (c["rs"] - c["bs"])).sum()) / (8 * n)
    return np.array([reach, rob, dd])
NAMES = ["reach_diff", "robust_diff", "d_diff"]
rng = np.random.default_rng(0); res = {}
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    common = sorted(set(caps_for(rl)) & set(caps_for(base)))
    if not common: continue
    if os.environ.get("JUDGED"):
        missing = [m for m in (rl, base) if not os.path.exists(os.path.join(os.environ["JUDGED"], m.split("/")[-1] + ".jsonl"))]
        if missing: print(f"{name:14s} skipped: no judged rows for {' and '.join(missing)}"); continue
    at = min((c for c in common if c >= cap), default=max(common))
    B, R = load(base, at), load(rl, at); pre, post = arrays(B, R, "pre"), arrays(B, R, "post")
    if pre["n"] == 0 or post["n"] == 0: continue
    pt = cell(post, np.ones(post["n"])) - cell(pre, np.ones(pre["n"])); boots = np.empty((NB, len(pt)))
    for b in range(NB):
        wa = np.bincount(rng.integers(0, pre["n"], pre["n"]), minlength=pre["n"]).astype(float)
        wb = np.bincount(rng.integers(0, post["n"], post["n"]), minlength=post["n"]).astype(float)
        boots[b] = cell(post, wb) - cell(pre, wa)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    o = {k: [float(pt[i]), float(lo[i]), float(hi[i])] for i, k in enumerate(NAMES)}
    o.update(at_cap=at, n_pre=pre["n"], n_post=post["n"], reach_pre=float(cell(pre, np.ones(pre["n"]))[0]), reach_post=float(cell(post, np.ones(post["n"]))[0]),
             p4=bool(lo[0] <= 0))  # the 2026 stratum does not show more reach than the 2025 one beyond the interval
    res[name] = o
    print(f"{name:13s} {at // 1024:>3d}k reach 2025 {o['reach_pre']:+.3f} 2026 {o['reach_post']:+.3f} diff {o['reach_diff'][0]:+.3f} [{o['reach_diff'][1]:+.3f},{o['reach_diff'][2]:+.3f}] "
          f"robust diff {o['robust_diff'][0]:+.3f} [{o['robust_diff'][1]:+.3f},{o['robust_diff'][2]:+.3f}] d diff {o['d_diff'][0]:+.3f} [{o['d_diff'][1]:+.3f},{o['d_diff'][2]:+.3f}] P4 {'holds' if o['p4'] else 'FAILS'}", flush=True)
ok = sum(o["p4"] for o in res.values()); print(f"P4: {ok} of {len(res)} recipes with the 2026-minus-2025 reach interval reaching zero or below")
json.dump(res, open(OUT, "w"), indent=1)
