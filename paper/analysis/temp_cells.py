"""Authors'-temperature check (Oct 19). For a pair read twice, at T=0.6 (the paper's rows) and at the recipe authors' temperature
(pair_temp rows), compute at every shared budget the three-part split of Figure 1 on physics: F, P (with the mirror added back),
net reach, the change d, finishing and accuracy per finished derivation for both models, with paired bootstrap intervals over
problems. usage: python temp_cells.py RL BASE rows_root_T06 rows_root_authors [out.json]"""
import json, glob, sys, os, collections
import numpy as np
RL, BASE, R06, RA = sys.argv[1:5]; OUT = sys.argv[5] if len(sys.argv) > 5 else None
def load(root, model, cap):
    fs = sorted(glob.glob(f"{root}/**/{model}_cap{cap}.jsonl", recursive=True)); assert fs, (root, model, cap)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip():
            r = json.loads(l)
            if r["kind"] == "phys": by[r["id"]].append(r)
    return by
def arrays(B, R):
    keys = sorted(k for k in B if k in R); ok = lambda r: r["score"] >= 1 - 1e-9
    a = lambda M, f: np.array([sum(f(r) for r in M[k]) for k in keys], float)
    bf, rf, bs, rs = a(B, lambda r: r["finished"]), a(R, lambda r: r["finished"]), a(B, ok), a(R, ok)
    bm, rm = a(B, lambda r: r["score"]), a(R, lambda r: r["score"])  # partial credit, for the capped score
    nb, nr = a(B, lambda r: 1), a(R, lambda r: 1)
    return dict(n=len(keys), bf=bf, rf=rf, bs=bs, rs=rs, bm=bm, rm=rm, nb=nb, nr=nr, never=(bf == nb) & (bs == 0), never_rl=(rf == nr) & (rs == 0))
def cell(c, w):
    n = w.sum(); C = ~c["never"]; W = lambda x, m=None: float((w * x * (1 if m is None else m)).sum())
    reach = W(c["rs"], c["never"]) / (8 * n); mirror = W(c["bs"], c["never_rl"]) / (8 * n)
    fb = W(c["bf"], C) / W(c["nb"], C); fr = W(c["rf"], C) / W(c["nr"], C)
    ab = W(c["bs"], C) / max(W(c["bf"], C), 1e-12); ar = W(c["rs"], C) / max(W(c["rf"], C), 1e-12); wc = W(np.ones_like(w), C) / n
    F = (fr - fb) * ar * wc; P = fb * (ar - ab) * wc + mirror; Rn = reach - mirror
    d = (W(c["rs"]) - W(c["bs"])) / (8 * n)  # full-credit change over all problems = F + P + Rn
    fin_b, fin_r = W(c["bf"]) / W(c["nb"]), W(c["rf"]) / W(c["nr"])
    acc_b, acc_r = W(c["bs"]) / max(W(c["bf"]), 1e-12), W(c["rs"]) / max(W(c["rf"]), 1e-12)
    return dict(F=F, P=P, reach=Rn, d=d, fin_b=fin_b, fin_r=fin_r, acc_b=acc_b, acc_r=acc_r, U=int((w * c["never"]).sum()), Up=int((w * c["never_rl"]).sum()))
def with_ci(c, nboot=20000, seed=0):
    rng = np.random.default_rng(seed); n = c["n"]; pt = cell(c, np.ones(n)); boots = collections.defaultdict(list)
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float); b = cell(c, w)
        for k in ("F", "P", "reach", "d"): boots[k].append(b[k])
    for k in ("F", "P", "reach", "d"): pt[k + "_ci"] = [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]
    return pt
caps = sorted({int(os.path.basename(f).split("_cap")[1].split(".")[0]) for root in (R06, RA) for f in glob.glob(f"{root}/**/{RL}_cap*.jsonl", recursive=True)})
res = {}
for cap in caps:
    row = {}
    for lab, root in (("T0.6", R06), ("authors", RA)):
        try: c = arrays(load(root, BASE, cap), load(root, RL, cap))
        except AssertionError: continue
        row[lab] = with_ci(c)
    if len(row) == 2: res[cap] = row
f = lambda v, ci: f"{v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
for cap, row in res.items():
    print(f"--- cap {cap // 1024}k")
    for lab, p in row.items():
        print(f"  {lab:8s} fin {p['fin_b']:.3f}->{p['fin_r']:.3f} acc|fin {p['acc_b']:.3f}->{p['acc_r']:.3f} | F {f(p['F'], p['F_ci'])} P {f(p['P'], p['P_ci'])} reach {f(p['reach'], p['reach_ci'])} d {f(p['d'], p['d_ci'])} | U {p['U']} U' {p['Up']}")
if OUT: json.dump({str(k): v for k, v in res.items()}, open(OUT, "w"), indent=1)
