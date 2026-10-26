"""Sensitivity of the pooled at-cap and beyond-cap shares (Figure 1, bottom) to how the gain is split (Oct 19). Three splits,
all exact (the parts sum to the full-credit change): (1) the paper's, P = fin_base x d pass@1(B|fin) on the problems outside U
with the mirror added back, F the rest; (2) symmetric, the cross term shared (F_s = dfin x mean accuracy, P_s = mean fin x dacc);
(3) matched, the accuracy change taken only on problems both models finish in at least seven of eight samples, P_m = fin_base x
(dacc on those problems), F the remainder, so selection into the finished set cannot move P. Same cells, loader and crossed
bootstrap as fig1_ci.py. usage: KINDS=phys,math python split_sensitivity.py out.json cap_split.json rows_root..."""
import json, glob, sys, collections, os
import numpy as np
OUT, CAPJ, ROOTS = sys.argv[1], sys.argv[2], sys.argv[3:]
KINDS = tuple(os.environ.get("KINDS", "phys,math").split(",")); NB = int(os.environ.get("NBOOT", "2000"))
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
LATE = ("pair_decomp9_32b", "pair_decomp10_native"); late = lambda r: any(t in r for t in LATE)
def load(model, cap):
    g = lambda keep: sorted(f for r in ROOTS if keep(r) for f in glob.glob(f"{r}/**/{model}_cap{cap}.jsonl", recursive=True))
    fs = g(lambda r: not late(r)) or g(late); assert fs, (model, cap)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip():
            r = json.loads(l)
            if r["kind"] in KINDS: by[(r["kind"], r["id"])].append(r)
    return by
def arrays(B, R, kind):
    keys = sorted(k for k in B if k in R and k[0] == kind)
    a = lambda M, f: np.array([sum(f(r) for r in M[k]) for k in keys], float); ok = lambda r: r["score"] >= 1 - 1e-9
    bf, rf, bs, rs = a(B, lambda r: r["finished"]), a(R, lambda r: r["finished"]), a(B, ok), a(R, ok); nb, nr = a(B, lambda r: 1), a(R, lambda r: 1)
    return dict(ids=[k[1] for k in keys], bf=bf, rf=rf, bs=bs, rs=rs, nb=nb, nr=nr, never=(bf == nb) & (bs == 0), never_rl=(rf == nr) & (rs == 0), matched=(bf >= 7) & (rf >= 7))
def cell(c, w):
    n = w.sum(); C = ~c["never"]; W = lambda x, m=None: float((w * x * (1 if m is None else m)).sum())
    reach = W(c["rs"], c["never"]) / (8 * n); mirror = W(c["bs"], c["never_rl"]) / (8 * n)
    fb = W(c["bf"], C) / W(c["nb"], C); fr = W(c["rf"], C) / W(c["nr"], C)
    ab = W(c["bs"], C) / max(W(c["bf"], C), 1e-12); ar = W(c["rs"], C) / max(W(c["rf"], C), 1e-12); wc = W(np.ones_like(w), C) / n
    F = (fr - fb) * ar * wc; P = fb * (ar - ab) * wc; dC = F + P
    Fs = (fr - fb) * (ar + ab) / 2 * wc; Ps = (fb + fr) / 2 * (ar - ab) * wc
    Mm = C & c["matched"]
    if W(c["bf"], Mm) > 0 and W(c["rf"], Mm) > 0:
        abm = W(c["bs"], Mm) / W(c["bf"], Mm); arm = W(c["rs"], Mm) / W(c["rf"], Mm); Pm = fb * (arm - abm) * wc
    else: Pm = P
    Fm = dC - Pm
    return {"paper": (F, P + mirror), "symmetric": (Fs, Ps + mirror), "matched": (Fm, Pm + mirror), "reach": reach - mirror}
def pool(vals):
    Rn = np.array([v["reach"] for v in vals]); out = {}
    for k in ("paper", "symmetric", "matched"):
        F = np.array([v[k][0] for v in vals]); P = np.array([v[k][1] for v in vals]); tot = abs(F).sum() + abs(P).sum() + max(0, Rn.sum())
        net = F.sum() + P.sum() + Rn.sum()
        out[k] = {"size": (abs(F).sum() / tot, abs(P).sum() / tot, max(0, Rn.sum()) / tot), "signed": (F.sum() / net, P.sum() / net, Rn.sum() / net)}
    return out
cs = json.load(open(CAPJ))["recipes"]; rng = np.random.default_rng(0); res = {}
for which in ("at", "beyond"):
    cells = {}
    for rec, c in cs.items():
        if c.get("below_only"): continue
        rl, base = PAIRS[rec][0], PAIRS[rec][1]; cap = c[f"{which}_cap"]; B, R = load(base, cap), load(rl, cap)
        for kind in KINDS:
            if not any(k[0] == kind and k in R for k in B): continue
            cells[(rec, kind)] = arrays(B, R, kind)
    recs = sorted({r for r, _ in cells}); point = pool([cell(c, np.ones(len(c["ids"]))) for c in cells.values()])
    ids = {kind: sorted({i for (r, k), c in cells.items() if k == kind for i in c["ids"]}) for kind in KINDS}
    boot = []
    for _ in range(NB):
        pick = rng.choice(recs, len(recs)); cnt = {kind: collections.Counter(rng.choice(ids[kind], len(ids[kind]))) for kind in KINDS}; vals = []
        for r in pick:
            for kind in KINDS:
                if (r, kind) not in cells: continue
                c = cells[(r, kind)]; w = np.array([cnt[kind][i] for i in c["ids"]], float)
                if w.sum() and (w * ~c["never"]).sum(): vals.append(cell(c, w))
        boot.append(pool(vals))
    res[which] = {}
    for k in ("paper", "symmetric", "matched"):
        res[which][k] = {}
        for s in ("size", "signed"):
            pt = [float(x) for x in point[k][s]]; ci = [[float(np.percentile([b[k][s][j] for b in boot], q)) for q in (2.5, 97.5)] for j in range(3)]
            res[which][k][s] = {"point": pt, "ci": ci}
            print(f"{which:7s} {k:9s} {s:6s} " + " ".join(f"{100*p:.0f} [{100*a:.0f}, {100*b:.0f}]" for p, (a, b) in zip(pt, ci)))
json.dump(res, open(OUT, "w"), indent=1)
