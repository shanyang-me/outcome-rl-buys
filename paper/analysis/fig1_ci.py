"""Intervals for the at-cap and beyond-cap split of Figure 1 (Oct 19). Same cells as fig1_story.py (reach on problems the base finished
eight times and never solved, F and P on the rest, the mirror moved from P to reach), recomputed with problem weights so a crossed
bootstrap can resample recipes and, in the same replicate, the shared problems. Reports, pooled over recipe x benchmark cells:
size shares (|F|, |P'|, max(0, reach')) as in Figure 1, signed shares of the net change, the same under a symmetric split
(F_s = dfin x mean accuracy, P_s = mean fin x dacc, which removes the choice of where the cross term goes) and under a matched split
(the accuracy change taken only on problems both models finish in at least seven of eight samples, F the remainder, so selection
into the finished set cannot enter P). The main bootstrap (2,000 replicates, seed 0) is unchanged from the paper's; a second
bootstrap (seed 1) resamples the five bases instead of the recipes, with the problems, and is reported as "ci_base".
usage: KINDS=phys,math python fig1_ci.py out.json cap_split.json rows_root [rows_root ...]"""
import json, glob, sys, collections, os
import numpy as np
OUT, CAPJ, ROOTS = sys.argv[1], sys.argv[2], sys.argv[3:]
KINDS = tuple(os.environ.get("KINDS", "phys,math").split(","))
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
LATE = ("pair_decomp9_32b", "pair_decomp10_native"); late = lambda r: any(t in r for t in LATE)
def load(model, cap, roots):
    g = lambda keep: sorted(f for r in roots if keep(r) for f in glob.glob(f"{r}/**/{model}_cap{cap}.jsonl", recursive=True))
    fs = g(lambda r: not late(r)) or g(late); assert fs, (model, cap)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip():
            r = json.loads(l)
            if r["kind"] in KINDS: by[(r["kind"], r["id"])].append(r)
    return by
def arrays(B, R, kind):
    keys = sorted(k for k in B if k in R and k[0] == kind)
    a = lambda M, f: np.array([sum(f(r) for r in M[k]) for k in keys], float)
    ok = lambda r: r["score"] >= 1 - 1e-9
    bf, rf = a(B, lambda r: r["finished"]), a(R, lambda r: r["finished"]); bs, rs = a(B, ok), a(R, ok)
    nb, nr = a(B, lambda r: 1), a(R, lambda r: 1)
    never = (bf == nb) & (bs == 0); never_rl = (rf == nr) & (rs == 0)
    return dict(ids=[k[1] for k in keys], bf=bf, rf=rf, bs=bs, rs=rs, nb=nb, nr=nr, never=never, never_rl=never_rl, matched=(bf >= 7) & (rf >= 7))
def cell(c, w):
    n = w.sum(); C = ~c["never"]; W = lambda x, m=None: float((w * x * (1 if m is None else m)).sum())
    reach = W(c["rs"], c["never"]) / (8 * n); mirror = W(c["bs"], c["never_rl"]) / (8 * n)
    fb = W(c["bf"], C) / W(c["nb"], C); fr = W(c["rf"], C) / W(c["nr"], C)
    ab = W(c["bs"], C) / max(W(c["bf"], C), 1e-12); ar = W(c["rs"], C) / max(W(c["rf"], C), 1e-12); wc = W(np.ones_like(w), C) / n
    F = (fr - fb) * ar * wc; P = fb * (ar - ab) * wc
    Fs = (fr - fb) * (ar + ab) / 2 * wc; Ps = (fb + fr) / 2 * (ar - ab) * wc
    Mm = C & c["matched"]
    if W(c["bf"], Mm) > 0 and W(c["rf"], Mm) > 0:
        Pm = fb * (W(c["rs"], Mm) / W(c["rf"], Mm) - W(c["bs"], Mm) / W(c["bf"], Mm)) * wc
    else: Pm = P
    Fm = F + P - Pm
    return F, P + mirror, Fs, Ps + mirror, reach - mirror, Fm, Pm + mirror
def pool(vals):
    F, P, Fs, Ps, Rn, Fm, Pm = (np.array(x) for x in zip(*vals))
    size = lambda F, P: (lambda tot: (abs(F).sum() / tot, abs(P).sum() / tot, max(0, Rn.sum()) / tot))(abs(F).sum() + abs(P).sum() + max(0, Rn.sum()))
    net = F.sum() + P.sum() + Rn.sum(); sgn = (F.sum() / net, P.sum() / net, Rn.sum() / net); sgs = (Fs.sum() / net, Ps.sum() / net, Rn.sum() / net)
    out = dict(size=size(F, P), signed=sgn, size_sym=size(Fs, Ps), signed_sym=sgs, size_m=size(Fm, Pm), signed_m=(Fm.sum() / net, Pm.sum() / net, Rn.sum() / net))
    # reliability minus finishing, as a share, so the ordering of the two terms gets its own interval
    out["diff"] = (out["size"][1] - out["size"][0], out["signed"][1] - out["signed"][0], out["size_sym"][1] - out["size_sym"][0])
    return out
cs = json.load(open(CAPJ))["recipes"]; R64 = [r for r in ROOTS if "cap64" in r or "pair_decomp5" in r or late(r)] or ROOTS
res = {}; rng = np.random.default_rng(0); keep = {}
def replicate(cells, recs_picked, cnt):
    vals = []
    for r in recs_picked:
        for kind in KINDS:
            if (r, kind) not in cells: continue
            c = cells[(r, kind)]; w = np.array([cnt[kind][i] for i in c["ids"]], float)
            if w.sum() and (w * ~c["never"]).sum(): vals.append(cell(c, w))
    return pool(vals)
for which in ("at", "beyond"):
    cells = {}
    for rec, c in cs.items():
        if c.get("below_only"): continue
        rl, base = PAIRS[rec][0], PAIRS[rec][1]; cap = c[f"{which}_cap"]
        B, R = load(base, cap, R64), load(rl, cap, R64)
        for kind in KINDS:
            if not any(k[0] == kind and k in R for k in B): continue
            cells[(rec, kind)] = arrays(B, R, kind)
    recs = sorted({r for r, _ in cells}); point = pool([cell(c, np.ones(len(c["ids"]))) for c in cells.values()])
    ids = {kind: sorted({i for (r, k), c in cells.items() if k == kind for i in c["ids"]}) for kind in KINDS}
    boot = []
    for _ in range(2000):
        pick = rng.choice(recs, len(recs)); cnt = {kind: collections.Counter(rng.choice(ids[kind], len(ids[kind]))) for kind in KINDS}
        boot.append(replicate(cells, pick, cnt))
    ci = {k: [[float(np.percentile([b[k][j] for b in boot], q)) for q in (2.5, 97.5)] for j in range(3)] for k in point}
    res[which] = {"point": {k: [float(x) for x in v] for k, v in point.items()}, "ci": ci, "cells": len(cells), "recipes": len(recs)}
    keep[which] = (cells, recs, ids)
    print(which, len(recs), "recipes", len(cells), "cells")
    for k in point: print(f"  {k:10s}", " ".join(f"{p:.3f} [{a:.3f}, {b:.3f}]" for p, (a, b) in zip(point[k], ci[k])))
# base-clustered bootstrap: resample the bases (each with all its recipes) and the problems
rng2 = np.random.default_rng(1)
for which in ("at", "beyond"):
    cells, recs, ids = keep[which]; base_of = {r: PAIRS[r][3] for r in recs}; bases = sorted(set(base_of.values())); boot = []
    for _ in range(2000):
        pickB = list(rng2.choice(bases, len(bases))); pick = [r for b in pickB for r in recs if base_of[r] == b]
        cnt = {kind: collections.Counter(rng2.choice(ids[kind], len(ids[kind]))) for kind in KINDS}
        boot.append(replicate(cells, pick, cnt))
    res[which]["ci_base"] = {k: [[float(np.percentile([b[k][j] for b in boot], q)) for q in (2.5, 97.5)] for j in range(3)] for k in res[which]["point"]}
    res[which]["bases"] = len(bases)
    print(which, "bases resampled:", " | ".join(f"{k} " + " ".join(f"[{a:.2f}, {b:.2f}]" for a, b in res[which]["ci_base"][k]) for k in ("size", "signed")))
json.dump(res, open(OUT, "w"), indent=1)
