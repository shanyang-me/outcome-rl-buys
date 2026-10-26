"""Intervals for Figure 1's budget rows (Oct 19), with the figure's own estimator: size shares |F|, |P'|, max(0, reach') pooled over
the eighteen recipes and both benchmarks, at 4k, 8k, 16k and 4k+8k together; crossed bootstrap over recipes and shared problems as in
fig1_ci.py. usage: python fig1_budget_ci.py out.json cap_split.json rows_root [rows_root ...]"""
import json, sys, os, glob, collections
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); OUT, ROOTS = sys.argv[1], sys.argv[3:]
src = open(os.path.join(H, "fig1_ci.py")).read(); exec(src[src.index("KINDS ="):src.index("cs = json.load")])
src = open(os.path.join(H, "overlap_per_recipe.py")).read(); exec(src[src.index("BASES = "):src.index("def load")])
rng = np.random.default_rng(0); res = {}
for label, caps in (("4k", (4096,)), ("8k", (8192,)), ("16k", (16384,)), ("4k+8k", (4096, 8192))):
    cells = {}
    for base, recipes in BASES.items():
        for cap in caps:
            B = load(base, cap, ROOTS)
            for rec in recipes:
                R = load(rec, cap, ROOTS)
                for kind in KINDS: cells[(rec, kind, cap)] = arrays(B, R, kind)
    recs = sorted({r for r, _, _ in cells}); point = pool([cell(c, np.ones(len(c["ids"]))) for c in cells.values()])
    ids = {kind: sorted({i for (r, k, _), c in cells.items() if k == kind for i in c["ids"]}) for kind in KINDS}
    boot = []
    for _ in range(2000):
        pick = rng.choice(recs, len(recs)); cnt = {kind: collections.Counter(rng.choice(ids[kind], len(ids[kind]))) for kind in KINDS}
        vals = []
        for r in pick:
            for (rr, kind, cap), c in cells.items():
                if rr != r: continue
                w = np.array([cnt[kind][i] for i in c["ids"]], float)
                if w.sum() and (w * ~c["never"]).sum(): vals.append(cell(c, w))
        boot.append(pool(vals))
    ci = {k: [[float(np.percentile([b[k][j] for b in boot], q)) for q in (2.5, 97.5)] for j in range(3)] for k in point}
    res[label] = {"point": {k: [float(x) for x in v] for k, v in point.items()}, "ci": ci, "recipes": len(recs), "cells": len(cells)}
    print(label, len(recs), "recipes", "size", " ".join(f"{p:.3f} [{a:.3f}, {b:.3f}]" for p, (a, b) in zip(point["size"], ci["size"])), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
