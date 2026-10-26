"""Harder-set cells by source (Oct 20): for every recipe whose two models are read (and, with JUDGED, judged) on the harder set, at the
recipe's cap, per source of problems (OlympiadBench physics, PHYBench, the PHYSICS olympiad tier, FrontierScience; the unfiltered
UGPhysics and PHYSICS draws): finishing, pass@1 per finished derivation, net reach with a paired bootstrap interval, and the problem
counts behind reach. The sources' release dates stratify contamination (OlympiadBench before every base, PHYBench and PHYSICS
alongside the bases, FrontierScience after every model).
usage: [JUDGED=dir] [SOURCES=kind|src,...] python hard_by_source.py out.json rows_root [rows_root ...]   (needs problems.jsonl under a rows_root)"""
import json, glob, sys, os, collections
import numpy as np
OUT, ROOTS = sys.argv[1], sys.argv[2:]; NB = int(os.environ.get("NBOOT", "2000"))
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
pf = next(f for r in ROOTS for f in glob.glob(f"{r}/**/problems.jsonl", recursive=True))
SRC = {}
for l in open(pf):
    if l.strip(): it = json.loads(l); SRC[(it["kind"], it["id"])] = f"{it['kind']}|{it['src']}"
ORDER = os.environ.get("SOURCES", "hard|OB-phys-en,hard|PHYBench,hard|PHYSICS-test,hard|FrontierSci-phys-short,refill|UGPhysics-en-1part,refill|PHYSICS-test").split(",")  # kind|src cells, in order; SOURCES overrides (the competition-math read)
def caps_for(model):
    return sorted({int(os.path.basename(f).split("_cap")[1].split(".")[0]) for r in ROOTS for f in glob.glob(f"{r}/**/{model}_cap*.jsonl", recursive=True)})
def load(model, cap):
    fs = sorted(f for r in ROOTS for f in glob.glob(f"{r}/**/{model}_cap{cap}.jsonl", recursive=True)); assert fs, (model, cap)
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
    if miss: raise SystemExit(f"{model}: the judged file lacks {miss} finished samples (stale or truncated); refusing to mix judged and matcher scores")
    return by
def arrays(B, R, keys):
    ok = lambda r: r["score"] >= 1 - 1e-9
    a = lambda M, f: np.array([sum(f(r) for r in M[k]) for k in keys], float)
    bf, rf, bs, rs = a(B, lambda r: r["finished"]), a(R, lambda r: r["finished"]), a(B, ok), a(R, ok); nb, nr = a(B, lambda r: 1), a(R, lambda r: 1)
    return dict(n=len(keys), bf=bf, rf=rf, bs=bs, rs=rs, nb=nb, nr=nr, never=(bf == nb) & (bs == 0), never_rl=(rf == nr) & (rs == 0))
def cell(c, w):
    n = w.sum(); W = lambda x, m=None: float((w * x * (1 if m is None else m)).sum())
    reach = W(c["rs"], c["never"]) / (8 * n); mirror = W(c["bs"], c["never_rl"]) / (8 * n)
    return np.array([reach - mirror, (W(c["rs"]) - W(c["bs"])) / (8 * n), W(c["bf"]) / W(c["nb"]), W(c["rf"]) / W(c["nr"]),
                     W(c["bs"]) / max(W(c["bf"]), 1e-12), W(c["rs"]) / max(W(c["rf"]), 1e-12)])
NAMES = ["reach", "d", "fin_b", "fin_r", "acc_b", "acc_r"]
rng = np.random.default_rng(0); res = {}
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    common = sorted(set(caps_for(rl)) & set(caps_for(base)))
    if not common: continue
    if os.environ.get("JUDGED"):
        missing = [m for m in (rl, base) if not os.path.exists(os.path.join(os.environ["JUDGED"], m + ".jsonl"))]
        if missing: print(f"{name:14s} skipped: no judged rows for {' and '.join(missing)}"); continue
    at = min((c for c in common if c >= cap), default=max(common)); B, R = load(base, at), load(rl, at)
    res[name] = {"base": tag, "at_cap": at, "cells": {}}
    for s in ORDER:
        keys = sorted(k for k in B if k in R and SRC.get(k) == s)
        if not keys: continue
        c = arrays(B, R, keys); n = c["n"]; pt = cell(c, np.ones(n)); boots = np.empty((NB, len(pt)))
        for b in range(NB): boots[b] = cell(c, np.bincount(rng.integers(0, n, n), minlength=n).astype(float))
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        o = {k: [float(pt[i]), float(lo[i]), float(hi[i])] for i, k in enumerate(NAMES)}
        o.update(n=n, U=int(c["never"].sum()), rl_on_U=int(((c["rs"] > 0) & c["never"]).sum()), rl_robust_on_U=int(((c["rs"] >= 4) & c["never"]).sum()),
                 Up=int(c["never_rl"].sum()), base_on_Up=int(((c["bs"] > 0) & c["never_rl"]).sum()))
        res[name]["cells"][s] = o
        print(f"{name:13s} {at // 1024:>3d}k {s:28s} n={n:3d} fin {o['fin_b'][0]:.2f}->{o['fin_r'][0]:.2f} acc|fin {o['acc_b'][0]:.3f}->{o['acc_r'][0]:.3f} "
              f"d {o['d'][0]:+.3f} reach {o['reach'][0]:+.3f} [{o['reach'][1]:+.3f},{o['reach'][2]:+.3f}] | U {o['U']} RL>=1 {o['rl_on_U']} >=4 {o['rl_robust_on_U']} | U' {o['Up']} base>=1 {o['base_on_Up']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
