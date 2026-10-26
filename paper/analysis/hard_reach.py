"""Harder-set read (Oct 20): for every recipe whose RL model and base are both read on the 671-problem set, at the recipe's training
cap (the largest shared budget at or above it) and at 16k, on the 471 harder problems ('hard'), the 200 unfiltered pool problems
('refill') and both together: finishing, pass@1(B|fin), the three-part split (F, P with the mirror added back, reach net of its
mirror), robust reach (wins in at least four of eight), the problem counts behind reach (|U|, RL wins on U, |U'|, base wins on U'),
with paired bootstrap intervals over problems. The refill rows against the 300 filtered problems give the filter's effect.
usage: [JUDGED=dir] python hard_reach.py out.json rows_root [rows_root ...]   (rows: <model>_cap<cap>.jsonl from pair_hard.py; JUDGED: judged/<model>.jsonl)"""
import json, glob, sys, os, collections
import numpy as np
OUT, ROOTS = sys.argv[1], sys.argv[2:]; NB = int(os.environ.get("NBOOT", "5000")); KINDS = os.environ.get("KINDS", "hard,refill").split(",")
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def caps_for(model):
    return sorted({int(os.path.basename(f).split("_cap")[1].split(".")[0]) for r in ROOTS for f in glob.glob(f"{r}/**/{model}_cap*.jsonl", recursive=True)})
def load(model, cap):
    fs = sorted(f for r in ROOTS for f in glob.glob(f"{r}/**/{model}_cap{cap}.jsonl", recursive=True)); assert fs, (model, cap)
    by = collections.defaultdict(list); jm = None
    if os.environ.get("JUDGED") and os.path.exists(f"{os.environ['JUDGED']}/{model}.jsonl"):  # judged scores (judge_hard.py) replace the matcher's
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
def arrays(B, R, kinds):
    keys = sorted(k for k in B if k in R and k[0] in kinds); ok = lambda r: r["score"] >= 1 - 1e-9
    a = lambda M, f: np.array([sum(f(r) for r in M[k]) for k in keys], float)
    bf, rf, bs, rs = a(B, lambda r: r["finished"]), a(R, lambda r: r["finished"]), a(B, ok), a(R, ok); nb, nr = a(B, lambda r: 1), a(R, lambda r: 1)
    return dict(n=len(keys), bf=bf, rf=rf, bs=bs, rs=rs, nb=nb, nr=nr, never=(bf == nb) & (bs == 0), never_rl=(rf == nr) & (rs == 0))
def cell(c, w):
    n = w.sum(); C = ~c["never"]; W = lambda x, m=None: float((w * x * (1 if m is None else m)).sum())
    reach = W(c["rs"], c["never"]) / (8 * n); mirror = W(c["bs"], c["never_rl"]) / (8 * n)
    rob = W(c["rs"] >= 4, c["never"]) / n; mrob = W(c["bs"] >= 4, c["never_rl"]) / n
    fb = W(c["bf"], C) / max(W(c["nb"], C), 1e-12); fr = W(c["rf"], C) / max(W(c["nr"], C), 1e-12)
    ab = W(c["bs"], C) / max(W(c["bf"], C), 1e-12); ar = W(c["rs"], C) / max(W(c["rf"], C), 1e-12); wc = W(np.ones_like(w), C) / n
    # matched: pass@1(B|fin) on the problems both models finish at least seven times in eight, where selection into the finished set cannot act
    M = (c["bf"] >= 7) & (c["rf"] >= 7); mab = W(c["bs"], M) / max(W(c["bf"], M), 1e-12); mar = W(c["rs"], M) / max(W(c["rf"], M), 1e-12)
    return np.array([(fr - fb) * ar * wc, fb * (ar - ab) * wc + mirror, reach - mirror, (W(c["rs"]) - W(c["bs"])) / (8 * n), rob - mrob,
                     W(c["bf"]) / W(c["nb"]), W(c["rf"]) / W(c["nr"]), W(c["bs"]) / max(W(c["bf"]), 1e-12), W(c["rs"]) / max(W(c["rf"]), 1e-12),
                     mab, mar, mar - mab] + matched_split(c, W, C, fb, ar, ab, wc, fr, mirror))
def matched_split(c, W, C, fb, ar, ab, wc, fr, mirror):
    # the split of Table 1's matched row, per pair: reliability taken only on problems outside U that both models finish at least
    # seven times in eight, finishing the remainder (so F + P is unchanged and selection into the finished set cannot enter P)
    Mm = C & (c["bf"] >= 7) & (c["rf"] >= 7); F = (fr - fb) * ar * wc; P = fb * (ar - ab) * wc
    Pm = fb * (W(c["rs"], Mm) / W(c["rf"], Mm) - W(c["bs"], Mm) / W(c["bf"], Mm)) * wc if W(c["bf"], Mm) > 0 and W(c["rf"], Mm) > 0 else P
    return [F + P - Pm, Pm + mirror]
NAMES = ["F", "P", "reach", "d", "robust", "fin_b", "fin_r", "acc_b", "acc_r", "m_acc_b", "m_acc_r", "m_delta", "F_m", "P_m"]
def stats(c, rng):
    n = c["n"]; pt = cell(c, np.ones(n)); boots = np.empty((NB, len(pt)))
    for b in range(NB): boots[b] = cell(c, np.bincount(rng.integers(0, n, n), minlength=n).astype(float))
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    out = {k: [float(pt[i]), float(lo[i]), float(hi[i])] for i, k in enumerate(NAMES)}
    out.update(n=n, U=int(c["never"].sum()), rl_on_U=int(((c["rs"] > 0) & c["never"]).sum()), rl_robust_on_U=int(((c["rs"] >= 4) & c["never"]).sum()),
               Up=int(c["never_rl"].sum()), base_on_Up=int(((c["bs"] > 0) & c["never_rl"]).sum()), base_robust_on_Up=int(((c["bs"] >= 4) & c["never_rl"]).sum()),
               n_matched=int(((c["bf"] >= 7) & (c["rf"] >= 7)).sum()))
    return out
rng = np.random.default_rng(0); res = {}
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    common = sorted(set(caps_for(rl)) & set(caps_for(base)))
    if not common: continue
    if os.environ.get("JUDGED"):  # a judged run uses judged scores on both sides or skips the pair; never one judged side against a matcher side
        missing = [m for m in (rl, base) if not os.path.exists(os.path.join(os.environ["JUDGED"], m + ".jsonl"))]
        if missing: print(f"{name:14s} skipped: no judged rows for {' and '.join(missing)}"); continue
    at = min((c for c in common if c >= cap), default=max(common)); budgets = sorted({at, 16384} & set(common))
    res[name] = {"base": tag, "train_cap": cap, "at_cap": at, "cells": {}}
    for bgt in budgets:
        B, R = load(base, bgt), load(rl, bgt)
        # cells per kind of problem and for all kinds together: hard and refill on the physics set; KINDS=pre,post on the math set
        for kinds, klab in [((k,), k) for k in KINDS] + [(tuple(KINDS), "all")]:
            c = arrays(B, R, kinds)
            if c["n"] == 0: continue
            s = stats(c, rng); res[name]["cells"][f"{bgt // 1024}k|{klab}"] = s
            print(f"{name:13s} {bgt // 1024:>3d}k {klab:6s} n={s['n']:3d} fin {s['fin_b'][0]:.2f}->{s['fin_r'][0]:.2f} acc|fin {s['acc_b'][0]:.3f}->{s['acc_r'][0]:.3f} "
                  f"F {s['F'][0]:+.3f} P {s['P'][0]:+.3f} reach {s['reach'][0]:+.3f} [{s['reach'][1]:+.3f},{s['reach'][2]:+.3f}] robust {s['robust'][0]:+.3f} "
                  f"| U {s['U']} RL>=1 {s['rl_on_U']} >=4 {s['rl_robust_on_U']} | U' {s['Up']} base>=1 {s['base_on_Up']} >=4 {s['base_robust_on_Up']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
