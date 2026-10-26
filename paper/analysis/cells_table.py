"""Every cell behind the cap rows of Figure 1 and all of Figure 2 (Oct 19). For each recipe at its training cap and one budget
beyond, on physics and on MATH-500 where read: fin@1(B) and pass@1(B|fin) for base -> RL over all problems, and the three parts
of the full-credit change with paired bootstrap intervals over problems: F, P (with the mirror added back) and reach net of its
mirror; d is their sum. Same cells and loader as fig1_ci.py. usage: python cells_table.py out.tex out.json cap_split.json rows_root..."""
import json, glob, sys, collections, os
import numpy as np
OUT, OUTJ, CAPJ, ROOTS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
NB = int(os.environ.get("NBOOT", "20000")); KINDS = ("phys", "math")
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
    if not keys: return None
    a = lambda M, f: np.array([sum(f(r) for r in M[k]) for k in keys], float); ok = lambda r: r["score"] >= 1 - 1e-9
    bf, rf, bs, rs = a(B, lambda r: r["finished"]), a(R, lambda r: r["finished"]), a(B, ok), a(R, ok); nb, nr = a(B, lambda r: 1), a(R, lambda r: 1)
    return dict(n=len(keys), bf=bf, rf=rf, bs=bs, rs=rs, nb=nb, nr=nr, never=(bf == nb) & (bs == 0), never_rl=(rf == nr) & (rs == 0))
def cell(c, w):
    n = w.sum(); C = ~c["never"]; W = lambda x, m=None: float((w * x * (1 if m is None else m)).sum())
    reach = W(c["rs"], c["never"]) / (8 * n); mirror = W(c["bs"], c["never_rl"]) / (8 * n)
    fb = W(c["bf"], C) / W(c["nb"], C); fr = W(c["rf"], C) / W(c["nr"], C)
    ab = W(c["bs"], C) / max(W(c["bf"], C), 1e-12); ar = W(c["rs"], C) / max(W(c["rf"], C), 1e-12); wc = W(np.ones_like(w), C) / n
    return np.array([(fr - fb) * ar * wc, fb * (ar - ab) * wc + mirror, reach - mirror, (W(c["rs"]) - W(c["bs"])) / (8 * n)])
def stats(c, rng):
    n = c["n"]; pt = cell(c, np.ones(n)); boots = np.empty((NB, 4))
    for b in range(NB): boots[b] = cell(c, np.bincount(rng.integers(0, n, n), minlength=n).astype(float))
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    full = lambda M, f: float(sum(f(r) for k in M for r in M[k]))
    return dict(n=n, fin_b=float(c["bf"].sum() / c["nb"].sum()), fin_r=float(c["rf"].sum() / c["nr"].sum()),
                acc_b=float(c["bs"].sum() / max(c["bf"].sum(), 1e-12)), acc_r=float(c["rs"].sum() / max(c["rf"].sum(), 1e-12)),
                F=[pt[0], lo[0], hi[0]], P=[pt[1], lo[1], hi[1]], reach=[pt[2], lo[2], hi[2]], d=[pt[3], lo[3], hi[3]], U=int(c["never"].sum()), Up=int(c["never_rl"].sum()))
cs = json.load(open(CAPJ))["recipes"]; rng = np.random.default_rng(0); res = {}
order = sorted(cs, key=lambda n: cs[n]["at"]["fb"])  # as Figure 2: by how often the base finishes at the cap, lowest first
for name in order:
    c = cs[name]; rl, base = PAIRS[name][0], PAIRS[name][1]; res[name] = {"base": PAIRS[name][3], "cells": []}
    for which in ("at", "beyond"):
        cap = c.get(f"{which}_cap")
        if not cap: continue
        B, R = load(base, cap), load(rl, cap)
        for kind in KINDS:
            a = arrays(B, R, kind)
            if a is None: continue
            s = stats(a, rng); s.update(which=which, kind=kind, cap=cap, label=c[f"{which}_label"] + (" (below cap)" if c.get("below_only") and which == "at" else ""))
            res[name]["cells"].append(s); print(name, kind, s["label"], f"F {s['F'][0]:+.3f} P {s['P'][0]:+.3f} [{s['P'][1]:+.3f},{s['P'][2]:+.3f}] reach {s['reach'][0]:+.3f} d {s['d'][0]:+.3f}", flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
def num(x, p=3): return ("$-$" if x < -0.5 * 10 ** -p else "") + ("%." + str(p) + "f") % abs(x) if abs(x) >= 0.5 * 10 ** -p else "0"
def ci(v): return "%s [%s, %s]" % (num(v[0]), num(v[1]), num(v[2]))
rows = []; SETS = {"phys": "physics", "math": "MATH-500"}
for name, r in res.items():
    first = True
    for s in r["cells"]:
        rows.append("%s & %s, %s & $%.2f\\to%.2f$ & $%.2f\\to%.2f$ & %s & %s & %s & %s \\\\" % (
            name.replace("Skywork-32B", "Skywork-OR1-32B").replace("AceReason-14B", "AceReason-Nemotron-14B") if first else "", SETS[s["kind"]], s["label"].replace(" (below cap)", "$^{b}$"),
            s["fin_b"], s["fin_r"], s["acc_b"], s["acc_r"], ci(s["F"]), ci(s["P"]), ci(s["reach"]), ci(s["d"])))
        first = False
    rows.append("\\addlinespace[1pt]")
HEAD = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llcccccc}", "\\toprule",
        "recipe & set, budget & fin@1$(B)$ & pass@1$(B\\,|\\,\\text{fin})$ & finishing $F$ & reliability $P$ & reach & change $\\Delta$ \\\\",
        " & & base$\\to$RL & base$\\to$RL & & & net of mirror & \\\\", "\\midrule"]
TAIL = ["\\bottomrule", "\\end{tabular}}",
        "\\caption{Every cell behind the cap rows of Figure~\\ref{fig:overview} and all of Figure~\\ref{fig:recipes}: each recipe at its training cap and one budget beyond, on physics and on MATH-500 where that set was read at the budget. "
        "\\finB{} and \\relB{} are over all problems, base$\\to$RL. The three parts split the change in full-credit pass@1 (so $\\Delta$ differs slightly from the capped score of Table~\\ref{tab:publicpairs}, which averages partial credit): the finishing term $F$, the reliability term $P$ with the mirror added back, and reach net of its mirror, with $95\\%$ paired bootstrap intervals over problems ($20{,}000$ resamples). "
        "Rows run as in Figure~\\ref{fig:recipes}, from the base that finishes least often at the cap.}",
        "\\label{tab:capcells}", "\\end{table*}"]
open(OUT, "w").write("\n".join(HEAD + rows + TAIL) + "\n"); print("rows", sum(1 for r in rows if "&" in r))
