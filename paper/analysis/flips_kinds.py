"""Figure 1's estimator per benchmark at 4k+8k, and each recipe's physics change at 16k against its change at its own cap (Oct 19).
usage: python flips_kinds.py cap_split.json rows_root [rows_root ...]"""
import json, sys, os, glob, collections
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); CAPJ, ROOTS = sys.argv[1], sys.argv[2:]
os.environ["KINDS"] = "phys,math"
src = open(os.path.join(H, "fig1_ci.py")).read(); exec(src[src.index("KINDS ="):src.index("cs = json.load")])
src = open(os.path.join(H, "overlap_per_recipe.py")).read(); exec(src[src.index("BASES = "):src.index("def load")])
for kind in ("phys", "math"):
    vals = []
    for base, recs in BASES.items():
        for cap in (4096, 8192):
            B = load(base, cap, ROOTS)
            for rec in recs:
                c = arrays(B, load(rec, cap, ROOTS), kind); vals.append(cell(c, np.ones(len(c["ids"]))))
    print(kind, "4k+8k size shares", [round(x, 3) for x in pool(vals)["size"]])
cs = json.load(open(CAPJ))["recipes"]; ok = lambda r: r["score"] >= 1 - 1e-9
for rec, c in cs.items():
    rl, base = PAIRS[rec][0], PAIRS[rec][1]
    B, R = load(base, 16384, ROOTS), load(rl, 16384, ROOTS); ks = [k for k in B if k in R and k[0] == "phys"]
    d16 = sum(ok(r) for k in ks for r in R[k]) / (8 * len(ks)) - sum(ok(r) for k in ks for r in B[k]) / (8 * len(ks))
    print(f"{rec:14s} 16k {d16:+.3f}  cap {c['at_label']} {c['at']['d']:+.3f}{'  FLIP' if d16 < 0 < c['at']['d'] else ''}")
