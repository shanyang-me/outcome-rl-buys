"""Full credit versus partial credit (Oct 19). Table 4's capped score is the mean outcome reward (fraction of gold parts matched);
Figures 1-2 count a sample right only at full credit. For every recipe at 16k and at its cap (physics), the change under each.
usage: python credit_check.py cap_split.json rows_root [rows_root ...]"""
import json, sys, os, glob, collections
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); CAPJ, ROOTS = sys.argv[1], sys.argv[2:]
os.environ["KINDS"] = "phys"
src = open(os.path.join(H, "fig1_ci.py")).read(); exec(src[src.index("KINDS ="):src.index("def arrays")])
cs = json.load(open(CAPJ))["recipes"]; R64 = [r for r in ROOTS if "cap64" in r or "pair_decomp5" in r or late(r)]
full = lambda r: float(r["score"] >= 1 - 1e-9); part = lambda r: float(r["score"])
rng = np.random.default_rng(0)
def change(B, R, f, boot=0):
    ks = sorted(k for k in B if k in R); d = np.array([sum(f(r) for r in R[k]) - sum(f(r) for r in B[k]) for k in ks]) / 8
    if not boot: return d.mean()
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(boot)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
worst = 0
for rec, c in cs.items():
    rl, base = PAIRS[rec][0], PAIRS[rec][1]; out = []
    for lab, cap, roots in (("16k", 16384, ROOTS), (c["at_label"], c["at_cap"], R64)):
        B, R = load(base, cap, roots), load(rl, cap, roots)
        df, dp = change(B, R, full), change(B, R, part); worst = max(worst, abs(df - dp))
        out.append(f"{lab:>4s} full {df:+.3f} part {dp:+.3f}")
        if rec in ("QuestA", "Polaris-1.7B", "Polaris-4B"):
            m, lo, hi = change(B, R, full, 4000); m2, lo2, hi2 = change(B, R, part, 4000)
            out.append(f"(CI full [{lo:+.3f},{hi:+.3f}] part [{lo2:+.3f},{hi2:+.3f}])")
    print(f"{rec:14s}", " | ".join(out))
print("largest |full - partial| difference in a change:", round(worst, 3))
