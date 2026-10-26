"""Rich-get-richer test on the released pairs: bin problems by the base's full solves out of 8 at a cap and report the RL model's
change in per-sample solve rate per bin; the reverse direction (bin by the RL model's solves, base's change) is the
regression-to-the-mean control, so 'net' = forward - reverse. Prediction from group-relative credit with variance p(1-p): a hump
in the middle, nothing at either end. usage: python hump_split.py pairs_rows_root out.json rows.tex [cap]"""
import json, glob, sys, collections, statistics as st
ROOT, OUT, TEX = sys.argv[1], sys.argv[2], sys.argv[3]; CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
BASES = {"DeepSeek-R1-Distill-Qwen-1.5B": ["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview", "JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
         "DeepSeek-R1-Distill-Qwen-7B": ["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B"], "Qwen3-4B": ["Polaris-4B-Preview"],
         "OpenMath-Nemotron-1.5B": ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"], "Qwen3-1.7B": ["e3-1.7B", "Polaris-1.7B-Preview"]}
BINS = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 7), (8, 8)]
def load(model):
    fs = glob.glob(f"{ROOT}/*/{model}_cap{CAP}.jsonl") + glob.glob(f"{ROOT}/{model}_cap{CAP}.jsonl"); by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return {k: (sum(1 for r in v if r["score"] >= 1 - 1e-9), len(v)) for k, v in by.items()}
def binof(c): return next(i for i, (lo, hi) in enumerate(BINS) if lo <= c <= hi)
out = {"cap": CAP, "bins": [list(b) for b in BINS], "sets": {}}; tex = []
for kind in ("phys", "math"):
    fwd = collections.defaultdict(list); rev = collections.defaultdict(list)
    for base, recipes in BASES.items():
        B = load(base)
        for rec in recipes:
            R = load(rec)
            for k in B:
                if k not in R or k[0] != kind: continue
                b, n = B[k]; r, _ = R[k]; fwd[binof(b)].append((r - b) / n); rev[binof(r)].append((b - r) / n)
    rows = []
    for i, (lo, hi) in enumerate(BINS):
        f, g = fwd[i], rev[i]
        rows.append({"bin": [lo, hi], "n_fwd": len(f), "rl_minus_base": st.mean(f), "n_rev": len(g), "base_minus_rl": st.mean(g), "net": st.mean(f) - st.mean(g)})
        if kind == "phys":
            lab = f"{lo}" if lo == hi else f"{lo}--{hi}"
            tex.append(f"{lab} & {len(f)} & ${st.mean(f):+.3f}$ & {len(g)} & ${st.mean(g):+.3f}$ & ${st.mean(f) - st.mean(g):+.3f}$ \\\\")
    out["sets"][kind] = rows
    print(kind, [(r["bin"], round(r["net"], 3)) for r in rows])
json.dump(out, open(OUT, "w"), indent=1); open(TEX, "w").write("\n".join(tex) + "\n")
