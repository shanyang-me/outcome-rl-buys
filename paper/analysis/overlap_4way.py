"""Four-way solve overlap for the released pairs: for each recipe x problem cell at a cap, does the base solve >=1 of 8
samples, does the RL model, both, or neither. Pooled per set and cap over the 18 pairs.
usage: python overlap_4way.py out.json rows_dir [rows_dir ...]   (each rows_dir holds <model>_cap<cap>.jsonl)"""
import json, glob, sys, collections
OUT, ROOTS = sys.argv[1], sys.argv[2:]
BASES = {"DeepSeek-R1-Distill-Qwen-1.5B": ["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview", "JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
         "DeepSeek-R1-Distill-Qwen-7B": ["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B"], "Qwen3-4B": ["Polaris-4B-Preview"],
         "OpenMath-Nemotron-1.5B": ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"], "Qwen3-1.7B": ["e3-1.7B", "Polaris-1.7B-Preview"]}
def load(model, cap):
    fs = [f for d in ROOTS for f in glob.glob(f"{d}/{model}_cap{cap}.jsonl")]
    if not fs: return None
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return {k: (sum(1 for r in v if r["score"] >= 1 - 1e-9), len(v)) for k, v in by.items()}
res = {}
for cap in (4096, 8192, 16384):
    agg = {k: collections.Counter() for k in ("phys", "math")}; nrec = 0
    for base, recipes in BASES.items():
        B = load(base, cap)
        if B is None: continue
        for rec in recipes:
            R = load(rec, cap)
            if R is None: continue
            nrec += 1
            for kind in ("phys", "math"):
                for k in B:
                    if k[0] != kind or k not in R: continue
                    b = B[k][0] >= 1; r = R[k][0] >= 1
                    agg[kind]["both" if b and r else "base_only" if b else "rl_only" if r else "neither"] += 1
    res[cap] = {kind: {**{c: agg[kind][c] for c in ("both", "base_only", "rl_only", "neither")}, "n": sum(agg[kind].values()), "recipes": nrec} for kind in agg}
    for kind, c in agg.items():
        n = sum(c.values()) or 1
        print(cap, kind, n, {k: round(c[k] / n, 3) for k in ("both", "base_only", "rl_only", "neither")}, "of RL-solved, base also:", round(c["both"] / max(1, c["both"] + c["rl_only"]), 3), "of base-solved, RL also:", round(c["both"] / max(1, c["both"] + c["base_only"]), 3))
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
