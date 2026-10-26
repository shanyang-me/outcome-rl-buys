"""Reach split for the released pairs: does an RL model solve problems its base never samples a solution for?
For each pair, at a cap (default 16,384), per problem: base solves b of 8 samples (full score), RL solves r of 8.
Reports, per recipe and set and pooled: (i) of the problems RL solves at least once, the share the base solves at least once;
(ii) on base-0/8 problems, how many RL solves >=1 and >=4 times and RL's per-sample rate; (iii) the reverse (RL-0/8 problems,
base's rates) as the symmetric control; (iv) a crossed bootstrap (recipes x problems) for the pooled rate difference.
usage: python reach_split.py pairs_rows_root out.json rows.tex [cap]   (pairs_rows_root holds <run>/<model>_cap<cap>.jsonl)"""
import json, glob, os, sys, random, collections, statistics as st
ROOT, OUT, TEX = sys.argv[1], sys.argv[2], sys.argv[3]; CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
BASES = {"DeepSeek-R1-Distill-Qwen-1.5B": ["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview", "JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
         "DeepSeek-R1-Distill-Qwen-7B": ["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B"], "Qwen3-4B": ["Polaris-4B-Preview"],
         "OpenMath-Nemotron-1.5B": ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"], "Qwen3-1.7B": ["e3-1.7B", "Polaris-1.7B-Preview"]}
SHORT = {"Nemotron-Research-Reasoning-Qwen-1.5B": "ProRL v2", "AReaL-1.5B-Preview-Stage-3": "AReaL-1.5B", "STILL-3-1.5B-preview": "STILL-3", "DeepScaleR-1.5B-Preview": "DeepScaleR",
         "Polaris-4B-Preview": "Polaris-4B", "Polaris-1.7B-Preview": "Polaris-1.7B", "AceReason-Nemotron-7B": "AceReason-7B", "DeepMath-Omn-1.5B": "DeepMath-Omn", "RLinf-math-1.5B": "RLinf-math",
         "JustRL-Nemotron-1.5B": "JustRL-Nemotron", "QuestA-Nemotron-1.5B": "QuestA-Nemotron", "KnowRL-Nemotron-1.5B": "KnowRL-Nemotron"}
def load(model):
    fs = glob.glob(f"{ROOT}/*/{model}_cap{CAP}.jsonl") + glob.glob(f"{ROOT}/{model}_cap{CAP}.jsonl"); assert fs, model
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return {k: (sum(1 for r in v if r["score"] >= 1 - 1e-9), len(v)) for k, v in by.items()}
out = {"cap": CAP, "per_recipe": [], "pooled": {}}; cells = collections.defaultdict(list); tex = []
for base, recipes in BASES.items():
    B = load(base)
    for rec in recipes:
        R = load(rec)
        for kind in ("phys", "math"):
            keys = [k for k in B if k[0] == kind and k in R]
            b0 = [k for k in keys if B[k][0] == 0]; r0 = [k for k in keys if R[k][0] == 0]; rl1 = [k for k in keys if R[k][0] >= 1]
            row = {"recipe": rec, "base": base, "set": kind, "n": len(keys), "base0": len(b0), "rl_ge1_on_base0": sum(1 for k in b0 if R[k][0] >= 1), "rl_ge4_on_base0": sum(1 for k in b0 if R[k][0] >= 4),
                   "rl_rate_on_base0": (sum(R[k][0] for k in b0) / sum(R[k][1] for k in b0)) if b0 else None,
                   "rl0": len(r0), "base_ge1_on_rl0": sum(1 for k in r0 if B[k][0] >= 1), "base_ge4_on_rl0": sum(1 for k in r0 if B[k][0] >= 4),
                   "base_rate_on_rl0": (sum(B[k][0] for k in r0) / sum(B[k][1] for k in r0)) if r0 else None,
                   "rl_solved": len(rl1), "rl_solved_base_ge1": sum(1 for k in rl1 if B[k][0] >= 1)}
            row["share_rl_solved_base_ge1"] = row["rl_solved_base_ge1"] / len(rl1) if rl1 else None
            out["per_recipe"].append(row)
            for k in keys: cells[kind].append((rec, k, B[k], R[k]))
            if kind == "phys":
                tex.append(f"{SHORT.get(rec, rec)} & {len(b0)} & {row['rl_ge1_on_base0']} & {row['rl_ge4_on_base0']} & {row['rl_rate_on_base0']:.3f} & {len(r0)} & {row['base_ge1_on_rl0']} & {row['base_ge4_on_rl0']} & {row['base_rate_on_rl0']:.3f} & {row['share_rl_solved_base_ge1']:.2f} \\\\")
def pooled(cs):
    b0 = [(b, r) for _, _, b, r in cs if b[0] == 0]; r0 = [(b, r) for _, _, b, r in cs if r[0] == 0]; rl1 = [(b, r) for _, _, b, r in cs if r[0] >= 1]
    return {"base0": len(b0), "rl_ge1_on_base0": sum(1 for b, r in b0 if r[0] >= 1), "rl_ge4_on_base0": sum(1 for b, r in b0 if r[0] >= 4), "rl_rate_on_base0": sum(r[0] for b, r in b0) / sum(r[1] for b, r in b0),
            "rl0": len(r0), "base_ge1_on_rl0": sum(1 for b, r in r0 if b[0] >= 1), "base_ge4_on_rl0": sum(1 for b, r in r0 if b[0] >= 4), "base_rate_on_rl0": sum(b[0] for b, r in r0) / sum(b[1] for b, r in r0),
            "share_rl_solved_base_ge1": sum(1 for b, r in rl1 if b[0] >= 1) / len(rl1)}
rng = random.Random(0)
for kind, cs in cells.items():
    p = pooled(cs); recs = sorted({c[0] for c in cs}); probs = sorted({c[1] for c in cs}); byrp = {(c[0], c[1]): c for c in cs}
    diffs = []
    for _ in range(2000):   # crossed bootstrap: resample recipes and problems; statistic = RL rate on base-0 minus base rate on RL-0
        rs = [recs[rng.randrange(len(recs))] for _ in recs]; ps = [probs[rng.randrange(len(probs))] for _ in probs]
        sub = [byrp[(r, q)] for r in rs for q in ps if (r, q) in byrp]
        if sub:
            q = pooled(sub) if any(b[0] == 0 for _, _, b, _ in sub) and any(r[0] == 0 for _, _, _, r in sub) else None
            if q: diffs.append(q["rl_rate_on_base0"] - q["base_rate_on_rl0"])
    diffs.sort(); p["rate_diff"] = p["rl_rate_on_base0"] - p["base_rate_on_rl0"]; p["rate_diff_ci_crossed"] = [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]]
    out["pooled"][kind] = p
    print(kind, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}))
json.dump(out, open(OUT, "w"), indent=1); open(TEX, "w").write("\n".join(tex) + "\n"); print("wrote", OUT, TEX)
