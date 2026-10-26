"""Per-recipe four-way solve overlap (both / base only / RL only / neither, at least one solve in eight) at each cap on physics,
plus a budget read of the exclusive cells: the base's finishing rate on the RL-only problems against its overall rate, how many
RL-only cells have at least one truncated base sample, and the mirror for the base-only cells.
usage: python overlap_per_recipe.py out.json rows_root [rows_root ...]   (searched recursively for <model>_cap<cap>.jsonl)"""
import json, glob, sys, collections
OUT, ROOTS = sys.argv[1], sys.argv[2:]
BASES = {"DeepSeek-R1-Distill-Qwen-1.5B": ["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview", "JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
         "DeepSeek-R1-Distill-Qwen-7B": ["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B"], "Qwen3-4B": ["Polaris-4B-Preview"],
         "OpenMath-Nemotron-1.5B": ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"], "Qwen3-1.7B": ["e3-1.7B", "Polaris-1.7B-Preview"]}
SHORT = {"DeepScaleR-1.5B-Preview": "DeepScaleR", "Nemotron-Research-Reasoning-Qwen-1.5B": "ProRL v2", "Open-RS3": "Open-RS3", "STILL-3-1.5B-preview": "STILL-3", "JustRL-DeepSeek-1.5B": "JustRL-D", "DeepMath-1.5B": "DeepMath", "RLinf-math-1.5B": "RLinf", "AReaL-1.5B-Preview-Stage-3": "AReaL", "Skywork-OR1-7B": "Skywork-7B", "AceReason-Nemotron-7B": "AceReason", "Skywork-OR1-Math-7B": "Skywork-Math", "Polaris-4B-Preview": "Polaris-4B", "JustRL-Nemotron-1.5B": "JustRL-N", "QuestA-Nemotron-1.5B": "QuestA", "KnowRL-Nemotron-1.5B": "KnowRL", "DeepMath-Omn-1.5B": "DeepMath-Omn", "e3-1.7B": "e3", "Polaris-1.7B-Preview": "Polaris-1.7B"}
def load(model, cap):
    fs = sorted(f for d in ROOTS for f in glob.glob(f"{d}/**/{model}_cap{cap}.jsonl", recursive=True)); assert fs, (model, cap)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return by
solved = lambda rows: sum(1 for r in rows if r["score"] >= 1 - 1e-9) >= 1
fin = lambda M, ks: sum(r["finished"] for k in ks for r in M[k]) / max(1, sum(len(M[k]) for k in ks))
out = {}
for cap in (4096, 8192, 16384):
    out[cap] = {}
    for base, recipes in BASES.items():
        B = load(base, cap)
        for rec in recipes:
            R = load(rec, cap); keys = [k for k in B if k[0] == "phys" and k in R]
            c = collections.Counter(); rl_only = []; base_only = []
            for k in keys:
                b, r = solved(B[k]), solved(R[k])
                c["both" if b and r else "base_only" if b else "rl_only" if r else "neither"] += 1
                if r and not b: rl_only.append(k)
                if b and not r: base_only.append(k)
            n = len(keys)
            out[cap][SHORT[rec]] = {**{x: c[x] / n for x in ("both", "base_only", "rl_only", "neither")}, "n": n,
                "base_finished_on_rl_only": fin(B, rl_only), "base_finished_overall": fin(B, keys),
                "rl_only_base_all_finished": sum(1 for k in rl_only if all(r["finished"] for r in B[k])),
                "rl_only_base_some_truncated": sum(1 for k in rl_only if any(not r["finished"] for r in B[k])),
                "rl_finished_on_base_only": fin(R, base_only), "rl_finished_overall": fin(R, keys),
                "base_only_rl_all_finished": sum(1 for k in base_only if all(r["finished"] for r in R[k])),
                "base_only_rl_some_truncated": sum(1 for k in base_only if any(not r["finished"] for r in R[k]))}
    print(cap, {k: (round(v["rl_only"], 3), round(v["base_only"], 3)) for k, v in out[cap].items()})
json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)
