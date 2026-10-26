"""Which recipes push the boundary at 16k on physics, and why. Per recipe: the net exclusive share (RL-only minus base-only cells
at >=1 of 8), the change in finishing rate and mean length over the base, the residue (RL-only cells on which the base finished all
eight samples and was wrong every time) and how many of those the RL model solves in >=4 of 8. Correlations of the net share with
the finishing change, the length change and the final training cap. Then whether residue problems recur across the recipes on one
base more than independent draws from the base's all-finished-wrong problems would give.
usage: python boundary_dig.py out.json rows_root"""
import json, glob, sys, collections, statistics as st, random
OUT, ROOT = sys.argv[1], sys.argv[2]
BASES = {"DeepSeek-R1-Distill-Qwen-1.5B": ["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview", "JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
         "DeepSeek-R1-Distill-Qwen-7B": ["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B"], "Qwen3-4B": ["Polaris-4B-Preview"],
         "OpenMath-Nemotron-1.5B": ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"], "Qwen3-1.7B": ["e3-1.7B", "Polaris-1.7B-Preview"]}
SHORT = {"DeepScaleR-1.5B-Preview": "DeepScaleR", "Nemotron-Research-Reasoning-Qwen-1.5B": "ProRL v2", "Open-RS3": "Open-RS3", "STILL-3-1.5B-preview": "STILL-3", "JustRL-DeepSeek-1.5B": "JustRL-D", "DeepMath-1.5B": "DeepMath", "RLinf-math-1.5B": "RLinf", "AReaL-1.5B-Preview-Stage-3": "AReaL", "Skywork-OR1-7B": "Skywork-7B", "AceReason-Nemotron-7B": "AceReason", "Skywork-OR1-Math-7B": "Skywork-Math", "Polaris-4B-Preview": "Polaris-4B", "JustRL-Nemotron-1.5B": "JustRL-N", "QuestA-Nemotron-1.5B": "QuestA", "KnowRL-Nemotron-1.5B": "KnowRL", "DeepMath-Omn-1.5B": "DeepMath-Omn", "e3-1.7B": "e3", "Polaris-1.7B-Preview": "Polaris-1.7B"}
FINALCAP_K = {"DeepScaleR": 24, "ProRL v2": 8, "Open-RS3": 3.5, "STILL-3": 29, "JustRL-D": 15, "DeepMath": 24, "RLinf": 27.6, "AReaL": 24, "Skywork-7B": 32, "Skywork-Math": 32, "AceReason": 32, "Polaris-4B": 52, "JustRL-N": 15, "QuestA": 24, "KnowRL": 24, "DeepMath-Omn": 24, "e3": 16, "Polaris-1.7B": 48}   # Table tab:recipes, final stage
def load(m):
    fs = sorted(glob.glob(f"{ROOT}/**/{m}_cap16384.jsonl", recursive=True)); assert fs, m
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return by
def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys); return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / ((sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** .5)
solved = lambda rows: sum(r["score"] >= 1 - 1e-9 for r in rows)
rec, residue_by_base, pools = {}, collections.defaultdict(dict), {}
for base, recipes in BASES.items():
    B = load(base); keys_b = [k for k in B if k[0] == "phys"]
    pools[base] = [k for k in keys_b if all(r["finished"] for r in B[k]) and solved(B[k]) == 0]
    for m in recipes:
        R = load(m); n = SHORT[m]; keys = [k for k in keys_b if k in R]
        rl_only = [k for k in keys if solved(B[k]) == 0 and solved(R[k]) >= 1]; base_only = [k for k in keys if solved(R[k]) == 0 and solved(B[k]) >= 1]
        residue = {k[1]: solved(R[k]) for k in rl_only if all(r["finished"] for r in B[k])}
        residue_by_base[base][n] = residue
        rec[n] = {"net_share": (len(rl_only) - len(base_only)) / len(keys), "rl_only": len(rl_only), "base_only": len(base_only),
                  "d_finished": st.mean(r["finished"] for k in keys for r in R[k]) - st.mean(r["finished"] for k in keys for r in B[k]),
                  "d_len": st.mean(r["len"] for k in keys for r in R[k]) - st.mean(r["len"] for k in keys for r in B[k]),
                  "residue": len(residue), "residue_ge4": sum(v >= 4 for v in residue.values()), "residue_solves": residue, "final_cap_k": FINALCAP_K[n]}
ns = list(rec)
out = {"per_recipe": rec, "corr_net_vs_d_finished": corr([rec[n]["d_finished"] for n in ns], [rec[n]["net_share"] for n in ns]),
       "corr_net_vs_d_len": corr([rec[n]["d_len"] for n in ns], [rec[n]["net_share"] for n in ns]),
       "corr_net_vs_final_cap": corr([rec[n]["final_cap_k"] for n in ns], [rec[n]["net_share"] for n in ns]), "recurrence": {}}
rng = random.Random(0)
for base, d in residue_by_base.items():
    if len(d) < 3: continue
    cnt = collections.Counter(p for res in d.values() for p in res); shared = sum(1 for v in cnt.values() if v >= 2)
    sims = []
    for _ in range(2000):
        c = collections.Counter(p for res in d.values() for p in rng.sample(pools[base], len(res))); sims.append(sum(1 for v in c.values() if v >= 2))
    sims.sort()
    out["recurrence"][base] = {"recipes": len(d), "residue_cells": sum(len(r) for r in d.values()), "distinct_problems": len(cnt), "in_ge2_recipes": shared,
                               "independent_mean": st.mean(sims), "independent_95": [sims[50], sims[1950]], "pool_all_finished_wrong": len(pools[base]),
                               "most_shared": cnt.most_common(8)}
for k, v in out.items():
    if k != "per_recipe": print(k, json.dumps(v, default=str)[:400])
json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)
