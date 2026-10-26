"""Rows of the paper's main-text table of the 18 recipes (tab:pairsmain): matched finished-length ratio, mean finishing change at
4k/8k, binding-cap dominance pooled over sets [problem CI] and by set, matched accuracy change at 16k on physics and MATH-500.
usage: python exp2/pairs_main18.py stats_all.json stats_dom_phys.json stats_dom_math.json pairs_ci_all.json out_rows.tex  (from the scratchpad root)"""
import json, sys
sys.path.insert(0, ".")
from families import FAMILIES
sa, sp, sm = (json.load(open(f)) for f in sys.argv[1:4]); C = json.load(open(sys.argv[4]))
NAME = {"Nemotron-Research-Reasoning-Qwen-1.5B": "ProRL v2", "DeepScaleR-1.5B-Preview": "DeepScaleR", "STILL-3-1.5B-preview": "STILL-3",
        "JustRL-DeepSeek-1.5B": "JustRL-DeepSeek", "DeepMath-1.5B": "DeepMath-1.5B", "RLinf-math-1.5B": "RLinf-math", "AReaL-1.5B-Preview-Stage-3": "AReaL-1.5B",
        "Skywork-OR1-7B": "Skywork-OR1-7B", "Skywork-OR1-Math-7B": "Skywork-OR1-Math-7B", "AceReason-Nemotron-7B": "AceReason-7B", "Polaris-4B-Preview": "Polaris-4B",
        "JustRL-Nemotron-1.5B": "JustRL-Nemotron", "QuestA-Nemotron-1.5B": "QuestA-Nemotron", "KnowRL-Nemotron-1.5B": "KnowRL-Nemotron", "DeepMath-Omn-1.5B": "DeepMath-Omn",
        "e3-1.7B": "e3-1.7B", "Polaris-1.7B-Preview": "Polaris-1.7B", "Open-RS3": "Open-RS3"}
BASE = {"DeepSeek-R1-Distill-Qwen-1.5B": "R1-Distill-1.5B", "DeepSeek-R1-Distill-Qwen-7B": "R1-Distill-7B", "Qwen3-4B": "Qwen3-4B",
        "OpenMath-Nemotron-1.5B": "OpenMath-Nemotron-1.5B", "Qwen3-1.7B": "Qwen3-1.7B"}
lv = sa["length_vs_finishing_matched"]; dom = sa["dominance_by_capset"]["4k+8k"]; dph = sp["dominance_by_capset"]["4k+8k"]; dma = sm["dominance_by_capset"]["4k+8k"]
cell = lambda rl, kind, cap: next(c for c in C.values() if c["rl"] == rl and c["set"] == kind and c["cap"] == cap)
z = lambda x, p=3: ("%+.*f" % (p, x)).replace("+0.", "+.").replace("-0.", "-.")
changing = lambda r: any(c["rl"] == r and c["cap"] in (4096, 8192) and (c["capped_gain"] or c["capped_loss"]) for c in C.values())
rows = []
for b, recs in FAMILIES.items():
    rows.append("\\multicolumn{7}{l}{\\textit{from %s}} \\\\" % BASE[b])
    for r in recs:
        L = lv["x_median_matched_finished_len_ratio"][r]; dfin = lv["y_mean_dfinished_4k8k"][r]
        d = dom["per_recipe"][r]; ci = dom["per_recipe_ci_problems"][r]
        mp, mm = cell(r, "phys", 16384)["matched"], cell(r, "math", 16384)["matched"]
        fmt = lambda m: ("$%s$ [$%s$, $%s$]" % (z(m["delta"]), z(m["ci"][0]), z(m["ci"][1]))) if "delta" in m else "--"
        dstr = ("$%.2f$ [$%.2f$, $%.2f$]" % (d, ci[0], ci[1])) if changing(r) else "-- (no change)"
        sets = ("$%.2f$ / $%.2f$" % (dph["per_recipe"][r], dma["per_recipe"][r])) if changing(r) else "--"
        rows.append("\\quad %s & $%.2f\\times$ & $%s$ & %s & %s & %s & %s \\\\" % (NAME[r], L, z(dfin, 2), dstr, sets, fmt(mp), fmt(mm)))
open(sys.argv[5], "w").write("\n".join(rows) + "\n"); print(len(rows), "rows")
