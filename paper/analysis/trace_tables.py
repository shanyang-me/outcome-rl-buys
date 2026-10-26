"""LaTeX tables for the trace appendix from trace_out/trace_summary.json and trace_out/reading_<model>.json.
usage: python trace_tables.py trace_out out.tex"""
import json, sys, os
O, OUT = sys.argv[1], sys.argv[2]
S = json.load(open(f"{O}/trace_summary.json"))
ORDER = [("DeepSeek-R1-Distill-Qwen-1.5B", "R1-Distill-1.5B (base)"), ("DeepScaleR-1.5B-Preview", "DeepScaleR"), ("Nemotron-Research-Reasoning-Qwen-1.5B", "ProRL v2"),
         ("Open-RS3", "Open-RS3"), ("STILL-3-1.5B-preview", "STILL-3"), ("JustRL-DeepSeek-1.5B", "JustRL-DeepSeek"), ("DeepMath-1.5B", "DeepMath-1.5B"),
         ("RLinf-math-1.5B", "RLinf-math"), ("AReaL-1.5B-Preview-Stage-3", "AReaL-1.5B"), ("DeepSeek-R1-Distill-Qwen-7B", "R1-Distill-7B (base)"), ("Skywork-OR1-7B", "Skywork-OR1-7B"),
         ("Skywork-OR1-Math-7B", "Skywork-OR1-Math-7B"), ("AceReason-Nemotron-7B", "AceReason-Nemotron-7B"), ("Qwen3-4B", "Qwen3-4B (base)"), ("Polaris-4B-Preview", "Polaris-4B"),
         ("OpenMath-Nemotron-1.5B", "OpenMath-Nemotron-1.5B (base)"), ("JustRL-Nemotron-1.5B", "JustRL-Nemotron"), ("QuestA-Nemotron-1.5B", "QuestA-Nemotron"),
         ("KnowRL-Nemotron-1.5B", "KnowRL-Nemotron"), ("DeepMath-Omn-1.5B", "DeepMath-Omn"), ("Qwen3-1.7B", "Qwen3-1.7B (base)"), ("e3-1.7B", "e3-1.7B"),
         ("Polaris-1.7B-Preview", "Polaris-1.7B")]
f = lambda x, p=2: "--" if x is None else ("%.*f" % (p, x))
rows = []
for m, name in ORDER:
    if m not in S: continue
    s = S[m]; isbase = "base" not in s
    matched = "--" if isbase or "matched_acc_base" not in s else "$%.3f\\to%.3f$ (%d)" % (s["matched_acc_base"], s["matched_acc_rl"], s["matched_n"])
    hes_m = "--" if isbase or "matched_hes_base" not in s else "$%.0f\\to%.0f$" % (s["matched_hes_base"], s["matched_hes_rl"])
    lenr = "--" if isbase or "matched_len_ratio" not in s else "$%.2f\\times$" % s["matched_len_ratio"]
    run = "--" if s.get("unfinished_runaway_share") is None else "%.0f\\%% of %d" % (100 * s["unfinished_runaway_share"], s["n_unfinished"])
    if isbase and rows: rows.append("\\midrule")
    rows.append(f"{name} & ${s['finished']:.2f}$ & ${s['mean_score']:.3f}$ & ${s['tokens_finished']:,.0f}$ & ${s['hesitation_per_finished']:.0f}$ & ${s['hesitation_per_1k_tokens']:.1f}$ & ${s['verification_per_finished']:.1f}$ & {run} & {lenr} & {hes_m} & {matched} \\\\".replace(",", "{,}"))
t1 = ["\\begin{tabular}{lcccccccccc}", "\\toprule",
      "model & finished & score & tokens & \\multicolumn{2}{c}{hesitation} & checks & unfinished & \\multicolumn{3}{c}{problems both finish (vs base)} \\\\",
      " & & & per finished & per finished & per 1k & per finished & that loop & length & hesitation & accuracy \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
rr = []
for m, name in ORDER:
    p = f"{O}/reading_{m}.json"
    if not os.path.exists(p): continue
    R = json.load(open(p)); s = S.get(m, {})
    et = R["error_types"]; ns = R["rl_new_step"].get("true", 0); sa = R["same_approach"].get("true", 0); lh = R["less_hesitation"].get("true", 0); lp = R["luck_plausible"].get("true", 0)
    RP = json.load(open(f"{O}/repertoire_check.json"))["by_model"].get(m, {"claimed": 0, "in_base": 0})
    rr.append(f"{name} & {s.get('discordant_rl_right_base_wrong', '--')} / {s.get('discordant_base_right_rl_wrong', '--')} & {R['n']} & {et.get('setup', 0)} & {et.get('execution', 0)} & {et.get('abandonment', 0)} & {et.get('gold_or_grader_issue', 0) + et.get('answer_form', 0) + et.get('incomplete', 0) + et.get('unclear', 0)} & {ns} & {RP['in_base']} & {sa} & {lh} & {lp} & {R['agreement_rate']:.2f} \\\\")
t2 = ["\\begin{tabular}{lcccccccccccc}", "\\toprule",
      " & RL right, base wrong / & read & \\multicolumn{4}{c}{where the read base sample fails} & step absent from & \\dots\\ but in another & same & RL hesitates & base could & reader \\\\",
      "RL model & base right, RL wrong & & setup & execution & abandons & other & that sample & base sample & approach & less & succeed & agreement \\\\", "\\midrule"] + rr + ["\\bottomrule", "\\end{tabular}"]
open(OUT, "w").write("% TRACE_TABLE_STATS\n" + "\n".join(t1) + "\n% TRACE_TABLE_READING\n" + "\n".join(t2) + "\n")
print(len(rows), "stat rows;", len(rr), "reading rows")
