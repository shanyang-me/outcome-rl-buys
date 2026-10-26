"""Appendix tables for the competition-math read (hard_reach.py JSON with KINDS=pre,post; hard_by_source.py JSON; p5_harness.py JSON).
(1) cells.tex (tab:mathcells): every recipe at its cap on the 276 problems: finishing, accuracy per finished derivation, the
three-part split, robust reach and the problem counts behind reach. (2) years.tex (tab:mathyears): per recipe at its cap, the
2025 contests (213 problems) against the 2026 contests (63, after every model's release): finishing, the change in pass@1 and net
reach on each, and the difference in net reach (2026 minus 2025) with a bootstrap interval from math_p4.py's JSON when given.
(3) harness.tex (tab:mathharness): our AIME 2025 pass@1 at each recipe's cap against the number its authors report.
usage: [P4=math_p4.json] python math_table.py math_reach.json p5.json cells.tex years.tex harness.tex"""
import json, sys, os
HR, P5, OUT1, OUT2, OUT3 = sys.argv[1:6]
hr, p5 = json.load(open(HR)), json.load(open(P5))
p4 = json.load(open(os.environ["P4"])) if os.environ.get("P4") else {}
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def f3(x): return "0" if abs(x) < 5e-4 else ("$-$" if x < 0 else "") + f"{abs(x):.3f}"
def ci(v): return f"{f3(v[0])} [{f3(v[1])}, {f3(v[2])}]"
order = [n for n in PAIRS if n in hr]
scorer = "judge" if any("judged" in str(v.get("scorer", "")) for v in hr.values()) or os.environ.get("SCORER") == "judge" else "matcher"
rows1, rows2, rows3 = [], [], []
for name in order:
    r = hr[name]; at = r["at_cap"]; L = f"{at // 1024}k"; c = r["cells"].get(f"{L}|all")
    if c is None: continue
    matched = f"{ci(c['m_delta'])} ({c['n_matched']})" if "m_delta" in c else "--"
    pm = ci(c["P_m"]) if "P_m" in c else "--"
    rows1.append(f"{name} & {L} & ${c['fin_b'][0]:.2f}\\to{c['fin_r'][0]:.2f}$ & ${c['acc_b'][0]:.2f}\\to{c['acc_r'][0]:.2f}$ & {ci(c['F'])} & {ci(c['P'])} & {pm} & {matched} & {ci(c['reach'])} & {f3(c['robust'][0])} & "
                 f"{c['U']} / {c['rl_on_U']} ({c['rl_robust_on_U']}) & {c['Up']} / {c['base_on_Up']} ({c['base_robust_on_Up']}) \\\\")
    pre, post = r["cells"].get(f"{L}|pre"), r["cells"].get(f"{L}|post")
    if pre and post:
        diff = p4.get(name, {}).get("reach_diff")
        rows2.append(f"{name} & {L} & ${pre['fin_b'][0]:.2f}\\to{pre['fin_r'][0]:.2f}$ & {ci(pre['d'])} & {ci(pre['reach'])} & ${post['fin_b'][0]:.2f}\\to{post['fin_r'][0]:.2f}$ & {ci(post['d'])} & {ci(post['reach'])} & "
                     + (ci(diff) if diff else "--") + " \\\\")
    rl = PAIRS[name][0].split("/")[-1]; q = p5.get("rl", {}).get(rl)
    if q and q.get("reported") is not None:
        a = q["authors"]; auth = f"{a.get('k') or '--'} / {(str(a['max_tokens'] // 1024) + 'k') if a.get('max_tokens') else '--'} / {a.get('temperature') if a.get('temperature') is not None else '--'}"
        rows3.append(f"{name} & {L} & {q['fin_at_cap']:.2f} & {q['pass1_at_cap']:.3f} & {q['reported'] / 100:.3f} & {f3(q['gap_at_cap'])} & {auth} & {'yes' if q['comparable'] else 'no'} \\\\")
sc = ("the fixed judge of Appendix~\\ref{sec:judge-v2} after the exact-match checker" if scorer == "judge" else "the frozen matcher in its exact mode")
with open(OUT1, "w") as h:
    h.write("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llcccccccccc}\n\\toprule\n")
    h.write("recipe & cap & fin@1$(B)$ & pass@1$(B\\,|\\,\\text{fin})$ & finishing $F$ & reliability $P$ & matched $P$ & matched $\\Delta$ ($n$) & reach & robust & $|U|$ / RL $\\geq1$ ($\\geq4$) & $|U'|$ / base $\\geq1$ ($\\geq4$) \\\\\n")
    h.write(" & & base$\\to$RL & base$\\to$RL & & & & pass@1$(B\\,|\\,\\text{fin})$ & net of mirror & net & & \\\\\n\\midrule\n" + "\n".join(rows1) + "\n\\bottomrule\n\\end{tabular}}\n")
    h.write("\\caption{The competition-math read at each recipe's cap: $276$ final-answer problems from eight 2025 and 2026 contests, eight samples per problem, scored by " + sc
            + ". Matched $\\Delta$: the change in pass@1 per finished derivation on the $n$ problems both models finish at least seven times in eight, where selection into the finished set cannot act. Matched $P$: reliability from that change, weighted as $P$ is ($F$ takes the rest), the per-pair version of Table~\\ref{tab:splitsens}'s matched split. "
            "$U$: problems the base finishes eight times and never solves; RL $\\geq1$ ($\\geq4$): how many of them the RL model solves at least once (at least four times); $U'$ and base likewise. Robust reach counts wins in at least four of eight, net of the mirror. Intervals: paired bootstrap over problems.}\n\\label{tab:mathcells}\n\\end{table*}\n")
with open(OUT2, "w") as h:
    h.write("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llcccccccc}\n\\toprule\n")
    h.write(" & & \\multicolumn{3}{c}{2025 contests ($213$)} & \\multicolumn{3}{c}{2026 contests ($63$)} & \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n")
    h.write("recipe & cap & fin@1 & $\\Delta$ & reach & fin@1 & $\\Delta$ & reach & reach, 2026 $-$ 2025 \\\\\n\\midrule\n" + "\n".join(rows2) + "\n\\bottomrule\n\\end{tabular}}\n")
    h.write("\\caption{Contamination check on the competition-math read: each recipe at its cap on the $213$ problems from 2025 contests and on the $63$ from AIME 2026 and HMMT February 2026, which postdate every model's release. fin@1: base$\\to$RL; $\\Delta$: the change in pass@1; reach net of mirror; the last column is the difference in net reach between the two strata, with a bootstrap interval over problems resampled within each stratum.}\n\\label{tab:mathyears}\n\\end{table*}\n")
with open(OUT3, "w") as h:
    h.write("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llcccccc}\n\\toprule\n")
    h.write("recipe & cap & fin@1 & ours & authors & gap & authors' $k$ / cap / $T$ & comparable \\\\\n\\midrule\n" + "\n".join(rows3) + "\n\\bottomrule\n\\end{tabular}}\n")
    h.write("\\caption{Harness check: our AIME 2025 pass@1 for each RL model at its recipe's cap (thirty problems, eight samples, the same scoring as Table~\\ref{tab:mathcells}) against the number its authors report, with the authors' sample count, token cap and temperature. A row is comparable when the authors sample at our temperature and top-p and at a budget within one step of the recipe's cap, so a gap there is a harness difference rather than a budget or temperature one.}\n\\label{tab:mathharness}\n\\end{table}\n")
print(f"{len(rows1)} recipes -> {OUT1}; {len(rows2)} year rows -> {OUT2}; {len(rows3)} harness rows -> {OUT3}")
