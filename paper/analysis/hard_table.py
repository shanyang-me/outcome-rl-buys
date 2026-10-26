"""Appendix tables for the harder, unfiltered set (hard_reach.py JSON). (1) cells.tex: every recipe at its cap on all 671 problems
(or on the problems both models were read on): finishing, accuracy per finished derivation, the three-part split, robust reach
(wins in at least four of eight) and the problem counts behind reach. (2) filter.tex: the filter's effect, each recipe at its cap on
the paper's 300 filtered problems (cap_cells.json), the 200 drawn from the same pools without the filter, and the 471 harder ones:
finishing, net reach and the change, with intervals. usage: python hard_table.py hard_reach.json cap_cells.json cells.tex filter.tex"""
import json, sys, os
HR, CC, OUT1, OUT2 = sys.argv[1:5]
hr, cc = json.load(open(HR)), json.load(open(CC))
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def f3(x): return ("$-$" if x < 0 else "") + f"{abs(x):.3f}".replace("0.000", "0")
def ci(v): return f"{f3(v[0])} [{f3(v[1])}, {f3(v[2])}]"
def pm(v): return f"{f3(v[0])} [{f3(v[1])}, {f3(v[2])}]"
order = [n for n in PAIRS if n in hr]
rows1, rows2 = [], []
for name in order:
    r = hr[name]; at = r["at_cap"]; key = f"{at // 1024}k|all"
    if key not in r["cells"]: continue
    c = r["cells"][key]
    rows1.append(f"{name} & {at // 1024}k & {c['n']} & ${c['fin_b'][0]:.2f}\\to{c['fin_r'][0]:.2f}$ & ${c['acc_b'][0]:.2f}\\to{c['acc_r'][0]:.2f}$ & {ci(c['F'])} & {ci(c['P'])} & {ci(c['reach'])} & {f3(c['robust'][0])} & "
                 f"{c['U']} / {c['rl_on_U']} ({c['rl_robust_on_U']}) & {c['Up']} / {c['base_on_Up']} ({c['base_robust_on_Up']}) \\\\")
    cells = {k: v for k, v in r["cells"].items() if k.startswith(f"{at // 1024}k|")}
    filt = next((x for x in cc.get(name, {}).get("cells", []) if x["which"] == "at" and x["kind"] == "phys"), None)
    parts = []
    for lab, x in (("filtered", filt), ("refill", cells.get(f"{at // 1024}k|refill")), ("hard", cells.get(f"{at // 1024}k|hard"))):
        if x is None: parts.append("-- & -- & --"); continue
        fb, fr = (x["fin_b"], x["fin_r"]) if lab == "filtered" else (x["fin_b"][0], x["fin_r"][0])
        parts.append(f"${fb:.2f}\\to{fr:.2f}$ & {ci(x['reach'])} & {ci(x['d'])}")
    rows2.append(f"{name} & {at // 1024}k & " + " & ".join(parts) + " \\\\")
with open(OUT1, "w") as h:
    h.write("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrcccccccc}\n\\toprule\n")
    h.write("recipe & cap & $n$ & fin@1$(B)$ & pass@1$(B\\,|\\,\\text{fin})$ & finishing $F$ & reliability $P$ & reach & robust & $|U|$ / RL $\\geq1$ ($\\geq4$) & $|U'|$ / base $\\geq1$ ($\\geq4$) \\\\\n")
    h.write(" & & & base$\\to$RL & base$\\to$RL & & & net of mirror & net & & \\\\\n\\midrule\n" + "\n".join(rows1) + "\n\\bottomrule\n\\end{tabular}}\n")
    h.write("\\caption{The harder, unfiltered set at each recipe's cap: $471$ harder problems and $200$ drawn from the paper's pools without the base filter, eight samples per problem, scored by " + ("the fixed judge of Appendix~\\ref{sec:judge-v2} with the deterministic checker and OlympiadBench's per-problem tolerances" if os.environ.get("SCORER", "judge") == "judge" else "the frozen matcher") + ". A recipe read on the $301$-problem subset is marked by its $n$. $U$: problems the base finishes eight times and never solves; RL $\\geq1$ ($\\geq4$): how many of them the RL model solves at least once (at least four times); $U'$ and base likewise. Robust reach counts wins in at least four of eight, net of the mirror. Intervals: paired bootstrap over problems.}\n\\label{tab:hardcells}\n\\end{table*}\n")
with open(OUT2, "w") as h:
    h.write("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llccccccccc}\n\\toprule\n")
    h.write(" & & \\multicolumn{3}{c}{filtered ($300$)} & \\multicolumn{3}{c}{same pools, unfiltered ($200$)} & \\multicolumn{3}{c}{harder ($471$)} \\\\\n")
    h.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\nrecipe & cap & fin@1 & reach & $\\Delta$ & fin@1 & reach & $\\Delta$ & fin@1 & reach & $\\Delta$ \\\\\n\\midrule\n" + "\n".join(rows2) + "\n\\bottomrule\n\\end{tabular}}\n")
    h.write("\\caption{The filter's effect: each recipe at its cap on the paper's $300$ physics problems (filtered to ones the controlled run's base sometimes solves), on $200$ problems from the same pools with the filter removed, and on the $471$ harder problems. fin@1: base$\\to$RL; reach net of mirror and the change $\\Delta$ in pass@1 with paired bootstrap intervals; all three sets scored by the frozen matcher.}\n\\label{tab:filter}\n\\end{table*}\n")
print(f"{len(rows1)} recipes -> {OUT1}, {OUT2}")
