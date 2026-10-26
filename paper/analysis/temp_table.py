"""Appendix table for the authors'-temperature check: one block per pair read twice (temp_cells.py JSONs), at 4k, 8k, 16k, the
training cap and one budget beyond, T=0.6 and the authors' temperature side by side. usage: python temp_table.py out.tex results/temp_cells_*.json"""
import json, sys, os
OUT, FILES = sys.argv[1], sys.argv[2:]
H = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
TEMP = {"Skywork-32B": "1.0", "Skywork-7B": "1.0", "Skywork-Math": "1.0", "Polaris-1.7B": "1.4", "Polaris-4B": "1.4", "QuestA": "0.7", "JustRL-N": "0.7", "JustRL-D": "0.7", "KnowRL": "0.9"}
LAB = {4096: "4k", 8192: "8k", 16384: "16k", 24576: "24k", 28672: "28k", 32768: "32k", 36864: "36k", 49152: "48k", 53248: "52k", 65536: "64k"}
def num(x, p=3): return ("$-$" if x < -0.5 * 10 ** -p else "") + ("%." + str(p) + "f") % abs(x) if abs(x) >= 0.5 * 10 ** -p else "0"
def ci(v, c): return "%s [%s, %s]" % (num(v), num(c[0]), num(c[1]))
rows = []
for f in FILES:
    name = os.path.basename(f)[len("temp_cells_"):-5]; res = json.load(open(f)); cap = PAIRS[name][2]
    caps = sorted(int(k) for k in res); at = min(c for c in caps if c >= cap); beyond = min((c for c in caps if c > at), default=None)
    show = [c for c in (4096, 8192, 16384, at, beyond) if c and c in caps]
    disp = name.replace("Skywork-32B", "Skywork-OR1-32B").replace("Skywork-7B", "Skywork-OR1-7B").replace("Skywork-Math", "Skywork-OR1-Math-7B")
    first = True
    for c in show:
        for lab, key in (("$0.6$", "T0.6"), (f"${TEMP[name]}$", "authors")):
            p = res[str(c)][key]; tag = LAB[c] + (" (cap)" if c == at else "")
            rows.append("%s & %s & %s & $%.2f\\to%.2f$ & $%.2f\\to%.2f$ & %s & %s & %s & %s \\\\" % (disp if first else "", tag if key == "T0.6" else "", lab, p["fin_b"], p["fin_r"], p["acc_b"], p["acc_r"], ci(p["F"], p["F_ci"]), ci(p["P"], p["P_ci"]), ci(p["reach"], p["reach_ci"]), ci(p["d"], p["d_ci"])))
            first = False
    rows.append("\\addlinespace[1pt]")
HEAD = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllcccccc}", "\\toprule",
        "recipe & budget & $T$ & fin@1$(B)$ & pass@1$(B\\,|\\,\\text{fin})$ & finishing $F$ & reliability $P$ & reach & change $\\Delta$ \\\\",
        " & & & base$\\to$RL & base$\\to$RL & & & net of mirror & \\\\", "\\midrule"]
TAIL = ["\\bottomrule", "\\end{tabular}}",
        "\\caption{Each pair read at temperature $0.6$ with top-$p$ $0.95$ (the paper's protocol) and at the temperature its authors evaluate with, top-$p$ $1.0$, eight samples per problem on physics, one generation at the larger cap read at each budget. Columns as in Table~\\ref{tab:capcells}.}",
        "\\label{tab:temp}", "\\end{table*}"]
open(OUT, "w").write("\n".join(HEAD + rows + TAIL) + "\n"); print("rows", sum(1 for r in rows if "&" in r))
