"""Regenerate the fig:recipes_budgets block (appendix, fixed-budget view) of main.tex from the per-recipe data files.
usage: python fig2_gen.py main.tex fig1_recipes.json overlap_per_recipe_phys.json [extra_pairs.json]
extra_pairs.json: {"Name": {"base": "tag", "cap": "label", "4096": {...F,P,d,...overlap fields...}, ...}} merged in as extra rows."""
import json, re, sys
MAIN, FREC, FOVL = sys.argv[1], sys.argv[2], sys.argv[3]; EXTRA = sys.argv[4] if len(sys.argv) > 4 else None
R = json.load(open(FREC)); O = json.load(open(FOVL))
BASE = {"DeepScaleR": "R1-1.5B", "ProRL v2": "R1-1.5B", "Open-RS3": "R1-1.5B", "STILL-3": "R1-1.5B", "JustRL-D": "R1-1.5B", "DeepMath": "R1-1.5B", "RLinf": "R1-1.5B", "AReaL": "R1-1.5B",
        "Skywork-7B": "R1-7B", "AceReason": "R1-7B", "Skywork-Math": "R1-7B", "Polaris-4B": "Q3-4B", "JustRL-N": "OMN-1.5B", "QuestA": "OMN-1.5B", "KnowRL": "OMN-1.5B", "DeepMath-Omn": "OMN-1.5B", "e3": "Q3-1.7B", "Polaris-1.7B": "Q3-1.7B"}
CAP = {"DeepScaleR": "8$\\to$24k", "ProRL v2": "8k", "Open-RS3": "3.5k", "STILL-3": "29k$^\\dagger$", "Skywork-7B": "16$\\to$32k", "Skywork-Math": "8$\\to$32k", "AceReason": "8$\\to$32k", "Polaris-4B": "40$\\to$52k",
       "JustRL-D": "15k", "DeepMath": "24k$^\\dagger$", "RLinf": "27.6k", "AReaL": "8$\\to$24k", "JustRL-N": "15k", "QuestA": "24k", "KnowRL": "24k", "DeepMath-Omn": "24k$^\\dagger$", "e3": "8$\\to$16k", "Polaris-1.7B": "40$\\to$48k"}
CAPS = ("4096", "8192", "16384"); LAB = {"4096": "4", "8192": "8", "16384": "16"}
if EXTRA:
    for name, d in json.load(open(EXTRA)).items():
        BASE[name] = d["base"]; CAP[name] = d["cap"]
        R[name] = {c: {k: d[c][k] for k in ("F", "P", "d")} for c in CAPS}
        for c in CAPS: O[c][name] = {k: v for k, v in d[c].items() if k not in ("F", "P", "d")}
names = sorted(R, key=lambda n: (O["16384"][n]["rl_only"] - O["16384"][n]["base_only"]) / O["16384"][n]["n"] if O["16384"][n]["rl_only"] > 1 else O["16384"][n]["rl_only"] - O["16384"][n]["base_only"])
ys = ",".join(names); top, bot = names[-1], names[0]; XMIN, XMAX = -0.4, 0.3
def share(o, k): return o[k] / o["n"] if o[k] > 1 or o["n"] > 1 else o[k]   # overlap files store shares for the 18, counts for extras
def counts(c):
    g = [k for k in names if R[k][c]["d"] >= 0.005]; l = [k for k in names if R[k][c]["d"] <= -0.005]
    gf = [k for k in g if R[k][c]["F"] > 0 and R[k][c]["F"] >= R[k][c]["P"]]; lf = [k for k in l if R[k][c]["F"] < 0 and -R[k][c]["F"] >= abs(R[k][c]["P"])]
    rl = [k for k in g + l if abs(R[k][c]["P"]) >= abs(R[k][c]["F"])]
    return len(gf), len(g), len(lf), len(l), rl
def split_panel(c, first):
    gf, g, lf, l, rl = counts(c); s = []
    s.append("\\begin{tikzpicture}\n\\begin{axis}[width=" + ("5.8cm" if first else "4.6cm") + ",height=8.6cm,xmin=%s,xmax=%s,enlarge y limits=0.04,clip=false,\n" % (XMIN, XMAX))
    s.append("  axis x line*=bottom,axis y line*=left,axis line style={black!60,thin},ytick style={draw=none},xtick={-0.3,-0.15,0,0.15,0.3},xticklabels={$-.3$,$-.15$,0,$.15$,$.3$},xtick style={draw=none},\n")
    s.append("  symbolic y coords={" + ys + "},ytick=data,yticklabels={},x tick label style={font=\\footnotesize},title style={font=\\small,yshift=-3pt},title={$%s$k: finishing carries %d of %d gains, %d of %d losses},\n" % (LAB[c], gf, g, lf, l))
    s.append("  xlabel={change in score vs base},label style={font=\\footnotesize}]\n\\addplot[draw=none,forget plot] coordinates {" + " ".join("(0,%s)" % k for k in names) + "};\n")
    for k in names:
        F, P = R[k][c]["F"], R[k][c]["P"]; d = F + P
        s.append("\\fill[finishink] ([yshift=1.4pt]axis cs:%.4f,%s) rectangle ([yshift=3.6pt]axis cs:%.4f,%s);\n" % (min(0, F), k, max(0, F), k))
        s.append("\\fill[reliaink] ([yshift=-1.1pt]axis cs:%.4f,%s) rectangle ([yshift=1.1pt]axis cs:%.4f,%s);\n" % (min(0, P), k, max(0, P), k))
        s.append("\\fill[black] ([yshift=-3.6pt]axis cs:%.4f,%s) rectangle ([yshift=-1.4pt]axis cs:%.4f,%s);\n" % (min(0, d), k, max(0, d), k))
    s.append("\\draw[black!60,thin] (axis cs:0,%s) -- (axis cs:0,%s);\n" % (bot, top))
    for k in names: s.append("\\node[anchor=west,font=\\footnotesize,inner sep=1pt] at (axis cs:%s,%s) {$%+.2f$};\n" % (XMAX + 0.01, k, R[k][c]["d"]))
    if first:
        for k in names: s.append("\\node[anchor=east,font=\\small,inner sep=1pt] at ([xshift=-3pt]axis cs:%s,%s) {%s \\textcolor{black!55}{\\footnotesize %s}};\n" % (XMIN, k, k, BASE[k]))
    s.append("\\node[anchor=south west,font=\\footnotesize,inner sep=1pt] at ([yshift=6pt]axis cs:%s,%s) {$\\Delta$};\n" % (XMAX + 0.01, top))
    for k in rl: s.append("\\fill[reliaink] (axis cs:-0.385,%s) circle (1.5pt);\n" % k)
    s.append("\\end{axis}\n\\end{tikzpicture}"); return "".join(s)
def wins_panel(c, last):
    s = ["\\begin{tikzpicture}\n\\begin{axis}[width=3.3cm,height=8.6cm,xmin=-20,xmax=20,enlarge y limits=0.04,clip=false,\n",
         "  axis x line*=bottom,axis y line*=left,axis line style={black!60,thin},ytick style={draw=none},xtick={-10,0,10},xticklabels={10,0,10},xtick style={draw=none},\n",
         "  symbolic y coords={" + ys + "},ytick=data,yticklabels={},x tick label style={font=\\footnotesize},title style={font=\\small,yshift=-3pt},title={$%s$k: within budget, wins alone},\n" % LAB[c],
         "  xlabel={base only \\hfill RL only},label style={font=\\footnotesize}]\n\\addplot[draw=none,forget plot] coordinates {" + " ".join("(0,%s)" % k for k in names) + "};\n"]
    for k in names:
        o = O[c][k]; bn = o["base_only_rl_all_finished"]; rn = o["rl_only_base_all_finished"]
        s.append("\\fill[black] ([yshift=-2.6pt]axis cs:%d,%s) rectangle ([yshift=2.6pt]axis cs:0,%s);\n" % (-bn, k, k))
        s.append("\\fill[black] ([yshift=-2.6pt]axis cs:0,%s) rectangle ([yshift=2.6pt]axis cs:%d,%s);\n" % (k, rn, k))
    s.append("\\draw[black!60,thin] (axis cs:0,%s) -- (axis cs:0,%s);\n" % (bot, top))
    if last:
        for k in names: s.append("\\node[anchor=west,font=\\footnotesize,inner sep=1pt] at ([xshift=0.25cm]axis cs:20,%s) {%s};\n" % (k, CAP[k]))
        s.append("\\node[anchor=south west,font=\\footnotesize,inner sep=1pt] at ([xshift=0.25cm,yshift=6pt]axis cs:20,%s) {train cap};\n" % top)
        s.append("\\draw[->,black!55,line width=0.8pt] ([xshift=1.75cm,yshift=4pt]axis cs:20,%s) -- ([xshift=1.75cm,yshift=-4pt]axis cs:20,%s) node[midway,sloped,above,font=\\footnotesize,text=black!70]{$16$k read: RL-only excess shrinks, base-only excess grows};\n" % (top, bot))
    s.append("\\end{axis}\n\\end{tikzpicture}"); return "".join(s)
row = "\\resizebox{\\textwidth}{!}{" + "\\hspace{3pt}".join(split_panel(c, c == "4096") + "\\hspace{3pt}" + wins_panel(c, c == "16384") for c in CAPS).replace("\\hspace{3pt}\\begin{tikzpicture}\n\\begin{axis}[width=5.8cm", "\\hspace{12pt}\\begin{tikzpicture}\n\\begin{axis}[width=5.8cm").replace("\\hspace{3pt}\\begin{tikzpicture}\n\\begin{axis}[width=4.6cm", "\\hspace{12pt}\\begin{tikzpicture}\n\\begin{axis}[width=4.6cm") + "}"
legend = ("\\resizebox{\\textwidth}{!}{\\begin{tikzpicture}\n\\begin{axis}[hide axis,xmin=0,xmax=1,ymin=0,ymax=1,scale only axis,width=16cm,height=1.2cm,legend style={font=\\scriptsize,draw=none,fill=none,at={(0.5,0.5)},anchor=center,legend columns=2,/tikz/every even column/.append style={column sep=0.4cm}},legend cell align=left]\n"
          "\\addlegendimage{area legend,fill=finishink,draw=finishink}\\addlegendentry{finishing term: how much more or less often it finishes}\n"
          "\\addlegendimage{area legend,fill=reliaink,draw=reliaink}\\addlegendentry{reliability term: how much more or less often it is right when it finishes}\n"
          "\\addlegendimage{area legend,fill=black,draw=black}\\addlegendentry{measured change, RL minus base (the two terms added)}\n"
          "\\addlegendimage{only marks,mark=*,mark size=1.5pt,reliaink}\\addlegendentry{reliability is the larger term in this recipe}\n"
          "\\addlegendimage{area legend,fill=black,draw=black}\\addlegendentry{problems only one model solves although the other finished all eight samples}\n"
          "\\end{axis}\n\\end{tikzpicture}}")
t = open(MAIN).read(); i = t.find("\\label{fig:recipes_budgets}"); s0 = t.rfind("\\begin{figure*}", 0, i); e0 = t.find("\\end{figure*}", i) + len("\\end{figure*}")
old = t[s0:e0]; cap = old[old.find("\\caption{"):old.find("\\label{fig:recipes_budgets}")]
C = {c: counts(c) for c in CAPS}
cap = re.sub(r"Every loss at every budget is a finishing loss.*?all but AReaL\. ",
             "Every loss at every budget is a finishing loss, a model that finishes less often than its base because training lengthened its derivations (%d of %d at $4$k, %d of %d at $8$k, %d of %d at $16$k), and most gains are finishing gains, a model that finishes more often (%d of %d, %d of %d, %d of %d); an amber dot at the left marks a recipe whose reliability term is the larger. Pooling the $4$k and $8$k cells as the text does, finishing is the larger term in $16$ of the $17$ recipes that change, all but AReaL. " % (C["4096"][2], C["4096"][3], C["8192"][2], C["8192"][3], C["16384"][2], C["16384"][3], C["4096"][0], C["4096"][1], C["8192"][0], C["8192"][1], C["16384"][0], C["16384"][1]), cap, flags=re.S)
new = "\\begin{figure*}[!t]\n\\centering\n\\definecolor{finishink}{RGB}{16,50,88}\n\\definecolor{reliaink}{RGB}{242,146,24}\n" + row + "\\\\[-2pt]\n" + legend + "\n\\vspace{-6pt}\n" + cap + "\\label{fig:recipes_budgets}\n\\end{figure*}"
t = t[:s0] + new + t[e0:]; open(MAIN, "w").write(t); print("fig:recipes regenerated:", len(names), "rows; top", top, "bottom", bot, "| counts", C)
