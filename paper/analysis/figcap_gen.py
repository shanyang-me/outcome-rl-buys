"""Write/refresh two figure blocks in main.tex from cap_split.json:
 - fig:capsplit  (figure*): per recipe, the split and the wins-alone panel at the training cap and beyond it;
 - the at/beyond panel inside fig:overview (between the pooled shares panel and the overlap panel).
usage: python figcap_gen.py main.tex cap_split.json"""
import json, re, sys
MAIN, CS = sys.argv[1], sys.argv[2]; D = json.load(open(CS)); R = D["recipes"]; P = D["pooled"]
names = sorted(R, key=lambda n: R[n]["at"]["F"])   # bottom-up: largest finishing gain at cap on top
full = [n for n in names if not R[n].get("below_only")]
ys = ",".join(names); top, bot = names[-1], names[0]; XMIN, XMAX = -0.4, 0.3
def split_panel(which, title, first):
    s = ["\\begin{tikzpicture}\n\\begin{axis}[width=" + ("5.8cm" if first else "4.6cm") + ",height=%.1fcm,xmin=%s,xmax=%s,enlarge y limits=0.06,clip=false,\n" % (max(3.2, 0.36 * len(names) + 1.2), XMIN, XMAX),
         "  axis x line*=bottom,axis y line*=left,axis line style={black!60,thin},ytick style={draw=none},xtick={-0.3,-0.15,0,0.15,0.3},xticklabels={$-.3$,$-.15$,0,$.15$,$.3$},xtick style={draw=none},\n",
         "  symbolic y coords={" + ys + "},ytick=data,yticklabels={},x tick label style={font=\\footnotesize},title style={font=\\small,yshift=-3pt},title={" + title + "},\n",
         "  xlabel={change in score vs base},label style={font=\\footnotesize}]\n\\addplot[draw=none,forget plot] coordinates {" + " ".join("(0,%s)" % k for k in names) + "};\n"]
    for k in names:
        c = R[k].get(which)
        if not c: s.append("\\node[anchor=west,font=\\scriptsize,text=black!60,inner sep=1pt] at (axis cs:%s,%s) {not read past its cap};\n" % (XMIN + 0.01, k)); continue
        F, Pp = c["F"], c["P"]; d = F + Pp
        s.append("\\fill[finishink] ([yshift=1.4pt]axis cs:%.4f,%s) rectangle ([yshift=3.6pt]axis cs:%.4f,%s);\n" % (min(0, F), k, max(0, F), k))
        s.append("\\fill[reliaink] ([yshift=-1.1pt]axis cs:%.4f,%s) rectangle ([yshift=1.1pt]axis cs:%.4f,%s);\n" % (min(0, Pp), k, max(0, Pp), k))
        s.append("\\fill[black] ([yshift=-3.6pt]axis cs:%.4f,%s) rectangle ([yshift=-1.4pt]axis cs:%.4f,%s);\n" % (min(0, d), k, max(0, d), k))
        s.append("\\node[anchor=west,font=\\footnotesize,inner sep=1pt] at (axis cs:%s,%s) {$%+.2f$ \\textcolor{black!55}{\\scriptsize %s}};\n" % (XMAX + 0.01, k, c["d"], R[k]["at_label" if which == "at" else "beyond_label"]))
    s.append("\\draw[black!60,thin] (axis cs:0,%s) -- (axis cs:0,%s);\n" % (bot, top))
    if first:
        for k in names: s.append("\\node[anchor=east,font=\\small,inner sep=1pt] at ([xshift=-3pt]axis cs:%s,%s) {%s \\textcolor{black!55}{\\footnotesize %s, cap %s}};\n" % (XMIN, k, k, R[k]["base"], R[k]["cap_label"]))
    s.append("\\node[anchor=south west,font=\\footnotesize,inner sep=1pt] at ([yshift=6pt]axis cs:%s,%s) {$\\Delta$};\n\\end{axis}\n\\end{tikzpicture}" % (XMAX + 0.01, top)); return "".join(s)
def wins_panel(which, title):
    s = ["\\begin{tikzpicture}\n\\begin{axis}[width=3.3cm,height=%.1fcm,xmin=-20,xmax=20,enlarge y limits=0.06,clip=false,\n" % max(3.2, 0.36 * len(names) + 1.2),
         "  axis x line*=bottom,axis y line*=left,axis line style={black!60,thin},ytick style={draw=none},xtick={-10,0,10},xticklabels={10,0,10},xtick style={draw=none},\n",
         "  symbolic y coords={" + ys + "},ytick=data,yticklabels={},x tick label style={font=\\footnotesize},title style={font=\\small,yshift=-3pt},title={" + title + "},\n",
         "  xlabel={base only \\hfill RL only},label style={font=\\footnotesize}]\n\\addplot[draw=none,forget plot] coordinates {" + " ".join("(0,%s)" % k for k in names) + "};\n"]
    for k in names:
        c = R[k].get(which)
        if not c: continue
        s.append("\\fill[black] ([yshift=-2.6pt]axis cs:%d,%s) rectangle ([yshift=2.6pt]axis cs:0,%s);\n" % (-c["base_only_rl_all_finished"], k, k))
        s.append("\\fill[black] ([yshift=-2.6pt]axis cs:0,%s) rectangle ([yshift=2.6pt]axis cs:%d,%s);\n" % (k, c["rl_only_base_all_finished"], k))
    s.append("\\draw[black!60,thin] (axis cs:0,%s) -- (axis cs:0,%s);\n\\end{axis}\n\\end{tikzpicture}" % (bot, top)); return "".join(s)
pa, pb = P.get("at", {}), P.get("beyond", {})
row = "\\resizebox{\\textwidth}{!}{" + split_panel("at", "at the training cap", True) + "\\hspace{3pt}" + wins_panel("at", "at the cap, wins alone") + "\\hspace{12pt}" + split_panel("beyond", "beyond the training cap", False) + "\\hspace{3pt}" + wins_panel("beyond", "beyond the cap, wins alone") + "}"
legend = ("\\resizebox{\\textwidth}{!}{\\begin{tikzpicture}\n\\begin{axis}[hide axis,xmin=0,xmax=1,ymin=0,ymax=1,scale only axis,width=16cm,height=0.9cm,legend style={font=\\scriptsize,draw=none,fill=none,at={(0.5,0.5)},anchor=center,legend columns=2,/tikz/every even column/.append style={column sep=0.4cm}},legend cell align=left]\n"
          "\\addlegendimage{area legend,fill=finishink,draw=finishink}\\addlegendentry{finishing term: how much more or less often it finishes}\n"
          "\\addlegendimage{area legend,fill=reliaink,draw=reliaink}\\addlegendentry{reliability term: how much more or less often it is right when it finishes}\n"
          "\\addlegendimage{area legend,fill=black,draw=black}\\addlegendentry{measured change, RL minus base (the two terms added)}\n"
          "\\addlegendimage{area legend,fill=black,draw=black}\\addlegendentry{problems only one model solves although the other finished all eight samples}\n\\end{axis}\n\\end{tikzpicture}}")
cap = ("\\caption{What outcome RL buys at each recipe's own training cap, and one budget beyond it, on physics: one $64$k-cap generation per model read at the derived budgets ($%d$ of $18$ released recipes read so far; the figure fills in as the read completes, and the fixed-budget view of all eighteen is Figure~\\ref{fig:recipes_budgets}). Left pair: at the smallest budget at or above the recipe's final training cap (printed after the change), three bars from zero per recipe: the finishing term in navy, the reliability term in amber, their sum in black, and the within-budget wins alone. Right pair: the same at the next budget up. "
       "Pooled over the recipes read, the mean finishing term is $%+.3f$ at the cap and $%+.3f$ beyond it, the mean reliability term $%+.3f$ and $%+.3f$; finishing is the larger term in $%d$ of $%d$ recipes at the cap and $%d$ of $%d$ beyond it, and every recipe read finishes more often than its base at its cap ($%d$ of $%d$). Recipes whose read did not reach their cap are marked.}\n"
       % (len(full), pa.get("F_mean", 0), pb.get("F_mean", 0), pa.get("P_mean", 0), pb.get("P_mean", 0), pa.get("finishing_larger", 0), pa.get("n", 0), pb.get("finishing_larger", 0), pb.get("n", 0), pa.get("finish_more", 0), pa.get("n", 0)))
block = "\\begin{figure*}[!t]\n\\centering\n\\definecolor{finishink}{RGB}{16,50,88}\n\\definecolor{reliaink}{RGB}{242,146,24}\n" + row + "\\\\[-2pt]\n" + legend + "\n\\vspace{-6pt}\n" + cap + "\\label{fig:capsplit}\n\\end{figure*}"
t = open(MAIN).read()
block = block.replace("\\label{fig:capsplit}", "\\label{fig:recipes}")
if "\\label{fig:capsplit}" in t:   # drop an older separate copy
    i = t.find("\\label{fig:capsplit}"); s0 = t.rfind("\\begin{figure*}", 0, i); e0 = t.find("\\end{figure*}", i) + len("\\end{figure*}"); t = t[:s0] + t[e0:].lstrip("\n")
i = t.find("\\label{fig:recipes}"); s0 = t.rfind("\\begin{figure*}", 0, i); e0 = t.find("\\end{figure*}", i) + len("\\end{figure*}")
oldfig = t[s0:e0]
if "The split and the overlap, recipe by recipe and budget by budget" in oldfig:   # first run: move the budget figure to the appendix
    oldfig = oldfig.replace("\\label{fig:recipes}", "\\label{fig:recipes_budgets}").replace("\\begin{figure*}[!t]", "\\begin{figure*}[t]")
    t = t[:s0] + block + t[e0:]
    k = t.find("\\label{sec:publicpairs}"); k = t.find("\n", k) + 1; t = t[:k] + oldfig + "\n" + t[k:]
else:
    t = t[:s0] + block + t[e0:]
# Figure 1: at/beyond panel (three bars pooled), inserted or refreshed between the shares panel and the overlap panel
panel = ("\\begin{tikzpicture}\n\\begin{axis}[scale only axis,width=4.6cm,height=1.0cm,xmin=-0.1,xmax=0.25,clip=false,\n"
         "  axis x line*=bottom,axis y line*=left,axis line style={black!60,thin},xtick={0,0.1,0.2},xticklabels={0,.1,.2},xtick style={draw=none},ytick style={draw=none},\n"
         "  symbolic y coords={at,beyond},ytick=data,yticklabels={},y dir=reverse,enlarge y limits=0.5,\n"
         "  tick label style={font=\\footnotesize},label style={font=\\footnotesize},xlabel={mean change vs base ($%d$ recipes read so far)}]\n"
         "\\addplot[draw=none,forget plot] coordinates {(0,at) (0,beyond)};\n" % pa.get("n", 0))
for key, F, Pp in (("at", pa.get("F_mean", 0), pa.get("P_mean", 0)), ("beyond", pb.get("F_mean", 0), pb.get("P_mean", 0))):
    d = F + Pp
    panel += "\\fill[finishink] ([yshift=2pt]axis cs:%.4f,%s) rectangle ([yshift=5.5pt]axis cs:%.4f,%s);\n" % (min(0, F), key, max(0, F), key)
    panel += "\\fill[reliaink] ([yshift=-1.7pt]axis cs:%.4f,%s) rectangle ([yshift=1.7pt]axis cs:%.4f,%s);\n" % (min(0, Pp), key, max(0, Pp), key)
    panel += "\\fill[black] ([yshift=-5.5pt]axis cs:%.4f,%s) rectangle ([yshift=-2pt]axis cs:%.4f,%s);\n" % (min(0, d), key, max(0, d), key)
panel += "\\draw[black!60,thin] (axis cs:0,at) -- (axis cs:0,beyond);\n\\node[anchor=east,font=\\footnotesize,inner sep=1pt] at ([xshift=-3pt]axis cs:-0.1,at) {at the training cap};\n\\node[anchor=east,font=\\footnotesize,inner sep=1pt] at ([xshift=-3pt]axis cs:-0.1,beyond) {beyond the cap};\n\\end{axis}\n\\end{tikzpicture}"
mark_s, mark_e = "%% CAPPANEL-BEGIN\n", "%% CAPPANEL-END\n"
if mark_s in t:
    s0 = t.find(mark_s); e0 = t.find(mark_e) + len(mark_e); t = t[:s0] + mark_s + panel + "\\\\[4pt]\n" + mark_e + t[e0:]
else:
    print("no CAPPANEL block in Figure 1 (fig1_gen.py owns it now); skipped")
open(MAIN, "w").write(t); print("fig:capsplit and the Figure 1 cap panel written:", len(names), "recipes;", len(full), "with a beyond-cap read")
