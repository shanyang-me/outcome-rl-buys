"""Rewrite Figure 1 (fig:overview) in main.tex as one ledger from fig1_story.json: rows = budgets 4k/8k/16k over the registered recipes,
then each recipe's training cap and one budget beyond (64k read); segments = finishing, reliability, reach (net of the mirror).
usage: python fig1_gen.py main.tex fig1_story.json"""
import json, sys
MAIN, J = sys.argv[1], sys.argv[2]; t = open(MAIN).read(); D = json.load(open(J))
CI = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None   # fig1_ci.json: intervals for the cap rows
W, HALF = 5.0, 0.21  # axis width in cm, half bar height in cm
rows = [("4k", D["budget"]["4096"]["pooled"], 3.0), ("8k", D["budget"]["8192"]["pooled"], 2.3), ("16k", D["budget"]["16384"]["pooled"], 1.6),
        ("at the training cap", D["cap"]["at"]["pooled"], 0.5), ("beyond the cap", D["cap"]["beyond"]["pooled"], -0.2)]
n18, ncap = D["budget"]["16384"]["pooled"]["n"], D["cap"]["at"]["pooled"]["n"]
pct = lambda x: ("$<1$" if 0 < x < 0.01 else "$%d$" % round(100 * x)) + "\\%"
L = ["\\resizebox{\\linewidth}{!}{\\begin{tikzpicture}[x=%.2fcm,y=1cm]" % W]
for name, p, y in rows:
    F, P, R = p["F"], p["P"], p["reach"]
    L.append("\\fill[finishink] (0,%.2f) rectangle (%.4f,%.2f);" % (y - HALF, F, y + HALF))
    L.append("\\fill[reliaink] (%.4f,%.2f) rectangle (%.4f,%.2f);" % (F, y - HALF, F + P, y + HALF))
    if R > 0.003: L.append("\\fill[pattern=north east lines,pattern color=reachink] (%.4f,%.2f) rectangle (1,%.2f);" % (F + P, y - HALF, y + HALF))
    Rr = p.get("reach_robust", 0.0)
    if Rr > 0.002: L.append("\\fill[reachink] (%.4f,%.2f) rectangle (1,%.2f);" % (1 - Rr, y - HALF, y + HALF))
    L.append("\\node[anchor=east,font=\\footnotesize,text=white,inner sep=1pt] at (%.4f,%.2f) {%s};" % (F - 0.012, y, pct(F)))
    L.append("\\node[font=\\footnotesize,inner sep=1pt] at (%.4f,%.2f) {%s};" % (F + P / 2, y, pct(P)))
    L.append("\\node[anchor=west,font=\\footnotesize,inner sep=1pt] at (1.015,%.2f) {reach %s%s};" % (y, "$0$" if R <= 0 else pct(R), (", robust %s" % pct(p["reach_robust"])) if p.get("reach_robust", 0) >= 0.005 else ""))
    L.append("\\node[anchor=east,font=\\footnotesize,inner sep=1pt] at (-0.02,%.2f) {%s};" % (y, name))
L += ["\\draw[black!60,thin] (0,-0.5) -- (0,3.3);", "\\draw[black!60,thin] (0,-0.5) -- (1,-0.5);",
      "\\foreach \\x/\\l in {0/0,0.5/half,1/all} \\node[anchor=north,font=\\footnotesize] at (\\x,-0.52) {\\l};",
      "\\node[anchor=north,font=\\footnotesize] at (0.5,-0.82) {share of the parts' total size};",
      "\\node[anchor=east,font=\\scriptsize,text=black!60,inner sep=1pt] at (-0.02,3.33) {budget, $%d$ recipes};" % n18,
      "\\node[anchor=east,font=\\scriptsize,text=black!60,inner sep=1pt] at (-0.02,0.83) {training cap, $%d$ recipes};" % ncap,
      "\\draw[black!60,thin,dashed] (-0.55,1.15) -- (1.0,1.15);",
      "\\node[anchor=north west,font=\\footnotesize,inner sep=1pt,align=left,text width=10.2cm] at (-0.62,-1.2) {\\tikz\\fill[finishink] (0,0) rectangle (0.25,0.18); finishing term $F$\\quad \\tikz\\fill[reliaink] (0,0) rectangle (0.25,0.18); reliability term $P$\\\\ \\tikz\\fill[pattern=north east lines,pattern color=reachink] (0,0) rectangle (0.25,0.18); reach, reach@8$(B)$\\quad \\tikz\\fill[reachink] (0,0) rectangle (0.25,0.18); robust reach, reach$_{\geq4}$@8$(B)$};",
      "\\end{tikzpicture}}"]
body = "\n".join(L) + "\n"
ms, me = "%% FIG1-BEGIN\n", "%% FIG1-END\n"
if ms in t:
    s0 = t.find(ms); e0 = t.find(me) + len(me); t = t[:s0] + ms + body + me + t[e0:]
else:
    i = t.find("\\label{fig:overview}"); fs = t.rfind("\\begin{figure}", 0, i); blk = t[fs:i]
    s0 = fs + blk.find("\\begin{tikzpicture}"); e0 = fs + blk.find("\\caption{")
    t = t[:s0] + ms + body + me + t[e0:]
if "\\usetikzlibrary{patterns}" not in t:
    t = t.replace("\\usepackage{pgfplots}", "\\usepackage{pgfplots}\n\\usetikzlibrary{patterns}", 1)
p16 = D["budget"]["16384"]["pooled"]; pa, pb = D["cap"]["at"]["pooled"], D["cap"]["beyond"]["pooled"]
bench = "physics and MATH-500" if "math" in D.get("kinds", ["phys", "math"]) else "physics"
def iv(which, j):
    p, (a, b) = CI[which]["point"]["size"][j], CI[which]["ci"]["size"][j]; return "$%d\\%%$ [$%d$, $%d$]" % (round(100 * p), round(100 * a), round(100 * b))
def dv(which):
    p, (a, b) = CI[which]["point"]["diff"][0], CI[which]["ci"]["diff"][0]; return "$%d$ points [$%d$, $%d$]" % (round(100 * p), round(100 * a), round(100 * b))
caption = ("\\caption{\\sloppy What outcome RL buys under a token budget. Each row splits the pooled change over the base on %s into finishing, reliability and reach net of its mirror, which add up to the change (Eqs.~\\eqref{eq:finrel}--\\eqref{eq:reach}), and shows each part's size as a share of the three; the solid green segment is robust reach (Eq.~\\eqref{eq:robustreach}). Sizes, not signs, because some recipes finish less than their base while gaining in reliability. "
  "Top: $%d$ recipes at three fixed budgets. Bottom: each recipe at its training cap and one budget beyond ($%d$ recipes, with the $14$B and $32$B pairs; per recipe in Figure~\\ref{fig:recipes}). "
  "" % (bench, n18, ncap)
  + ("Reach is %s at the cap and %s beyond (robust %s and %s). Signed parts, in pass@1 points, are in \\S\\ref{sec:reliability}. Intervals are $95\\%%$, over recipes and problems. Both Polaris pairs are read at and past the cap on physics only.}\n"
     % (iv("at", 2), iv("beyond", 2), pct(pa.get("reach_robust", 0)), pct(pb.get("reach_robust", 0))) if CI else
     "Reach is %s at the cap and %s beyond (robust %s and %s).}\n" % (pct(pa["reach"]), pct(pb["reach"]), pct(pa.get("reach_robust", 0)), pct(pb.get("reach_robust", 0)))))
i = t.find("\\label{fig:overview}"); s = t.rfind("\\caption{", 0, i); t = t[:s] + caption + t[i:]
open(MAIN, "w").write(t); print("Figure 1 ledger written:", [(r[0], round(r[1]["F"], 3), round(r[1]["P"], 3), round(r[1]["reach"], 3)) for r in rows])
