"""Figure 1 (fig:overview) in signed pass@1 points on physics, from Table 6's cells (cap_cells.json): at each recipe's training
cap and one budget beyond, the mean over recipes of the change over the base, split into finishing F, reliability P (mirror added
back) and reach net of its mirror, which add up to the change. Rows by regime: the recipes whose base still runs out of tokens at
the cap, those whose base finishes, and all twenty. Positive parts stack right of zero, negative parts left. Replaces the block
between the FIG1 markers and the caption. usage: python fig1_signed_gen.py main.tex cap_cells.json"""
import json, sys
import numpy as np
MAIN, CJ = sys.argv[1], sys.argv[2]; t = open(MAIN).read(); C = json.load(open(CJ))
BOUND = ("ProRL v2", "Open-RS3", "e3", "JustRL-D")
def mean(which, keep):
    cells = [x for n in C if keep(n) for x in C[n]["cells"] if x["which"] == which and x["kind"] == "phys"]
    return {k: float(np.mean([x[k][0] for x in cells])) for k in ("F", "P", "reach", "d")}, len(cells)
groups = [("cap-bound", lambda n: n in BOUND), ("finishes at cap", lambda n: n not in BOUND), ("all", lambda n: True)]
X = 60.0                       # cm per unit of pass@1 (0.01 = 0.6 cm)
H = 0.19                       # half bar height
x0, x1 = -0.01, 0.09           # axis range
f = lambda v: ("$+%.3f$" % v) if v > 0 else ("$%.3f$" % v if v < 0 else "$0$")
L = ["\\resizebox{\\linewidth}{!}{\\begin{tikzpicture}[x=%.1fcm,y=1cm]" % X]
y = 0.0; ylabels = []
for which, title in (("at", "at the training cap"), ("beyond", "one budget beyond")):
    L.append("\\node[anchor=east,font=\\footnotesize\\itshape,text=black!70,inner sep=1pt] at (%.5f,%.2f) {%s};" % (x0 - 0.001, y + 0.36, title))
    for name, keep in groups:
        m, n = mean(which, keep)
        pos, neg = 0.0, 0.0
        for k, style in (("F", "finishink"), ("P", "reliaink"), ("reach", "pattern=north east lines,pattern color=reachink")):
            v = m[k]
            if abs(v) < 1e-6: continue
            if v > 0: a, b = pos, pos + v; pos = b
            else: a, b = neg + v, neg; neg = a
            L.append("\\fill[%s] (%.5f,%.2f) rectangle (%.5f,%.2f);" % (style if "pattern" in style else style, a, y - H, b, y + H))
            if abs(v) >= 0.012:
                col = "white" if k in ("F",) else "black"
                L.append("\\node[font=\\scriptsize,text=%s,inner sep=0.5pt] at (%.5f,%.2f) {%s};" % (col, (a + b) / 2, y, f(v)))
        L.append("\\node[anchor=west,font=\\scriptsize,inner sep=1pt] at (%.5f,%.2f) {= %s (%s, %s, %s)};" % (max(pos, 0) + 0.0012, y, f(m["d"]), f(m["F"]), f(m["P"]), f(m["reach"])))
        L.append("\\node[anchor=east,font=\\footnotesize,inner sep=1pt] at (%.5f,%.2f) {%s ($%d$)};" % (x0 - 0.001, y, name, n))
        y -= 0.46
    y -= 0.30
ybot = y + 0.30 + 0.46 - H - 0.12
L += ["\\draw[black!60,thin] (0,%.2f) -- (0,0.62);" % ybot,
      "\\draw[black!60,thin] (%.4f,%.2f) -- (%.4f,%.2f);" % (x0, ybot, x1, ybot),
      "\\foreach \\x/\\l in {-0.01/$-0.01$,0/$0$,0.02/$0.02$,0.04/$0.04$,0.06/$0.06$,0.08/$0.08$} \\node[anchor=north,font=\\scriptsize] at (\\x,%.2f) {\\l};" % (ybot - 0.02),
      "\\node[anchor=north,font=\\footnotesize] at (0.04,%.2f) {change in pass@1 over the base, physics, mean over recipes};" % (ybot - 0.33),
      "\\node[anchor=north west,font=\\footnotesize,inner sep=1pt,align=left] at (%.4f,%.2f) {\\tikz[x=1cm,y=1cm]\\fill[finishink] (0,0) rectangle (0.25,0.18); finishing $F$\\quad \\tikz[x=1cm,y=1cm]\\fill[reliaink] (0,0) rectangle (0.25,0.18); reliability $P$\\quad \\tikz[x=1cm,y=1cm]\\fill[pattern=north east lines,pattern color=reachink] (0,0) rectangle (0.25,0.18); reach, net of its mirror};" % (x0 - 0.034, ybot - 0.72),
      "\\end{tikzpicture}}"]
body = "\n".join(L) + "\n"
ms, me = "%% FIG1-BEGIN\n", "%% FIG1-END\n"
assert ms in t and me in t, "Figure 1 markers missing"
s0 = t.find(ms); e0 = t.find(me) + len(me); t = t[:s0] + ms + body + me + t[e0:]
mb, _ = mean("at", lambda n: n in BOUND); ma, _ = mean("at", lambda n: True); mo, _ = mean("beyond", lambda n: n in BOUND)
caption = ("\\caption{\\sloppy What outcome RL buys, in pass@1 points on physics. Each row is the mean over its recipes of the change over the base at the recipe's training cap, or one budget beyond, "
           "split into finishing, reliability and reach net of its mirror, which add up to it (Eqs.~\\eqref{eq:finrel}--\\eqref{eq:reach}); the bracket lists the three parts. "
           "Where the base still runs out of tokens at the cap (cap-bound: ProRL v2, Open-RS3, e3, JustRL-DeepSeek) the gain is finishing; where it already finishes, reliability. "
           "Taken only on problems both models finish (Table~\\ref{tab:capcells}, matched $P$), reliability at the cap is $+0.021$, $+0.032$ and $+0.030$ for the three groups. "
           "Each recipe is in Figure~\\ref{fig:recipes} and Table~\\ref{tab:capcells}.}\n")
i = t.find("\\label{fig:overview}"); s = t.rfind("\\caption{", 0, i); t = t[:s] + caption + t[i:]
open(MAIN, "w").write(t); print("signed Figure 1 written; at cap all:", {k: round(v, 3) for k, v in ma.items()})
