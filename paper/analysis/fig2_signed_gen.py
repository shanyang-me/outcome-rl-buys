"""Figure 2 (fig:recipes) in signed pass@1 points on physics, from Table 6's cells (cap_cells.json): for each recipe, at its training
cap and one budget beyond, the change over the base split into finishing F, reliability P (mirror added back) and reach net of its
mirror, stacked from zero (a part that lowers the score lies left of zero), with the change and the budget at the right. The
wins-alone panels are unchanged (cap_split.json). usage: python fig2_signed_gen.py main.tex cap_split.json cap_cells.json"""
import json, sys
MAIN, CS = sys.argv[1], sys.argv[2]; D = json.load(open(CS)); R = D["recipes"]; P = D["pooled"]
ST = None; CC = json.load(open(sys.argv[3]))
def cell(n, which):
    xs = [x for x in CC.get(n, {}).get("cells", []) if x["which"] == which and x["kind"] == "phys"]
    return xs[0] if xs else None
def story(n, which):
    # the recipe's cells on both benchmarks, pooled exactly as Figure 1 pools its rows
    if not ST or which not in ST: return None
    cs = [ST[which]["recipes"][k] for k in (n + "|phys", n + "|math") if k in ST[which]["recipes"]]
    if not cs: return None
    return {"F": sum(c["F"] for c in cs), "aF": sum(abs(c["F"]) for c in cs), "aP": sum(abs(c["P"] + c["mirror"]) for c in cs),
            "P": sum(c["P"] + c["mirror"] for c in cs), "net": sum(c["reach"] - c["mirror"] for c in cs),
            "rob": sum(c["reach_robust"] - c["mirror_robust"] for c in cs), "d": sum(c["d"] for c in cs) / len(cs), "k": len(cs)}
def share3(c, n, which):
    g = story(n, which)
    if not g: F, Pp = c["F"], c["P"]; t = abs(F) + abs(Pp) or 1; return abs(F) / t, abs(Pp) / t, 0.0, 0.0, F, Pp
    net = max(0.0, g["net"]); rob = min(net, max(0.0, g["rob"]))  # net reach, clipped after pooling, as in Figure 1
    t = g["aF"] + g["aP"] + net or 1; return g["aF"] / t, g["aP"] / t, net / t, rob / t, g["F"], g["P"]
def share(c): F, Pp = c["F"], c["P"]; tot = abs(F) + abs(Pp); return (abs(F) / tot if tot > 1e-9 else 0.5), F, Pp
names = sorted(R, key=lambda n: (cell(n, "at")["fin_b"], n))  # by how often the base finishes at the cap (physics), lowest first
W, H, GAP = 4.2, 0.20, 0.52
X0, X1 = -0.06, 0.21; SC = 1.0 / (X1 - X0)          # the panel spans 0..1 in tikz x units
px = lambda v: (v - X0) * SC
def sg(v): return ("$+%.3f$" % v) if v > 0.0005 else ("$-%.3f$" % abs(v) if v < -0.0005 else "$0$")
def panel(which, title, first):
    L = ["\\begin{tikzpicture}[x=%.2fcm,y=1cm,baseline=0pt]" % W, "\\node[font=\\small] at (0.5,%.2f) {%s};" % (GAP * 0.9, title)]
    for i, n in enumerate(names):
        y = -i * GAP; r = R[n]; x = cell(n, which)
        if first:
            x0 = cell(n, "at"); tiny = x0 is not None and abs(x0["F"][0]) + abs(x0["P"][0]) < 0.005
            L.append("\\node[anchor=east,font=\\footnotesize,inner sep=1pt] at (-0.02,%.2f) {%s%s \\textcolor{black!55}{\\scriptsize %s, cap %s, base finishes %d\\%%}};" % (y, "\\textcolor{red!80!black}{\\scriptsize (parts $<0.005$)} " if tiny else "", n, r["base"], r["cap_label"], round(100 * cell(n, "at")["fin_b"])))
        if not x:
            L.append("\\node[anchor=west,font=\\scriptsize,text=black!55] at (%.4f,%.2f) {not read past its cap};" % (px(0), y)); continue
        pos, neg = 0.0, 0.0
        for k, style in (("F", "finishink"), ("P", "reliaink"), ("reach", "pattern=north east lines,pattern color=reachink")):
            v = x[k][0]
            if abs(v) < 1e-6: continue
            if v > 0: a, b = pos, pos + v; pos = b
            else: a, b = neg + v, neg; neg = a
            L.append("\\fill[%s] (%.4f,%.2f) rectangle (%.4f,%.2f);" % (style, px(a), y - H, px(b), y + H))
            if abs(v) >= 0.08: L.append("\\node[font=\\scriptsize,text=%s,inner sep=0.5pt] at (%.4f,%.2f) {%s};" % ("white" if k == "F" else "black", px((a + b) / 2), y, sg(v)))
        bud = r["at_label"] if which == "at" else r.get("beyond_label")
        L.append("\\node[anchor=west,font=\\scriptsize,inner sep=1pt] at (%.4f,%.2f) {%s \\textcolor{black!55}{%s}};" % (px(max(pos, 0)) + 0.01, y, sg(x["d"][0]), bud))
    yb = -(len(names) - 1) * GAP - 0.4
    L += ["\\draw[black!60,thin] (%.4f,%.2f) -- (%.4f,%.2f);" % (px(0), GAP * 0.4, px(0), yb),
          "\\draw[black!50,thin] (0,%.2f) -- (1,%.2f);" % (yb, yb),
          "\\foreach \\v/\\l in {-0.05/$-0.05$,0/$0$,0.05/$0.05$,0.1/$0.10$,0.15/$0.15$,0.2/$0.20$} \\node[anchor=north,font=\\scriptsize] at ({(\\v+%.2f)*%.4f},%.2f) {\\l};" % (-X0, SC, yb - 0.02),
          "\\node[anchor=north,font=\\scriptsize] at (0.5,%.2f) {change in pass@1 over the base, physics};" % (yb - 0.3),
          "\\end{tikzpicture}"]
    return "\n".join(L)
def wins(which, title):
    M = max(max(R[n][which].get("rl_only_base_all_finished", 0), R[n][which].get("base_only_rl_all_finished", 0)) for n in names if R[n].get(which)) or 1
    sx = 1.0 / M
    L = ["\\begin{tikzpicture}[x=0.8cm,y=1cm,baseline=0pt]", "\\node[font=\\small] at (0,%.2f) {%s};" % (GAP * 0.9, title)]
    for i, n in enumerate(names):
        y = -i * GAP; c = R[n].get(which)
        if not c: continue
        a_, b_ = c.get("rl_only_base_all_finished", 0), c.get("base_only_rl_all_finished", 0)
        L.append("\\fill[black!35] (%.3f,%.2f) rectangle (0,%.2f);" % (-b_ * sx, y - H, y + H))
        L.append("\\fill[black] (0,%.2f) rectangle (%.3f,%.2f);" % (y - H, a_ * sx, y + H))
        L.append("\\node[anchor=east,font=\\scriptsize,inner sep=1pt] at (%.3f,%.2f) {%d};" % (-b_ * sx - 0.03, y, b_))
        L.append("\\node[anchor=west,font=\\scriptsize,inner sep=1pt] at (%.3f,%.2f) {%d};" % (a_ * sx + 0.03, y, a_))
    yb = -(len(names) - 1) * GAP - 0.4
    L += ["\\draw[black!60,thin] (0,%.2f) -- (0,%.2f);" % (GAP * 0.4, yb),
          "\\node[anchor=north,font=\\scriptsize] at (-0.75,%.2f) {base only};" % (yb - 0.02), "\\node[anchor=north,font=\\scriptsize] at (0.75,%.2f) {RL only};" % (yb - 0.02), "\\end{tikzpicture}"]
    return "\n".join(L)
pa, pb = P.get("at", {}), P.get("beyond", {}); full = [n for n in names if not R[n].get("below_only")]
legend = ("\\tikz\\fill[finishink] (0,0) rectangle (0.3,0.18); finishing term $F$\\quad \\tikz\\fill[reliaink] (0,0) rectangle (0.3,0.18); reliability term $P$\\quad "
          "\\tikz\\fill[pattern=north east lines,pattern color=reachink] (0,0) rectangle (0.3,0.18); reach, net of its mirror")
EXPECT = ("Polaris-1.7B", "Polaris-4B", "Skywork-32B")  # recipes whose read past the cap came last
_below = [n for n in names if R[n].get("below_only")]; _absent = [n for n in EXPECT if n not in R]
todo = ("" if not (_below or _absent) else
        "Not yet read at the cap: " + ", ".join(_below + _absent) + ".")
_phys_only = [n for n in names if ST and f"{n}|phys" in ST["at"]["recipes"] and f"{n}|math" not in ST["at"]["recipes"]]
if _phys_only: todo += " " + " and ".join(_phys_only) + (" are" if len(_phys_only) > 1 else " is") + " read at and past the cap on physics only."
cap = ("\\caption{What outcome RL buys at each recipe's own training cap and at one budget beyond it, in pass@1 points on physics (Table~\\ref{tab:capcells}'s cells; it also has MATH-500). "
       "Each row is one recipe, split as in Figure~\\ref{fig:overview}; a part left of zero lowers the score, and the number at the right is the change, with the budget read. "
       "Red \\emph{parts} $<0.005$: finishing and reliability are under $0.005$ in size together, so the split rests on almost nothing. "
       "Rows run from the base that finishes least often at the cap to the one that finishes most (left, physics). "
       "Finishing leads where the base still runs out of tokens at the cap; where it finishes, reliability rises in all sixteen recipes. "
       "The wins-alone panels count physics problems that only the RL model solves (black, right) or only the base (grey, left), while the other model finished all eight samples. $^\\dagger$: the cap is soft (DeepMath's overlong penalty starts at $20$k) or read just below it (STILL-3, read at $28$k).}\n")
block = ("\\begin{figure*}[!t]\n\\centering\n\\begin{minipage}{0.86\\textwidth}\n\\definecolor{finishink}{RGB}{16,50,88}\n\\definecolor{reliaink}{RGB}{242,146,24}\n\\definecolor{reachink}{RGB}{0,158,115}\n\\resizebox{\\textwidth}{!}{%\n"
         + panel("at", "at the training cap", True) + "\\hspace{1.1cm}" + wins("at", "at the cap, wins alone") + "\\hspace{0.5cm}" + panel("beyond", "one budget beyond the cap", False) + "\\hspace{1.1cm}" + wins("beyond", "beyond the cap, wins alone") + "}\\\\[3pt]\n{\\scriptsize " + legend + "}\n\\end{minipage}\n" + cap + "\\label{fig:recipes}\n\\end{figure*}")
t = open(MAIN).read(); i = t.find("\\label{fig:recipes}"); s0 = t.rfind("\\begin{figure*}", 0, i); e0 = t.find("\\end{figure*}", i) + len("\\end{figure*}")
t = t[:s0] + block + t[e0:]; open(MAIN, "w").write(t)
print("Figure 2 signed written:", [(n, round(cell(n, "at")["d"][0], 3) if cell(n, "at") else None) for n in names])
