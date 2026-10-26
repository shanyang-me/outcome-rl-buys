"""Figure 2 (fig:recipes) as per-recipe shares: for each recipe, at its training cap and one budget beyond, the change in score
split into finishing and reliability, drawn as shares of |F|+|P| with the percentage printed in each part; a term that lowers the
score is printed with a minus sign. The measured change is printed at the right of each bar.
usage: python fig2_share_gen.py main.tex cap_split.json [fig1_story.json]
With fig1_story.json the bar gains a third part, reach net of its mirror (green hatch; robust reach solid green), and the shares
are those of Figure 1: |F|, |P| and max(0, reach - mirror) over their sum, on physics."""
import json, sys
MAIN, CS = sys.argv[1], sys.argv[2]; D = json.load(open(CS)); R = D["recipes"]; P = D["pooled"]
ST = json.load(open(sys.argv[3]))["cap"] if len(sys.argv) > 3 else None
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
names = sorted(R, key=lambda n: (R[n]["at"]["fb"], -share3(R[n]["at"], n, "at")[0]))  # by how often the base finishes at the cap (physics), lowest first
W, H, GAP = 4.2, 0.20, 0.52
def lab(v, s): return ("$-$" if v < 0 else "") + "%d\\%%" % round(100 * s)
def panel(which, title, first):
    L = ["\\begin{tikzpicture}[x=%.2fcm,y=1cm]" % W, "\\node[font=\\small] at (0.5,%.2f) {%s};" % (GAP * 0.9, title)]
    for i, n in enumerate(names):
        y = -i * GAP; r = R[n]; c = r.get(which)
        if first:
            cap = r["at_label"] if which == "at" else r["cap_label"]
            g0 = story(n, "at"); tiny = g0 is not None and (g0["aF"] + g0["aP"]) / g0["k"] < 0.005  # the parts, not the net: large parts that cancel are not tiny
            L.append("\\node[anchor=east,font=\\footnotesize,inner sep=1pt] at (-0.02,%.2f) {%s%s \\textcolor{black!55}{\\scriptsize %s, cap %s, base finishes %d\\%%}};" % (y, "\\textcolor{red!80!black}{\\scriptsize (change $<0.005$)} " if tiny else "", n, r["base"], r["cap_label"], round(100 * r["at"]["fb"])))
        if not c:
            L.append("\\node[anchor=west,font=\\scriptsize,text=black!55] at (0,%.2f) {not read past its cap};" % y); continue
        g = story(n, which); dd = g["d"] if g else c["d"]
        s, sp, sr, srr, F, Pp = share3(c, n, which)
        fi, ri, gi = "finishink", "reliaink", "reachink"
        L.append("\\fill[%s] (0,%.2f) rectangle (%.4f,%.2f);" % (fi, y - H, s, y + H))
        L.append("\\fill[%s] (%.4f,%.2f) rectangle (%.4f,%.2f);" % (ri, s, y - H, s + sp, y + H))
        if sr > 0.002: L.append("\\fill[pattern=north east lines,pattern color=%s] (%.4f,%.2f) rectangle (1,%.2f);" % (gi, s + sp, y - H, y + H))
        if srr > 0.002: L.append("\\fill[%s] (%.4f,%.2f) rectangle (1,%.2f);" % (gi, 1 - srr, y - H, y + H))
        if s >= 0.12: L.append("\\node[font=\\scriptsize,text=white] at (%.4f,%.2f) {%s};" % (s / 2, y, lab(F, s)))
        if sp >= 0.12: L.append("\\node[font=\\scriptsize] at (%.4f,%.2f) {%s};" % (s + sp / 2, y, lab(Pp, sp)))
        sr = sr if g else None
        bud = r["at_label"] if which == "at" else r.get("beyond_label")
        pc = lambda x: "$<1$\\%" if 0 < x < 0.005 else "%d\\%%" % round(100 * x)
        rtxt = "" if sr is None else ("reach %s%s" % ("$0$" if sr <= 0 else pc(sr), (", robust %s" % pc(srr)) if srr >= 0.005 else ""))
        L.append("\\node[anchor=west,font=\\scriptsize,inner sep=1pt] at (1.02,%.2f) {%s \\textcolor{black!55}{$%+.2f$, %s}};" % (y, rtxt, dd, bud))
    yb = -(len(names) - 1) * GAP - 0.4
    L += ["\\draw[black!50,thin] (0,%.2f) -- (1,%.2f);" % (yb, yb),
          "\\foreach \\x/\\l in {0/0,0.5/half,1/all} \\node[anchor=north,font=\\scriptsize] at (\\x,%.2f) {\\l};" % (yb - 0.02),
          "\\node[anchor=north,font=\\scriptsize] at (0.5,%.2f) {share of the parts' total size};" % (yb - 0.3),
          "\\end{tikzpicture}"]
    return "\n".join(L)
def wins(which, title):
    M = max(max(R[n][which].get("rl_only_base_all_finished", 0), R[n][which].get("base_only_rl_all_finished", 0)) for n in names if R[n].get(which)) or 1
    sx = 1.0 / M
    L = ["\\begin{tikzpicture}[x=0.8cm,y=1cm]", "\\node[font=\\small] at (0,%.2f) {%s};" % (GAP * 0.9, title)]
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
          "\\tikz\\fill[pattern=north east lines,pattern color=reachink] (0,0) rectangle (0.3,0.18); reach, reach@8$(B)$\\quad \\tikz\\fill[reachink] (0,0) rectangle (0.3,0.18); robust reach, reach$_{\geq4}$@8$(B)$")
EXPECT = ("Polaris-1.7B", "Polaris-4B", "Skywork-32B")  # recipes whose read past the cap came last
_below = [n for n in names if R[n].get("below_only")]; _absent = [n for n in EXPECT if n not in R]
todo = ("" if not (_below or _absent) else
        "Not yet read at the cap: " + ", ".join(_below + _absent) + ".")
_phys_only = [n for n in names if ST and f"{n}|phys" in ST["at"]["recipes"] and f"{n}|math" not in ST["at"]["recipes"]]
if _phys_only: todo += " " + " and ".join(_phys_only) + (" are" if len(_phys_only) > 1 else " is") + " read at and past the cap on physics only."
cap = ("\\caption{What outcome RL buys at each recipe's own training cap and at one budget beyond it, on physics and MATH-500 together. "
       "Each row is one recipe, split as in Figure~\\ref{fig:overview} into finishing, reliability and reach (net of its mirror). "
       "A minus sign marks a part that lowers the score. The grey text at the right gives the mean change and the budget read. A recipe marked in red \\emph{change} $<0.005$ has finishing and reliability terms under $0.005$ in size together, so its split rests on almost nothing. "
       "Rows run from the base that finishes least often at the cap to the one that finishes most (printed at the left, physics). "
       "Finishing leads where the base still runs out of tokens; where it already finishes at the cap, finishing moves either way and reliability rises in all sixteen recipes. One budget beyond, reliability is about zero or lower for e3 and Open-RS3, two of the four whose base ran out at the cap. "
       "The wins-alone panels count physics problems that only the RL model solves (black, right) or only the base (grey, left), while the other model finished all eight samples. $^\\dagger$: the cap is soft (DeepMath's overlong penalty starts at $20$k) or read just below it (STILL-3, read at $28$k). "
       + todo + "}\n")
block = ("\\begin{figure*}[!t]\n\\centering\n\\begin{minipage}{0.86\\textwidth}\n\\definecolor{finishink}{RGB}{16,50,88}\n\\definecolor{reliaink}{RGB}{242,146,24}\n\\definecolor{reachink}{RGB}{0,158,115}\n\\resizebox{\\textwidth}{!}{%\n"
         + panel("at", "at the training cap", True) + "\\hspace{1.1cm}" + wins("at", "at the cap, wins alone") + "\\hspace{0.5cm}" + panel("beyond", "one budget beyond the cap", False) + "\\hspace{1.1cm}" + wins("beyond", "beyond the cap, wins alone") + "}\\\\[3pt]\n{\\scriptsize " + legend + "}\n\\end{minipage}\n" + cap + "\\label{fig:recipes}\n\\end{figure*}")
t = open(MAIN).read(); i = t.find("\\label{fig:recipes}"); s0 = t.rfind("\\begin{figure*}", 0, i); e0 = t.find("\\end{figure*}", i) + len("\\end{figure*}")
t = t[:s0] + block + t[e0:]; open(MAIN, "w").write(t)
print("Figure 2 shares written:", [(n, round(share(R[n]["at"])[0], 2)) for n in names])
