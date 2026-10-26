"""The paper's Figure 'fig:pairs': for each of the 108 cells (18 recipes x 2 sets x 3 caps) the change in capped mean score
against its finishing term F, as pgfplots coordinates, plus the caption's fit numbers (variance of the change accounted for
by the diagonal y = F, i.e. 1 - sum((y-F)^2)/sum((y-ybar)^2), by subset) and the count of recipes with P > 0 at 16k on physics.
usage: python fig_pairs.py results/pairs_ci_all_final.json [out.tex]"""
import json, sys

d = json.load(open(sys.argv[1]))
cells = list(d.values())

def fit(sub):
    y = [c["mean"]["delta"] for c in sub]; F = [c["finishing_term"] for c in sub]; ybar = sum(y) / len(y)
    return 1 - sum((b - a) ** 2 for a, b in zip(F, y)) / sum((b - ybar) ** 2 for b in y)

print("diagonal accounts for: all %.3f | 4k+8k %.3f | 16k %.3f | MATH %.3f | physics %.3f" % (
    fit(cells), fit([c for c in cells if c["cap"] != 16384]), fit([c for c in cells if c["cap"] == 16384]),
    fit([c for c in cells if c["set"] == "math"]), fit([c for c in cells if c["set"] == "phys"])))
print("P > 0 at 16k on physics:", sum(1 for c in cells if c["set"] == "phys" and c["cap"] == 16384 and c["mean"]["delta"] - c["finishing_term"] > 0), "of 18")

order = [("phys", 4096), ("math", 4096), ("phys", 8192), ("math", 8192), ("phys", 16384), ("math", 16384)]
style = {("phys", 4096): "mark=*,mark size=1.6pt,black", ("phys", 8192): "mark=*,mark size=1.6pt,black!55",
         ("phys", 16384): "mark=o,mark size=1.6pt,black,thick", ("math", 4096): "mark=square*,mark size=1.4pt,blue!70!black",
         ("math", 8192): "mark=square*,mark size=1.4pt,blue!35", ("math", 16384): "mark=square,mark size=1.4pt,blue!70!black,thick"}
name = {("phys", 4096): "physics $4$k", ("phys", 8192): "physics $8$k", ("phys", 16384): "physics $16$k",
        ("math", 4096): "MATH-500 $4$k", ("math", 8192): "MATH-500 $8$k", ("math", 16384): "MATH-500 $16$k"}
lines = []
for key in order:
    pts = " ".join("(%.3f,%.3f)" % (c["finishing_term"], c["mean"]["delta"]) for c in cells if (c["set"], c["cap"]) == key)
    lines.append("\\addplot[only marks,%s] coordinates {%s};" % (style[key], pts))
lines.append("\\legend{" + ",".join(name[k] for k in order) + "}")
out = "\n".join(lines) + "\n"
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(out)
else:
    print(out[:400], "...")
