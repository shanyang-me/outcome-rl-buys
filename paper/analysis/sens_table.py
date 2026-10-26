"""Appendix table of the pooled cap shares under the paper's, symmetric and matched splits, from one bootstrap run (fig1_ci.json),
plus the paper's split with the bases resampled instead of the recipes. usage: python sens_table.py out.tex fig1_ci.json"""
import json, sys
OUT, J = sys.argv[1], sys.argv[2]; D = json.load(open(J))
def cells(which, key, ci="ci"):
    return " & ".join("%d [%d, %d]" % (round(100 * p), round(100 * a), round(100 * b)) for p, (a, b) in zip(D[which]["point"][key], D[which][ci][key]))
rows = []
for which, lab in (("at", "at the cap"), ("beyond", "one beyond")):
    for name, ks, ci in (("the paper's", ("size", "signed"), "ci"), ("symmetric", ("size_sym", "signed_sym"), "ci"), ("matched", ("size_m", "signed_m"), "ci"), ("the paper's, bases resampled", ("size", "signed"), "ci_base")):
        rows.append("%s & %s & %s & %s \\\\" % (lab if name == "the paper's" else "", name, cells(which, ks[0], ci), cells(which, ks[1], ci)))
    if which == "at": rows.append("\\addlinespace[2pt]")
HEAD = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llcccccc}", "\\toprule",
        " & & \\multicolumn{3}{c}{size shares (\\%)} & \\multicolumn{3}{c}{signed shares of the net change (\\%)} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}", "budget & split & finishing & reliability & reach & finishing & reliability & reach \\\\", "\\midrule"]
TAIL = ["\\bottomrule", "\\end{tabular}",
        "\\caption{The pooled shares of Figure~\\ref{fig:overview}'s cap rows under three exact splits of the same cells, from the same $2{,}000$ crossed-bootstrap replicates over recipes and problems: the paper's (the cross term in finishing, \\relB{} over each model's own finished derivations), symmetric (the cross term shared) and matched (the accuracy change taken only on problems both models finish in at least seven of eight samples, the finishing term the remainder). The last row of each block resamples the five bases, each with all its recipes, instead of the recipes.}",
        "\\label{tab:splitsens}", "\\end{table}"]
open(OUT, "w").write("\n".join(HEAD + rows + TAIL) + "\n")
for which in ("at", "beyond"):
    print(which, "matched size", cells(which, "size_m"), "| base-resampled size", cells(which, "size", "ci_base"), "| base-resampled signed", cells(which, "signed", "ci_base"))
