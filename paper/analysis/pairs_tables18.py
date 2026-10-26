"""All tab:publicpairs floats for the 18 recipes: one page per group of recipes (by base), the first float carries the caption and
label, the others are ContinuedFloat with a short caption. Row format as pairs_rows.py.
usage: python exp2/pairs_tables18.py pairs_ci_all.json out.tex [rows_dir]  (rows_dir: suppress the per-finished interval where a model finishes on < 15 problems)"""
import json, sys, glob, collections
d = json.load(open(sys.argv[1]))
FEW = set()   # (rl, set, cap) cells where the base or the RL model finishes on fewer than 15 problems
if len(sys.argv) > 3:
    def nfin(m, cap, kind):
        by = collections.defaultdict(bool)
        for l in open(glob.glob(f"{sys.argv[3]}/*/{m}_cap{cap}.jsonl")[0]):
            r = json.loads(l)
            if r["kind"] == kind: by[r["id"]] |= bool(r["finished"])
        return sum(by.values())
    for c in d.values():
        if min(nfin(c["base"], c["cap"], c["set"]), nfin(c["rl"], c["cap"], c["set"])) < 15: FEW.add((c["rl"], c["set"], c["cap"]))
NAMES = {"DeepScaleR-1.5B-Preview": "DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B": "ProRL: Nemotron-Research-Reasoning-Qwen-1.5B (v2)",
         "Open-RS3": "Open-RS3", "STILL-3-1.5B-preview": "STILL-3-1.5B-preview",
         "JustRL-DeepSeek-1.5B": "JustRL-DeepSeek-1.5B", "DeepMath-1.5B": "DeepMath-1.5B", "RLinf-math-1.5B": "RLinf-math-1.5B",
         "AReaL-1.5B-Preview-Stage-3": "AReaL-1.5B-Preview-Stage-3",
         "Skywork-OR1-7B": "Skywork-OR1-7B", "AceReason-Nemotron-7B": "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B": "Skywork-OR1-Math-7B",
         "Polaris-4B-Preview": "Polaris-4B-Preview",
         "JustRL-Nemotron-1.5B": "JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B": "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B": "KnowRL-Nemotron-1.5B",
         "DeepMath-Omn-1.5B": "DeepMath-Omn-1.5B", "e3-1.7B": "e3-1.7B", "Polaris-1.7B-Preview": "Polaris-1.7B-Preview"}
PAGES = [(["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview"], None),
         (["JustRL-DeepSeek-1.5B", "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
          "the four added recipes on DeepSeek-R1-Distill-Qwen-1.5B"),
         (["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B", "Polaris-4B-Preview"], "the pairs on DeepSeek-R1-Distill-Qwen-7B and Qwen3-4B"),
         (["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"], "the pairs on OpenMath-Nemotron-1.5B"),
         (["e3-1.7B", "Polaris-1.7B-Preview"], "the pairs on Qwen3-1.7B")]
assert sorted(sum((p for p, _ in PAGES), [])) == sorted(NAMES)
CAPTION = r"""\caption{Eighteen released outcome-RL models against their thinking bases, decomposed ($300$ physics
problems and MATH-500, eight samples per problem). Each cell gives base$\to$RL over the change with its
$95\%$ problem-clustered bootstrap interval. ``Matched'' is the change in accuracy per finished
derivation on the problems both models finish in at least seven of eight samples (count in
parentheses; omitted below $15$ problems). ``Finishing share'' is the finishing term
$\Delta\mathrm{finished}\times\mathrm{acc}_{\mathrm{RL}}$ as a share of the capped change, shown
where that change's printed interval excludes zero: between $0$ and $100\%$ the per-finished term has
the same sign as the change (below $50\%$ it is the larger part), above $100\%$ the opposite sign, and
below zero the finishing term opposes the change. Accuracy per finished derivation has no interval where
a model finishes on fewer than $15$ problems (QuestA on physics at $4$k: $7$ finished samples on $2$
problems). Tokens: RL over base mean length, an unfinished sample counted at the cap.}
\label{tab:publicpairs}"""
HEAD = r"""\centering\scriptsize
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.0}
\begin{tabular}{lcccccr}
\toprule
set, cap & capped score, base$\to$RL & finished & \pairtwo{accuracy per}{finished derivation} & \pairtwo{matched $\Delta$}{(problems)} & finishing share & tokens \\
\midrule"""
def z(x, p):
    if abs(x) < 0.5 * 10 ** -p: return "0." + "0" * p
    return ("%+.*f" % (p, x)).replace("+0.", "+.").replace("-0.", "-.")
def two(q, p):
    return "\\pairtwo{$%.*f\\to%.*f$}{$%s$ [$%s$, $%s$]}" % (p, q["base"], p, q["rl"], z(q["delta"], p), z(q["ci"][0], p), z(q["ci"][1], p))
by = {}
for c in d.values(): by.setdefault(c["rl"], []).append(c)
out, missing = [], []
for i, (recipes, cont) in enumerate(PAGES):
    rows = []
    for r in recipes:
        cells = sorted(by.get(r, []), key=lambda c: (c["set"] != "phys", c["cap"]))
        if not cells: missing.append(r); continue
        if rows: rows.append("\\addlinespace[2pt]")
        rows.append("\\multicolumn{7}{l}{\\textit{%s} from %s} \\\\" % (NAMES[r], cells[0]["base"]))
        for c in cells:
            setcap = "%s, %dk" % ("physics" if c["set"] == "phys" else "MATH", c["cap"] // 1024)
            share = ("%.0f\\%%" % (100 * c["finishing_share"])) if (c["capped_gain"] or c["capped_loss"]) else "--"
            m = c["matched"]
            matched = ("\\pairtwo{$%s$ (%d)}{[$%s$, $%s$]}" % (z(m["delta"], 3), m["n"], z(m["ci"][0], 3), z(m["ci"][1], 3))) if "delta" in m else "-- (%d)" % m["n"]
            sgf = two(c["sgf"], 3) if (c["rl"], c["set"], c["cap"]) not in FEW else "\\pairtwo{$%.3f\\to%.3f$}{--}" % (c["sgf"]["base"], c["sgf"]["rl"])
            rows.append("\\quad %s & %s & %s & %s & %s & %s & $%.2f\\times$ \\\\" % (setcap, two(c["mean"], 3), two(c["finished"], 2), sgf, matched, share, c["len_ratio"]))
    if not rows: continue
    out += ["\\begin{table*}[tp]"] + (["\\ContinuedFloat"] if cont else []) + [HEAD] + rows + ["\\bottomrule", "\\end{tabular}"]
    out += [CAPTION if not cont else "\\caption{(continued) %s%s; columns as above.}" % (cont[0].upper(), cont[1:])]
    out += ["\\end{table*}"]
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(f"{sum(len(v) for v in by.values())} cells, {len(by)} recipes; missing: {missing}")
