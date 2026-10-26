"""Table 4 rows for the 14B and 32B pairs (Oct 19): the three fixed budgets and the recipe's training cap, physics and MATH-500,
with the same problem-clustered bootstrap (20,000 resamples) and row format as pairs_ci.py / pairs_tables18.py.
usage: python pairs_extra_table.py out.tex rows_root"""
import json, glob, sys, random, collections
OUT, ROOT = sys.argv[1], sys.argv[2]; B = 20000; rng = random.Random(0)
PAIRS = [("pair_decomp5c_cap48", "AceReason-Nemotron-14B", "DeepSeek-R1-Distill-Qwen-14B", "R1-Distill-14B", (4096, 8192, 16384, 32768)),
         ("pair_decomp9_32b", "Skywork-OR1-32B", "DeepSeek-R1-Distill-Qwen-32B", "R1-Distill-32B", (4096, 8192, 16384, 24576))]
def load(d, model, cap):
    by = collections.defaultdict(list)
    for l in open(f"{ROOT}/{d}/{model}_cap{cap}.jsonl"):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return {k: (len(v), sum(r["finished"] for r in v), sum(r["score"] for r in v), sum(r["score"] for r in v if r["finished"]),
                1.0 if any(r["score"] >= 1 - 1e-9 for r in v) else 0.0, sum(r["len"] for r in v), any(r["finished"] for r in v)) for k, v in by.items()}
def stats(S, ids):
    n = sum(S[i][0] for i in ids); nf = sum(S[i][1] for i in ids)
    return {"mean": sum(S[i][2] for i in ids) / n, "finished": nf / n, "sgf": (sum(S[i][3] for i in ids) / nf) if nf else 0.0, "len": sum(S[i][5] for i in ids) / n}
def z(x, p):
    if abs(x) < 0.5 * 10 ** -p: return "0." + "0" * p
    return ("%+.*f" % (p, x)).replace("+0.", "+.").replace("-0.", "-.")
def two(q, p): return "\\pairtwo{$%.*f\\to%.*f$}{$%s$ [$%s$, $%s$]}" % (p, q["base"], p, q["rl"], z(q["delta"], p), z(q["ci"][0], p), z(q["ci"][1], p))
rows = []
for d, rl, base, blab, caps in PAIRS:
    if rows: rows.append("\\addlinespace[2pt]")
    rows.append("\\multicolumn{7}{l}{\\textit{%s} from %s} \\\\" % (rl, blab))
    for kind in ("phys", "math"):
        for cap in caps:
            Sb, Sr = load(d, base, cap), load(d, rl, cap); ids = sorted(k for k in Sb if k[0] == kind and k in Sr)
            pb, pr = stats(Sb, ids), stats(Sr, ids); boots = collections.defaultdict(list)
            for _ in range(B):
                s = rng.choices(ids, k=len(ids)); b, r = stats(Sb, s), stats(Sr, s)
                for q in ("mean", "finished", "sgf"): boots[q].append(r[q] - b[q])
            cell = {}
            for q in ("mean", "finished", "sgf"):
                bs = sorted(boots[q]); cell[q] = {"base": pb[q], "rl": pr[q], "delta": pr[q] - b[q] if False else pr[q] - pb[q], "ci": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]}
            gain, loss = round(cell["mean"]["ci"][0], 3) > 0, round(cell["mean"]["ci"][1], 3) < 0
            share = ("%.0f\\%%" % (100 * cell["finished"]["delta"] * cell["sgf"]["rl"] / cell["mean"]["delta"])) if (gain or loss) else "--"
            both = [i for i in ids if Sb[i][1] >= 7 and Sr[i][1] >= 7]
            if len(both) >= 15:
                sgf = lambda S, I: sum(S[i][3] for i in I) / sum(S[i][1] for i in I)
                pt = sgf(Sr, both) - sgf(Sb, both); bs = sorted(sgf(Sr, s) - sgf(Sb, s) for s in (rng.choices(both, k=len(both)) for _ in range(B)))
                matched = "\\pairtwo{$%s$ (%d)}{[$%s$, $%s$]}" % (z(pt, 3), len(both), z(bs[int(0.025 * B)], 3), z(bs[int(0.975 * B) - 1], 3))
            else: matched = "-- (%d)" % len(both)
            few = min(sum(S[i][6] for i in ids) for S in (Sb, Sr)) < 15
            sg = two(cell["sgf"], 3) if not few else "\\pairtwo{$%.3f\\to%.3f$}{--}" % (cell["sgf"]["base"], cell["sgf"]["rl"])
            setcap = "%s, %dk%s" % ("physics" if kind == "phys" else "MATH", cap // 1024, " (cap)" if cap == caps[-1] else "")
            rows.append("\\quad %s & %s & %s & %s & %s & %s & $%.2f\\times$ \\\\" % (setcap, two(cell["mean"], 3), two(cell["finished"], 2), sg, matched, share, pr["len"] / pb["len"]))
            print(rows[-1][:150], flush=True)
HEAD = r"""\begin{table*}[tp]
\ContinuedFloat
\centering\scriptsize
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.0}
\begin{tabular}{lcccccr}
\toprule
set, budget & capped score, base$\to$RL & \finB{} & \relB{} & \pairtwo{matched $\Delta$}{(problems)} & $F/\Delta$ & tokens \\
\midrule"""
open(OUT, "w").write("\n".join([HEAD] + rows + ["\\bottomrule", "\\end{tabular}", "\\caption{(continued) The $14$B and $32$B pairs, read in addition and not registered, at the three fixed budgets and at the recipe's training cap; columns as above, plus $F/\\Delta$, the finishing term as a share of the signed change (above $100\\%$ or negative when $F$ and $P$ have opposite signs), and tokens, the RL model's mean sample length over the base's.}", "\\end{table*}"]) + "\n")
print("EXTRA_TABLE_DONE")
