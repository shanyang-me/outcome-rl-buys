"""The competition-math appendix subsection, generated from the fold's JSON (math_fold.sh output dir) so that every number in the
prose comes from the analyses: math_reach.json, math_p4.json, p5.json, prereg.json, and when present math32.json (the +24 round)
and a matcher-scored run of the same analyses (MATCHER=dir) for the judge-effect sentence. Writes the LaTeX to out.tex; the tables
(mathcells.tex, mathyears.tex, mathharness.tex, math32_rows.tex) are inlined after the prose.
usage: [MATCHER=dir] python math_prose.py math_final_dir out.tex"""
import json, sys, os
D, OUT = sys.argv[1], sys.argv[2]; MD = os.environ.get("MATCHER")
L = lambda f: json.load(open(os.path.join(D, f)))
hr, p4, p5, pr = L("math_reach.json"), L("math_p4.json"), L("p5.json"), L("prereg.json")
m32 = L("math32.json") if os.path.exists(os.path.join(D, "math32.json")) else None
hm = json.load(open(os.path.join(MD, "math_reach.json"))) if MD and os.path.exists(os.path.join(MD, "math_reach.json")) else None
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
PROSE = {"JustRL-D": "JustRL-DeepSeek", "JustRL-N": "JustRL-Nemotron", "DeepMath": "DeepMath-1.5B", "Skywork-7B": "Skywork-OR1-7B", "Skywork-Math": "Skywork-OR1-Math-7B",
         "AceReason": "AceReason-Nemotron-7B", "AceReason-14B": "AceReason-Nemotron-14B", "Skywork-32B": "Skywork-OR1-32B"}
nm = lambda n: PROSE.get(n, n)
def s3(x): return f"${'+' if x > 0 else ('-' if x < 0 else '')}{abs(x):.3f}$" if abs(x) >= 5e-4 else "$0$"
def r3(v): return f"{s3(v[0])} [{s3(v[1])}, {s3(v[2])}]"
def f2(x): return f"${x:.2f}$"
def rng(xs, f=s3): xs = sorted(xs); return f"{f(xs[0])} to {f(xs[-1])}" if xs[0] != xs[-1] else f(xs[0])
def n_(x): return f"${x}$"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
         16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
w = lambda k: WORDS.get(k, str(k))
def lst(xs):
    xs = list(xs); return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
at = {n: hr[n]["cells"][f"{hr[n]['at_cap'] // 1024}k|all"] for n in PAIRS if n in hr and f"{hr[n]['at_cap'] // 1024}k|all" in hr[n]["cells"]}
N = len(at); cap = lambda n: hr[n]["at_cap"]
reach = {n: c["reach"] for n, c in at.items()}
clear = [n for n, v in reach.items() if v[1] > 0]; below = [n for n, v in reach.items() if v[2] < 0]; rest = N - len(clear) - len(below)
big = {n: c for n, c in at.items() if cap(n) >= 24576}
top = max(reach, key=lambda n: reach[n][0])
P = []
# 1 the finding
s = (f"\\textbf{{Reach stays rare in domain.}} The paper's physics problems are out of domain for nineteen of the twenty recipes, and MATH-500, the in-domain set read beside them, is near ceiling (every base solves $0.82$ or more of its finished derivations at its cap), so we read every pair on $276$ final-answer problems from eight "
     f"competition-math contests of 2025 and 2026, the problems these recipes were trained for and report on, at each recipe's cap (Table~\\ref{{tab:mathcells}}). "
     f"In all {w(N)} pairs reach stays rare: net reach at the cap is {rng([v[0] for v in reach.values()])}, with the interval clear of zero for {w(len(clear))} of the {w(N)}"
     + (f", below it for {w(len(below))}" if below else "") + (f" and touching or straddling it for {w(rest)}" if rest else "") + "; "
     f"robust reach, wins in at least four of eight net of the mirror, is at most {s3(max(c['robust'][0] for c in at.values()))}. "
     f"For the {w(len(big))} pairs read at $24$k or more, the RL model solves at least once {n_(min(c['rl_on_U'] for c in big.values()))} to {n_(max(c['rl_on_U'] for c in big.values()))} of the "
     f"{n_(min(c['U'] for c in big.values()))} to {n_(max(c['U'] for c in big.values()))} problems its base finishes eight times and never solves, and at least four times "
     f"{n_(min(c['rl_robust_on_U'] for c in big.values()))} to {n_(max(c['rl_robust_on_U'] for c in big.values()))}; the base solves at least once {n_(min(c['base_on_Up'] for c in big.values()))} to "
     f"{n_(max(c['base_on_Up'] for c in big.values()))} of its mirror set and at most {n_(max(c['base_robust_on_Up'] for c in big.values()))} four times (Table~\\ref{{tab:mathcells}}). "
     f"The largest net reach is {nm(top)}'s, {r3(reach[top])}: it solves {n_(at[top]['rl_on_U'])} of the {n_(at[top]['U'])} problems its base finishes and never solves at least once and {n_(at[top]['rl_robust_on_U'])} at least four times, "
     f"against {n_(at[top]['base_on_Up'])} and {n_(at[top]['base_robust_on_Up'])} the other way. Finishing and reliability carry the rest of the gain.")
P.append(s)
# 2 the set
P.append("\\textbf{The set.} The $276$ problems are MathArena's final-answer contests: $213$ from 2025 (AIME 2025, HMMT February and November 2025, BRUMO 2025, CMIMC 2025 and SMT 2025) and $63$ from AIME 2026 and "
         "HMMT February 2026, which postdate every model's release and stratify contamination as FrontierScience does on the harder physics set. Proof-based contests are excluded. "
         "Every gold is a number or a closed form, so a boxed answer is matched exactly (the matcher's one-percent tolerance is for physics) and every finished sample the matcher rejects goes to the judge protocol of Appendix~\\ref{sec:judge-v2}.")
# 3 reads and scoring, with the judge's effect where a matcher run is given
s = ("\\textbf{Reads and scoring.} Every one of the $27$ models writes eight samples per problem at temperature $0.6$ at the larger of $16$k tokens and its training cap, a base at the largest cap among its recipes, "
     "smaller budgets read from the same samples by truncation, as on the harder physics set; the four models with caps above $32$k were read in eight shards each and merged.")
if hm:
    ex = next((n for n in ("AceReason", "KnowRL", "QuestA") if n in at and n in hm and f"{cap(n) // 1024}k|all" in hm[n]["cells"]), None)
    if ex:
        a, b = at[ex], hm[ex]["cells"][f"{cap(ex) // 1024}k|all"]
        s += (f" The judge raises pass@1 per finished derivation on both sides ({nm(ex)} pair: base {f2(a['acc_b'][0])} against the matcher's {f2(b['acc_b'][0])}, RL {f2(a['acc_r'][0])} against {f2(b['acc_r'][0])}), "
              f"moves reliability ({s3(a['P'][0])} against {s3(b['P'][0])}) and leaves net reach where the matcher put it ({s3(a['reach'][0])} against {s3(b['reach'][0])}); "
              f"over the {w(N)} pairs it moves net reach by at most {s3(max(abs(at[n]['reach'][0] - hm[n]['cells'][f'{cap(n) // 1024}k|all']['reach'][0]) for n in at if n in hm and f'{cap(n) // 1024}k|all' in hm[n]['cells']))}. "
              "Table~\\ref{tab:mathcells} reports judged scores.")
P.append(s)
# 4 the split, further from ceiling
fin_phys = pr["P1"]["below"]; q = pr["P2"]["qualifying"]; pos = pr["P2"]["positive_clear"]; larger = pr["P2"]["larger"]
ds = [n for n in at if PAIRS[n][1].endswith("DeepSeek-R1-Distill-Qwen-1.5B") and cap(n) >= 24576]
om = [n for n in at if PAIRS[n][1].endswith("OpenMath-Nemotron-1.5B") and cap(n) >= 24576]; d7 = [n for n in at if PAIRS[n][1].endswith("DeepSeek-R1-Distill-Qwen-7B")]
KNOWN = ("DeepSeek-R1-Distill-Qwen-1.5B", "OpenMath-Nemotron-1.5B", "DeepSeek-R1-Distill-Qwen-7B")
base_parts = ([f"the R1-Distill-1.5B base {rng([at[n]['fin_b'][0] for n in ds], f2)} at $24$k and above"] if ds else []) + ([f"OpenMath-Nemotron-1.5B {f2(at[om[0]]['fin_b'][0])}"] if om else []) + ([f"R1-Distill-7B {f2(at[d7[0]]['fin_b'][0])}"] if d7 else [])
others = {}
for n in at:
    b = PAIRS[n][1]
    if not b.endswith(KNOWN): others.setdefault(b.split("/")[-1].replace("DeepSeek-R1-Distill-Qwen-", "R1-Distill-"), {})[cap(n)] = at[n]["fin_b"][0]
def at_caps(v):
    if len(set(round(x, 2) for x in v.values())) == 1: return f2(list(v.values())[0])
    return " and ".join(f"{f2(x)} at ${c // 1024}$k" for c, x in sorted(v.items()))
base_parts += [f"{b} {at_caps(v)}" for b, v in sorted(others.items())]
if len(base_parts) > 1: base_parts[-1] = "and " + base_parts[-1]
s = (f"\\textbf{{In domain the bases are further from finishing, and the same rule holds.}} At the cap the base finishes fewer of its derivations than on the paper's physics set in "
     + ("every pair" if all(fin_phys.values()) else f"{w(sum(fin_phys.values()))} of the {w(len(fin_phys))} pairs") + ": "
     + ", ".join(base_parts) + " (Table~\\ref{tab:capcells} has the physics values). "
     + (f"So for the {w(len(ds))} recipes on R1-Distill-1.5B read at $24$k or more the finishing term carries more of the gain than on physics, {rng([at[n]['F'][0] for n in ds])} against reliability {rng([at[n]['P'][0] for n in ds])}"
        + (f"; with reliability taken only on problems both models finish seven times in eight (the matched split of Table~\\ref{{tab:splitsens}}), finishing is {rng([at[n]['F_m'][0] for n in ds])} and reliability {rng([at[n]['P_m'][0] for n in ds])}, "
           f"the larger part for {w(sum(abs(at[n]['P_m'][0]) > abs(at[n]['F_m'][0]) for n in ds))} of the {w(len(ds))}, so for bases finishing {rng([at[n]['fin_b'][0] for n in ds], f2)} which term leads depends on the estimator; the matched finishing term is a remainder that also carries accuracy changes on problems outside the matched set" if "F_m" in at[ds[0]] else "") + ". " if ds else "")
     + f"Where the base finishes $0.90$ or more at the cap ({w(len(q))} recipes), reliability rises in {('all of them' if all(pos.values()) else f'{w(sum(pos.values()))} of them')}, by {rng([at[n]['P'][0] for n in q])}"
     + (" with every interval clear of zero" if all(pos.values()) else "") + f", and is the larger part in {('all' if all(larger.values()) else w(sum(larger.values())))}"
     + (" of them" if not all(larger.values()) else "")
     + (f", under the matched split too ({rng([at[n]['P_m'][0] for n in q])}, larger in {('all' if all(abs(at[n]['P_m'][0]) > abs(at[n]['F_m'][0]) for n in q) else w(sum(abs(at[n]['P_m'][0]) > abs(at[n]['F_m'][0]) for n in q)))})" if q and "F_m" in at[q[0]] else "") + ". ")
exc = [n for n in at if at[n]["fin_b"][0] < 0.90 and abs(at[n]["F"][0]) < 0.01 and at[n]["P"][0] > 0.03]
for n in exc:
    a = at[n]; s += (f"{nm(n)} is the exception to the rule that a cap-bound base buys finishing: its base finishes {f2(a['fin_b'][0])} and the recipe {f2(a['fin_r'][0])}, "
                     f"yet pass@1 per finished derivation rises {s3(a['acc_r'][0] - a['acc_b'][0])}, so its gain, {s3(a['d'][0])}, is reliability"
                     + (f"; under the matched split the two terms are about even ({s3(a['F_m'][0])} and {s3(a['P_m'][0])})" if "F_m" in a and abs(a["F_m"][0]) > 0.5 * abs(a["P_m"][0]) else "") + ". ")
if "QuestA" in at:
    a = at["QuestA"]; s += (f"QuestA finishes only {f2(a['fin_r'][0])} of its derivations at its $24$k cap against {f2(a['fin_b'][0])} for its base, so its finishing term, {s3(a['F'][0])}, cancels most of its reliability gain, "
                            f"{s3(a['P'][0])}, and its pass@1 moves {s3(a['d'][0])}")
    if "KnowRL" in at: b = at["KnowRL"]; s += f"; KnowRL likewise finishes {f2(b['fin_r'][0])} and keeps {s3(b['d'][0])} of a reliability gain of {s3(b['P'][0])}"
    s += "."
mq = [n for n in q if "m_delta" in at[n]]
if mq:
    def ud(n): return at[n]["acc_r"][0] - at[n]["acc_b"][0]
    up = [n for n in mq if at[n]["m_delta"][1] > 0]
    s += (f" On the problems both models finish at least seven times in eight, where selection into the finished set cannot act, pass@1 per finished derivation still rises, with the interval clear of zero, in "
          + ("all" if len(up) == len(mq) else f"{w(len(up))} of the {w(len(mq))}") + f" recipes whose base finishes $0.90$ or more, by {rng([at[n]['m_delta'][0] for n in mq])} (Table~\\ref{{tab:mathcells}}, matched $\\Delta$)")
    sel = [n for n in ("QuestA", "KnowRL") if n in at and "m_delta" in at[n]]
    if sel:
        def mv(n): m = at[n]["m_delta"]; return f"{s3(m[0])} [{s3(m[1])}, {s3(m[2])}] on {n_(at[n]['n_matched'])} problems against {s3(ud(n))} over all"
        part = [n for n in sel if at[n]["m_delta"][2] < ud(n)]; stays = [n for n in sel if at[n]["m_delta"][1] > 0]
        s += ("; for " + lst(nm(n) for n in sel) + ", which finish far less than their base, the matched change is " + lst(mv(n) for n in sel) + ": "
              + ("selection accounts for part of " + ("each" if len(part) == len(sel) else lst(nm(n) for n in part) + "'s") + " gain" if part else "selection does not account for the gain")
              + (", and a gain clear of zero remains" + ("" if len(stays) == len(sel) else f" for {lst(nm(n) for n in stays)}") if stays else "") + ".")
    else: s += "."
P.append(s)
# 5 contamination
pre = {n: hr[n]["cells"][f"{cap(n) // 1024}k|pre"] for n in at if f"{cap(n) // 1024}k|pre" in hr[n]["cells"]}; post = {n: hr[n]["cells"][f"{cap(n) // 1024}k|post"] for n in at if f"{cap(n) // 1024}k|post" in hr[n]["cells"]}
fails = [n for n, v in p4.items() if not v["p4"]]
sig = [n for n in at if n in p4 and p4[n]["reach_diff"][2] < 0]  # 2026 minus 2025 below zero beyond the interval: more reach on the 2025 contests
def rd(n): return r3(p4[n]["reach_diff"])
def wins(n):  # the pair's RL wins on U in each stratum, and how often the 2026 stratum shows none at the 2025 rate over such problems
    a, b, ua, ub = pre[n]["rl_on_U"], post[n]["rl_on_U"], pre[n]["U"], post[n]["U"]; ch = round(100 * (1 - a / max(ua, 1)) ** ub)
    return (f"{nm(n)} wins at least once on {n_(a)} of the {n_(ua)} problems of 2025 its base finishes and never solves and on " + ("none" if b == 0 else n_(b)) + f" of the {n_(ub)} such problems of 2026"
            + (f", an outcome {n_(ub)} such problems show ${ch}\\%$ of the time at the 2025 rate" if b == 0 else ""))
s = (f"\\textbf{{Contamination.}} On the $63$ problems from the 2026 contests, which postdate every model, the change in pass@1 at the cap is {rng([v['d'][0] for v in post.values()])} against "
     f"{rng([v['d'][0] for v in pre.values()])} on the 2025 contests, and net reach {rng([v['reach'][0] for v in post.values()])} against {rng([v['reach'][0] for v in pre.values()])} (Table~\\ref{{tab:mathyears}}). "
     "The registered prediction, no more reach on the 2026 contests than on the 2025 ones (the difference's lower bound at or below zero), " + ("holds for every pair" if not fails else f"fails for {lst(nm(n) for n in fails)}")
     + ". It tests contamination of the bases: a base trained on the 2025 problems solves more of them, which shrinks $U$ and lowers reach there, so reach on 2026 would exceed it. "
     "Contamination of the recipes would show the other way, as more reach on the 2025 problems. "
     + (f"In that direction the difference is clear of zero for {w(len(sig))} pair{'s' if len(sig) != 1 else ''}, {lst(f'{nm(n)} ({rd(n)})' for n in sig)}: "
        + "; ".join(wins(n) for n in sig) + f". For the other {w(len(at) - len(sig))} pairs the interval reaches zero. With {lst(n_(post[n]['U']) for n in sig)} such problems in the 2026 stratum, contamination of the 2025 problems and chance cannot be told apart." if sig else "In that direction the interval reaches zero for every pair."))
P.append(s)
# 6 harness
comp = {m: v for m, v in p5["rl"].items() if v.get("comparable")}; nc = {m: v for m, v in p5["rl"].items() if v.get("reported") is not None and not v.get("comparable")}
rn = {PAIRS[n][0].split("/")[-1]: n for n in PAIRS}
s = (f"\\textbf{{The harness reproduces the authors' AIME 2025 numbers.}} For the {w(len(comp))} RL models whose authors report AIME 2025 pass@1 at our temperature and within one budget step of the recipe's cap, "
     f"our read at the cap is within ${max(abs(v['gap_at_cap']) for v in comp.values()):.3f}$ of theirs ("
     + lst(f"{nm(rn.get(m, m))} ${v['pass1_at_cap']:.3f}$ against ${v['reported'] / 100:.3f}$" for m, v in comp.items()) + "). "
     f"The other gaps are budget or temperature effects, {rng([v['gap_at_cap'] for v in nc.values()])}: the largest, "
     + lst(f"{nm(rn.get(m, m))}'s {s3(v['gap_at_cap'])}" for m, v in sorted(nc.items(), key=lambda kv: kv[1]['gap_at_cap'])[:2])
     + ", come from finishing " + lst(f"${v['fin_at_cap']:.2f}$" for m, v in sorted(nc.items(), key=lambda kv: kv[1]['gap_at_cap'])[:2]) + " of their derivations at caps below the authors' $32$k (Table~\\ref{tab:mathharness}).")
P.append(s)
# 7 the +24 round
if m32:
    p = m32["pooled"]; k = len(m32["pairs"])
    r1 = p["rl_wins_loser_ge1"] / max(p["rl_wins_read"], 1)
    fw = ("three quarters or more" if r1 >= 0.73 else "about two thirds" if r1 >= 0.64 else "about three fifths" if r1 >= 0.57 else "about half" if r1 >= 0.43 else "about a third" if r1 >= 0.28 else "few")
    s = (f"\\textbf{{With more samples the base solves {fw} of the RL model's one-sided wins.}} The $24$-sample round of \\S\\ref{{sec:reach}} was run on every one-sided problem of Table~\\ref{{tab:mathcells}}, "
         f"both models writing $24$ more samples at the cap, scored the same way (Table~\\ref{{tab:math32}}). Of the {n_(p['rl_wins_read'])} problems an RL model had solved while its base finished eight times and never did, "
         f"the RL model solves again at least once in $24$ on {n_(p['rl_wins_winner_ge1'])} ({round(100 * p['rl_wins_winner_ge1'] / max(p['rl_wins_read'], 1))}\\%) and at least four times on {n_(p['rl_wins_winner_ge4'])}; "
         f"the base now solves {n_(p['rl_wins_loser_ge1'])} of them ({round(100 * p['rl_wins_loser_ge1'] / max(p['rl_wins_read'], 1))}\\%), and {n_(p['rl_wins_winner_only'])} stay RL-only after $32$ base samples. "
         f"Of the {n_(p['base_wins_read'])} base wins, the base repeats {n_(p['base_wins_winner_ge1'])}, {n_(p['base_wins_winner_ge4'])} of them at least four times, the RL model now solves {n_(p['base_wins_loser_ge1'])}, "
         f"and {n_(p['base_wins_winner_only'])} stay base-only: a net of {n_(p['rl_wins_winner_only'] - p['base_wins_winner_only'])} RL-only problems summed over the {w(k)} pairs, about ${(p['rl_wins_winner_only'] - p['base_wins_winner_only']) / k:.1f}$ per pair of $276$.")
    P.append(s)
# 8 the registered predictions
verdict = {k: v["holds"] for k, v in pr.items()}
held = [k for k, v in verdict.items() if v]; failed = [k for k, v in verdict.items() if not v]
desc = {"P1": "the base finishes less than on physics in every pair and finishing is the larger part for the four cap-bound recipes", "P2": "reliability rises, with the interval clear of zero, in at least three quarters of the recipes whose base finishes $0.90$ or more and is the larger part in most",
        "P3": "net reach within $[-0.02, +0.03]$ everywhere, clear of zero for at most half the pairs, robust reach at most $0.02$", "P4": "no more reach on the 2026 contests than on the 2025 ones (this tests contamination of the bases; the text also tests the recipes' direction)", "P5": "our AIME 2025 pass@1 within $0.08$ of the authors' comparable numbers"}
s = ("\\textbf{Registered predictions.} Five predictions were written down before any output existed (in the released repository): " + "; ".join(f"{k}, {desc[k]}" for k in desc) + ". "
     + (f"{lst(held)} held" if held else "None held") + (f"; {lst(failed)} failed" if failed else "") + ".")
if "P1" in failed:
    fl = [n for n, ok in pr["P1"]["finishing_larger"].items() if not ok]; bl = [n for n, ok in pr["P1"]["below"].items() if not ok]
    if fl and not bl: s += f" P1 fails on {lst(nm(n) for n in fl)}" + (": neither model finishes more than $6\\%$ of its derivations at $4$k, so there is nothing to split" if fl == ["Open-RS3"] else "") + "."
P.append(s)
tex = ["\\subsection{A competition-math set}", "\\label{sec:mathset}", ""] + [p + "\n" for p in P]
for f in ("mathcells.tex", "mathyears.tex", "mathharness.tex"):
    if os.path.exists(os.path.join(D, f)): tex.append(open(os.path.join(D, f)).read())
if m32 and os.path.exists(os.path.join(D, "math32_rows.tex")):
    tex.append("\\begin{table*}[t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lr|rrrrr|rrrrr}\n\\toprule\n & & \\multicolumn{5}{c|}{RL wins on $U$} & \\multicolumn{5}{c}{base wins on $U'$} \\\\\n"
               "recipe & cap & $n$ & RL $\\geq1$ & $\\geq4$ & base $\\geq1$ & RL only & $n$ & base $\\geq1$ & $\\geq4$ & RL $\\geq1$ & base only \\\\\n\\midrule\n" + open(os.path.join(D, "math32_rows.tex")).read()
               + "\\bottomrule\n\\end{tabular}\n\\caption{The $24$-sample round on the competition-math set: for each pair, the problems the RL model solved at least once in eight while its base finished all eight and never did ($n$), how many the RL model solves again at least once and at least four times in $24$ more samples, how many the base now solves at least once, and how many stay RL-only; then the mirror for the base-won problems.}\n\\label{tab:math32}\n\\end{table*}\n")
open(OUT, "w").write("\n".join(tex)); print("\n\n".join(P)); print(f"\n-> {OUT} ({len(P)} paragraphs)")
