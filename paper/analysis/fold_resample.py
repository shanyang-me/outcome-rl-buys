"""Fold the two resampling rounds into main.tex (Oct 21): a "Reach at k=32" appendix subsection with tab:reach32 (from
reach32_table.py's JSON and rows), the harder set's +24 paragraph with tab:hard32 (from hard32_analysis.py's JSON and
hard32_table.py's rows), one sentence in the main text's reach paragraph, and the two status rows. Blocks sit between comment
markers, so a rerun with fresher data replaces them. usage: python fold_resample.py main.tex resample_dir"""
import json, re, sys
P, D = sys.argv[1], sys.argv[2]; s = open(P).read()
R = json.load(open(f"{D}/reach32_table.json")); Hn = json.load(open(f"{D}/hard32_analysis.json"))
rl_rows = open(f"{D}/reach32_rows.tex").read().strip(); h_rows = open(f"{D}/hard32_rows.tex").read().strip()
pr, pb = R["pooled"]["rl"], R["pooled"]["base"]
rec_done = [v for v in R["recipes"].values() if v["status"] == "done"]; base_done = [v for v in R["bases"].values() if v["status"] == "done"]
n_models_done = len(rec_done) + len(base_done); n_models = len(R["recipes"]) + len(R["bases"])
pct = lambda a, b: f"${100 * a / max(b, 1):.0f}\\%$"
shares = sorted(v["ge1"] / v["read"] for v in rec_done if v["read"] >= 20)
lo, hi = (f"${100 * shares[0]:.0f}\\%$", f"${100 * shares[-1]:.0f}\\%$") if shares else ("--", "--")
lo_name = min((v for v in rec_done if v["read"] >= 20), key=lambda v: v["ge1"] / v["read"], default=None)
hi_name = max((v for v in rec_done if v["read"] >= 20), key=lambda v: v["ge1"] / v["read"], default=None)
name_of = lambda v: next(k for k, x in R["recipes"].items() if x is v)
partial = n_models_done < n_models
status32 = (f"all ${n_models}$ models done; Table~\\ref{{tab:reach32}}" if not partial else
            f"${n_models_done}$ of ${n_models}$ models done, the rest running (Appendix~\\ref{{sec:reach32}})")
reach32 = f"""% reach32-block-begin
\\subsection{{Reach at $k=32$}}
\\label{{sec:reach32}}
\\textbf{{Sampled $24$ more times, the bases recover their one-sided problems at least as often as the RL models do.}} Every RL model wrote $24$ more samples at its cap on the physics problems its base had finished eight times and never solved in the paper's read (the physics part of $U$ in Table~\\ref{{tab:robustreach}}), and every base wrote $24$ more on the problems its RL models had finished eight times and never solved, once per base on the union of those sets over its recipes, at the largest of their caps (Table~\\ref{{tab:reach32}}). The RL models solve at least one of the $24$ on ${pr['ge1']}$ of their ${pr['n']}$ problems ({pct(pr['ge1'], pr['read'])}) and at least four on ${pr['ge4']}$, at ${pr['solved'] / max(pr['samples'], 1):.3f}$ solves per sample; the bases solve at least one on ${pb['ge1']}$ of ${pb['n']}$ ({pct(pb['ge1'], pb['read'])}), at least four on ${pb['ge4']}$, at ${pb['solved'] / max(pb['samples'], 1):.3f}$ per sample. Per recipe the share solved at least once runs from {lo} ({name_of(lo_name) if lo_name else ''}) to {hi} ({name_of(hi_name) if hi_name else ''}). A problem a model never solved in eight samples is solved in the next $24$ about a quarter of the time on either side, so the one-sided counts at $k=8$ are what sampling produces, and the mirror is noise of the same size rather than coverage the RL model lost. For the recipes trained at $16$k and below the base's read uses a larger budget than the RL model's, so its recovery there is an upper bound.{" Reads still running are marked in the table." if partial else ""}

\\begin{{table}}[t]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{3.5pt}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{llr|rrrr}}
\\toprule
model & base / recipes & cap & problems & $\\geq1$ of $24$ & $\\geq4$ & per sample \\\\
\\midrule
{rl_rows}
\\bottomrule
\\end{{tabular}}}}
\\caption{{Reach at $k=32$ on the paper's physics problems. RL rows: the problems the base finished eight times and never solved in the paper's read, how many of them the RL model solves in at least one and in at least four of $24$ new samples at the recipe's cap, and its solves per sample. Base rows: the union of the mirror sets over the recipes listed, read once at the largest of their caps. {"A count in italics is the part of a read finished when this version was written." if partial else ""}}}
\\label{{tab:reach32}}
\\end{{table}}
% reach32-block-end
"""
hp = Hn["pooled"]; n32 = len(Hn["pairs"]); n32_all = 22
tail32 = (" The round covers " + str(n32) + " pairs; the rest are running (Table~\\ref{tab:inprogress})." if n32 < n32_all else "")
hard32 = f"""% hard32-block-begin

\\textbf{{About half of the one-sided wins are reached by the other side with more samples.}} The $24$-sample round of \\S\\ref{{sec:reach}} was run on every one-sided problem of Table~\\ref{{tab:hardcells}}, both models writing $24$ more samples at the cap, scored by the same judge (Table~\\ref{{tab:hard32}}). Of the ${hp['rl_wins_read']}$ problems an RL model had solved while its base finished eight times and never did, the RL model solves again at least once in $24$ on ${hp['rl_wins_winner_ge1']}$ ({pct(hp['rl_wins_winner_ge1'], hp['rl_wins_read'])}) and at least four times on ${hp['rl_wins_winner_ge4']}$; the base now solves ${hp['rl_wins_loser_ge1']}$ of them ({pct(hp['rl_wins_loser_ge1'], hp['rl_wins_read'])}), and ${hp['rl_wins_winner_only']}$ ({pct(hp['rl_wins_winner_only'], hp['rl_wins_read'])}) stay RL-only after $32$ base samples. The mirror is the same: of the ${hp['base_wins_read']}$ base wins, the base repeats ${hp['base_wins_winner_ge1']}$ ({pct(hp['base_wins_winner_ge1'], hp['base_wins_read'])}), ${hp['base_wins_winner_ge4']}$ of them at least four times, the RL model now solves ${hp['base_wins_loser_ge1']}$ ({pct(hp['base_wins_loser_ge1'], hp['base_wins_read'])}), and ${hp['base_wins_winner_only']}$ stay base-only. So about half of what looks like new solving on either side is reached by the other model with more samples, and the half that is not is split evenly between the two.{tail32}

\\begin{{table*}}[t]
\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lr|rrrrr|rrrrr}}
\\toprule
 & & \\multicolumn{{5}}{{c|}}{{RL wins on $U$}} & \\multicolumn{{5}}{{c}}{{base wins on $U'$}} \\\\
recipe & cap & $n$ & RL $\\geq1$ & $\\geq4$ & base $\\geq1$ & RL only & $n$ & base $\\geq1$ & $\\geq4$ & RL $\\geq1$ & base only \\\\
\\midrule
{h_rows}
\\bottomrule
\\end{{tabular}}
\\caption{{The $24$-sample round on the harder set, scored by the fixed judge. RL wins: the problems the base finished eight times and never solved that the RL model solved at least once in eight (Table~\\ref{{tab:hardcells}}); of them, how many the RL model solves again at least once and at least four times in $24$ new samples, how many the base now solves at least once, and how many only the RL model solves. The base columns are the mirror. An $n$ written $a/b$ gives the wins in Table~\\ref{{tab:hardcells}} and the number resampled, for the models whose one-sided sets were fixed before their full-set judging.}}
\\label{{tab:hard32}}
\\end{{table*}}
% hard32-block-end"""
sent = (f"% reach32-sentence-begin\nWith $24$ more samples at the cap, the bases recover {pct(pb['ge1'], pb['read'])} of the problems their RL models never solved in eight "
        f"and the RL models {pct(pr['ge1'], pr['read'])} of their bases' (Appendix~\\ref{{sec:reach32}}).\n% reach32-sentence-end")
def put(block, begin, end, anchor_old=None, before=False, replace=False):
    global s
    if begin in s:
        a, b = s.index(begin), s.index(end) + len(end); s = s[:a] + block + s[b:]
    else:
        assert anchor_old and s.count(anchor_old) == 1, anchor_old[:60]
        s = s.replace(anchor_old, block if replace else (block + "\n\n" + anchor_old) if before else (anchor_old + "\n" + block))
put(reach32, "% reach32-block-begin", "% reach32-block-end", "\\subsection{A harder, unfiltered physics set}", before=True)
put(hard32, "% hard32-block-begin", "% hard32-block-end", "The $24$-sample resampling of every one-sided problem, as in \\S\\ref{sec:reach}, is running (Table~\\ref{tab:inprogress}).", replace=True)
put(sent, "% reach32-sentence-begin", "% reach32-sentence-end", "So reach is small because new solving is rare: a handful of problems per recipe.")
rows = {"Judging the harder set": f"judge passes done for all $27$ models; the $24$-sample round is done and judged for ${n32}$ of $22$ pairs" + (", the rest running" if n32 < n32_all else "") + " (Table~\\ref{tab:hard32})",
        "Reach at $k=32$": status32}
for lab, st in rows.items():
    m = re.search(r"^(" + re.escape(lab) + r" & .*? & )(.*?)( & .*? \\\\)$", s, re.M); assert m, lab
    s = s[:m.start(2)] + st + s[m.end(2):]
open(P, "w").write(s); print(f"folded: reach32 {n_models_done}/{n_models} models, hard32 {n32} pairs")
