"""LaTeX rows for the harder set's +24 table (Oct 21), from hard32_analysis.py's JSON. Per pair: the RL-won problems (in U, RL solved
at least once in eight) with how many were resampled, how many the RL model solves again at least once and at least four times
in 24, how many the base now solves at least once, and how many stay RL-only; then the mirror for the base-won problems.
usage: python hard32_table.py hard32_analysis.json rows.tex"""
import json, sys
A = json.load(open(sys.argv[1])); lines = []
def cell(x):
    rd = f"{x['n']}" if x["read"] == x["n"] else f"{x['n']}/{x['read']}"
    if not x["read"]: return f"{rd} & -- & -- & -- & --"
    return f"{rd} & {x['winner_ge1']} & {x['winner_ge4']} & {x['loser_ge1']} & {x['winner_only']}"
for name, o in A["pairs"].items():
    lines.append(f"{name} & {o['cap'] // 1024}k & {cell(o['rl_wins'])} & {cell(o['base_wins'])} \\\\")
p = A["pooled"]; lines.append("\\midrule")
lines.append(f"all & & {p['rl_wins_read']} & {p['rl_wins_winner_ge1']} & {p['rl_wins_winner_ge4']} & {p['rl_wins_loser_ge1']} & {p['rl_wins_winner_only']} & "
             f"{p['base_wins_read']} & {p['base_wins_winner_ge1']} & {p['base_wins_winner_ge4']} & {p['base_wins_loser_ge1']} & {p['base_wins_winner_only']} \\\\")
open(sys.argv[2], "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
