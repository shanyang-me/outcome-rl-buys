"""Round 22 (N5, N6, W4): Appendix A's pooled shares at the cap and at the next budget, on physics over the twenty recipes
(paper_tables/capcells.csv), with the formula the appendix states: sum|F|, sum|P| and max(0, sum reach) over their total. Matched:
P replaced by matched P and F by the remainder (change - matched P - reach), on recipes with a matched read (all at the cap; 19 beyond,
e3 has none). Intervals: bootstrap over the seven bases (bases drawn with replacement, all their recipes kept; 20,000 draws, seed 0).
Also reliability's lead over finishing (P share - F share) with its base-resampled interval.
usage: python3 -I pooled_shares.py [capcells.csv]"""
import csv, os, random, sys
CAP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_tables", "capcells.csv")
BASE = {"Open-RS3": "R1-1.5B", "ProRL v2": "R1-1.5B", "JustRL-D": "R1-1.5B", "DeepScaleR": "R1-1.5B", "DeepMath": "R1-1.5B", "AReaL": "R1-1.5B",
        "RLinf": "R1-1.5B", "STILL-3": "R1-1.5B", "JustRL-N": "OMN-1.5B", "QuestA": "OMN-1.5B", "KnowRL": "OMN-1.5B", "DeepMath-Omn": "OMN-1.5B",
        "e3": "Q3-1.7B", "Polaris-1.7B": "Q3-1.7B", "Polaris-4B": "Q3-4B", "Skywork-7B": "R1-7B", "Skywork-Math": "R1-7B", "AceReason": "R1-7B",
        "AceReason-Nemotron-14B": "R1-14B", "Skywork-OR1-32B": "R1-32B"}
v = lambda s: None if s.strip() in ("--", "") else float(s.split()[0])
rows = {}; cur = None
for r in csv.reader(l for l in open(CAP) if not l.startswith("#")):
    if len(r) < 9 or r[0] == "recipe" or r[2] == "base->RL": continue
    cur = r[0] or cur
    if r[1].startswith("physics"): rows.setdefault(cur, []).append(dict(F=v(r[4]), P=v(r[5]), mP=v(r[6]), reach=v(r[7]), ch=v(r[8])))
assert set(rows) == set(BASE)
def shares(names, j, matched):
    cells = [rows[n][j] for n in names if not matched or rows[n][j]["mP"] is not None]
    sF = sum(abs(c["ch"] - c["mP"] - c["reach"] if matched else c["F"]) for c in cells); sP = sum(abs(c["mP"] if matched else c["P"]) for c in cells)
    sR = max(0.0, sum(c["reach"] for c in cells)); T = sF + sP + sR
    return 100 * sF / T, 100 * sP / T, 100 * sR / T, len(cells)
bases = sorted(set(BASE.values())); by = {b: [n for n in BASE if BASE[n] == b] for b in bases}
for j, where in ((0, "at the cap"), (1, "next budget")):
    for matched in (False, True):
        f, p, r, n = shares(list(BASE), j, matched); rng = random.Random(0); fs, ps, ls = [], [], []
        for _ in range(20000):
            names = [x for b in (rng.choice(bases) for _ in bases) for x in by[b]]
            if matched and not any(rows[x][j]["mP"] is not None for x in names): continue
            a, c, _, _ = shares(names, j, matched); fs.append(a); ps.append(c); ls.append(c - a)
        q = lambda xs: (sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1])
        print("%-11s %-7s (%d recipes): finishing %.0f%% %s, reliability %.0f%% %s, reach %.0f%%; reliability lead %.0f points [%.0f, %.0f]"
              % (where, "matched" if matched else "plain", n, f, "[%.0f, %.0f]" % q(fs), p, "[%.0f, %.0f]" % q(ps), r, p - f, *q(ls)))
