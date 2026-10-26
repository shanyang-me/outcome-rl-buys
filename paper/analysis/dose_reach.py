"""Dose regression for reach (Oct 17): robust reach net of its mirror on physics, at each recipe's training cap, against documented
RL steps of every recipe Table tab:recipes gives them for (16; round 22 added QuestA, KnowRL and Skywork-32B, which the
first version left out; ProRL v2: 3,000 in all, its model card scaling v1's 2,000 to 3,000; round 21 corrected an earlier
2,000 + 3,000 = 5,000, which gave rho 0.47). A step is not the same unit across recipes (batch sizes
differ), so this is a rank test, Spearman with a permutation p-value.
usage: python dose_reach.py fig1_story.json"""
import json, sys, random, statistics as st, math
STEPS = {"DeepScaleR": 1750, "ProRL v2": 3000, "Open-RS3": 50, "STILL-3": 210, "Skywork-7B": 1320, "Skywork-Math": 2160, "Polaris-4B": 700,
         "JustRL-D": 4380, "DeepMath": 1800, "AReaL": 435, "JustRL-N": 3440, "DeepMath-Omn": 700, "Polaris-1.7B": 500,
         "QuestA": 2000, "KnowRL": 2960, "Skywork-32B": 1000}  # round 22: every recipe whose steps Table 6 gives (16; RLinf and e3 n.d., AceReason none)
d = json.load(open(sys.argv[1]))["cap"]["at"]["recipes"]
rows = [(n, STEPS[n], d[f"{n}|phys"]["reach_robust"] - d[f"{n}|phys"]["mirror_robust"]) for n in STEPS if f"{n}|phys" in d]
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def rho(a, b):
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
x = [r[1] for r in rows]; y = [r[2] for r in rows]; r0 = rho(x, y); rng = random.Random(0); ge = 0
for _ in range(20000):
    yy = y[:]; rng.shuffle(yy); ge += abs(rho(x, yy)) >= abs(r0) - 1e-12
out = {"n": len(rows), "spearman": r0, "perm_p": ge / 20000, "rows": sorted(rows, key=lambda r: r[1])}
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "rows"}))
for n, s_, v in out["rows"]: print(f"  {n:13s} {s_:5d} steps  robust net reach {v*300:+.1f} problems")
json.dump(out, open(sys.argv[1].replace("fig1_story.json", "dose_reach.json"), "w"), indent=1)
