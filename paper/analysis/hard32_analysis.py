"""The harder set's +24 round (Oct 20): on every one-sided problem of a pair (RL solved it at least once in eight while the base,
finishing all eight, never did; or the reverse), both models wrote 24 more samples at the recipe's cap. For each pair this reports
whether each win repeats: of the RL-won problems, how many the RL model solves again at least once (and at least four times) in 24,
and how many the base now solves at least once; and the same for the base-won problems. The original wins come from the judged
harder-set rows (hard_reach.py's judged dir); the 24 new samples are scored by the matcher unless a judged dir for them is given
with JUDGED32 (judge_hard.py run on the +24 texts), in which case score_j is used on both sides alike.
usage: [JUDGED32=dir] python hard32_analysis.py targets.json rows32_dir judged_hard_dir rows_hard_root out.json"""
import json, sys, os, glob, collections
TJ = json.load(open(sys.argv[1])); T = TJ["targets"]; NOTE = TJ.get("pairs", {}); RD, JD, HR, OUT = sys.argv[2:6]; J32 = os.environ.get("JUDGED32")
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def hard_counts(model, cap):
    """per problem at the cap on the harder set: (samples, finished, solved) under the judged scores"""
    fs = glob.glob(f"{HR}/**/{model}_cap{cap}.jsonl", recursive=True)
    jf = f"{JD}/{model}.jsonl"
    if not fs or not os.path.exists(jf): return None
    jm = {(r["id"], r["k"]): r["score_j"] for r in (json.loads(l) for l in open(jf) if l.strip())}
    c = collections.defaultdict(lambda: [0, 0, 0])
    for l in open(fs[0]):
        if not l.strip(): continue
        r = json.loads(l); s = jm[(r["id"], r["k"])] if r["finished"] else 0.0
        v = c[r["id"]]; v[0] += 1; v[1] += r["finished"]; v[2] += s >= 1 - 1e-9
    return c
def new_counts(model, at):
    """per problem: (samples, solved) in the 24 new samples; a sample longer than the pair's comparison cap counts as unfinished
    (the +24 samples were written at the model's harder-set generation cap, which exceeds the pair's cap for ProRL v2)"""
    f = f"{RD}/{model}.jsonl"
    if not os.path.exists(f): return None
    jm = None
    if J32 and os.path.exists(f"{J32}/{model}.jsonl"): jm = {(r["id"], r["k"]): r["score_j"] for r in (json.loads(l) for l in open(f"{J32}/{model}.jsonl") if l.strip())}
    c = collections.defaultdict(lambda: [0, 0])
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l); fin = r["finished"] and r.get("len", 0) <= at; s = (jm.get((r["id"], r["k"]), r["score"]) if jm is not None else r["score"]) if fin else 0.0
        v = c[r["id"]]; v[0] += 1; v[1] += s >= 1 - 1e-9
    return c
res = {}; pooled = collections.Counter()
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    rn, bn = rl.split("/")[-1], base.split("/")[-1]
    if rn not in T or bn not in T: continue
    at = NOTE.get(name, {}).get("at_cap", T[rn]["cap"]); B, R = hard_counts(bn, at), hard_counts(rn, at); NB, NR = new_counts(bn, at), new_counts(rn, at)
    if B is None or R is None: print(f"{name:13s} judged harder rows missing"); continue
    if NB is None or NR is None: print(f"{name:13s} +24 rows missing for {'base' if NB is None else 'RL'}"); continue
    ids = sorted(set(B) & set(R))
    U = [p for p in ids if B[p][1] == B[p][0] and B[p][2] == 0]; Up = [p for p in ids if R[p][1] == R[p][0] and R[p][2] == 0]
    rl_wins = [p for p in U if R[p][2] > 0]; base_wins = [p for p in Up if B[p][2] > 0]
    def repeat(wins, winner, loser):
        w = [p for p in wins if p in winner and p in loser]
        return {"n": len(wins), "read": len(w), "winner_ge1": sum(winner[p][1] >= 1 for p in w), "winner_ge4": sum(winner[p][1] >= 4 for p in w),
                "loser_ge1": sum(loser[p][1] >= 1 for p in w), "winner_only": sum(winner[p][1] >= 1 and loser[p][1] == 0 for p in w), "neither": sum(winner[p][1] == 0 and loser[p][1] == 0 for p in w)}
    o = {"cap": at, "U": len(U), "Uprime": len(Up), "rl_wins": repeat(rl_wins, NR, NB), "base_wins": repeat(base_wins, NB, NR), "scored": "judge" if J32 else "matcher"}
    res[name] = o
    for side in ("rl_wins", "base_wins"):
        for q in ("read", "winner_ge1", "winner_ge4", "loser_ge1", "winner_only", "neither"): pooled[f"{side}_{q}"] += o[side][q]
    f = lambda x: f"{x['read']:3d} read: winner again {x['winner_ge1']:3d} (>=4 {x['winner_ge4']:3d}), loser now {x['loser_ge1']:3d}, winner only {x['winner_only']:3d}, neither {x['neither']:3d}"
    print(f"{name:13s} {at // 1024:>3d}k | RL wins on U ({o['rl_wins']['n']}): {f(o['rl_wins'])}\n{'':17s} | base wins on U' ({o['base_wins']['n']}): {f(o['base_wins'])}")
print("pooled:", dict(pooled))
json.dump({"pairs": res, "pooled": dict(pooled)}, open(OUT, "w"), indent=1)
