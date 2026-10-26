"""Targets for the harder set's +24 round (Oct 20). For every pair read on the harder set, at the recipe's cap, the one-sided
problems: in U (base finished 8 of 8 and never solved) those the RL model solved at least once, and in U' (RL model finished 8 of 8
and never solved) those the base solved at least once; judged scores where judged/<model>.jsonl exists, else the matcher's. Both
models of a pair then get 24 more samples on the pair's one-sided problems (seed 2, k = 8..31) at the model's harder-set generation
cap; a base's set is the union over its recipes. Output {model_name: {mid, cap, ids, pairs}} for pair_reach32.py (PAIR_TARGETS).
usage: python hard32_targets.py out.json rows_root hard_spec.json [judged_dir]"""
import json, glob, sys, os, collections
OUT, ROOT, SPEC = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])); JUDGED = sys.argv[4] if len(sys.argv) > 4 else None
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def caps_for(m): return sorted({int(os.path.basename(f).split("_cap")[1].split(".")[0]) for f in glob.glob(f"{ROOT}/{m}_cap*.jsonl")})
def counts(m, cap):
    jm = None
    if JUDGED and os.path.exists(f"{JUDGED}/{m}.jsonl"):
        jm = {(r["id"], r["k"]): r["score_j"] for r in (json.loads(l) for l in open(f"{JUDGED}/{m}.jsonl") if l.strip())}
    c = collections.defaultdict(lambda: [0, 0, 0]); miss = 0  # id -> [samples, finished, solved]
    for l in open(f"{ROOT}/{m}_cap{cap}.jsonl"):
        if not l.strip(): continue
        r = json.loads(l); s = r["score"] if r["finished"] else 0.0
        if jm is not None and r["finished"]:
            if (r["id"], r["k"]) not in jm: miss += 1
            s = jm.get((r["id"], r["k"]), s)
        v = c[r["id"]]; v[0] += 1; v[1] += r["finished"]; v[2] += s >= 1 - 1e-9
    if miss: raise SystemExit(f"{m}: the judged file lacks {miss} finished samples (stale or truncated); refusing to mix judged and matcher scores")
    return c, jm is not None
targets, note = collections.defaultdict(set), {}
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    rn, bn = rl.split("/")[-1], base.split("/")[-1]
    common = sorted(set(caps_for(rn)) & set(caps_for(bn)))
    if not common: print(f"{name:13s} rows missing"); continue
    at = min((c for c in common if c >= cap), default=max(common))
    B, jb = counts(bn, at); R, jr = counts(rn, at)
    if JUDGED and not (jb and jr):  # with a judged dir, both sides must be judged: never one judged side against a matcher side
        print(f"{name:13s} skipped: no judged rows for {' and '.join(m for m, j in ((rn, jr), (bn, jb)) if not j)}"); continue
    ids = sorted(set(B) & set(R)); U = [i for i in ids if B[i][1] == B[i][0] and B[i][2] == 0]; Up = [i for i in ids if R[i][1] == R[i][0] and R[i][2] == 0]
    rl_wins = [i for i in U if R[i][2] > 0]; base_wins = [i for i in Up if B[i][2] > 0]; S = set(rl_wins) | set(base_wins)
    targets[rn] |= S; targets[bn] |= S
    note[name] = {"at_cap": at, "n": len(ids), "U": len(U), "rl_wins_on_U": len(rl_wins), "Up": len(Up), "base_wins_on_Up": len(base_wins), "judged": jb and jr}
    print(f"{name:13s} at {at // 1024:>3d}k n={len(ids):3d} U={len(U):3d} RL>=1 {len(rl_wins):3d} | U'={len(Up):3d} base>=1 {len(base_wins):3d} | one-sided {len(S):3d} judged={jb and jr}")
# full model ids (org/name) come from the +24 chain's model list; PAIRS and the spec hold short names
chain = open(os.path.join(H, "..", "..", "..", "pod_scripts", "hard32_chain.sh")).read()
mids = {m.split("/")[-1]: m for m in chain.split('M="', 1)[1].split('"', 1)[0].split()}
for rl, base, *_ in PAIRS.values():
    for m in (rl, base): mids.setdefault(m.split("/")[-1], m)
missing = [m for m in targets if "/" not in mids.get(m, "")]
assert not missing, f"no org/name id for {missing}"
out = {m: {"mid": mids[m], "cap": SPEC[m][2], "ids": sorted(s), "n": len(s)} for m, s in targets.items() if s}
json.dump({"targets": out, "pairs": note}, open(OUT, "w"), indent=1)
print(f"{len(out)} models, {sum(v['n'] for v in out.values())} model-problem targets x 24 samples")
