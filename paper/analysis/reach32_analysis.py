"""Reach at k=32 on the paper's physics problems (Oct 20): every RL model wrote 24 more samples at its cap on the problems its
base never solved in eight (U), and every base 24 more on the problems its RL model never solved (U'); the sets come from the
targets file that drove the reads. For each recipe this reports, per side, how many of its one-sided problems the model solves at
least once and at least four times in the 24 new samples, and the per-sample solve rate on them, so the two sides can be compared
like for like: if the mirror is sampling luck the base recovers its share of U' at about the rate the RL model recovers its share of
U. A model whose read is unfinished can be scored from its partial file with --partials (needs the repo's matcher).
usage: python reach32_analysis.py targets.json rows_dir out.json [--partials]"""
import json, sys, os, glob, collections
T = json.load(open(sys.argv[1])); T = T.get("targets", T); RD, OUT = sys.argv[2], sys.argv[3]; PART = "--partials" in sys.argv
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
def rows_for(model):
    f = f"{RD}/{model}.jsonl"
    if os.path.exists(f): return [json.loads(l) for l in open(f) if l.strip()], "done"
    p = f"{RD}/partial_{model}.jsonl"
    if PART and os.path.exists(p):
        sys.path.insert(0, os.environ.get("PAIR_REPO", "/workspace/repo")); from rl.w3.outcome_reward import compute_score
        items = {it["id"]: it for it in (json.loads(l) for l in open(os.environ["PAIR_ITEMS"]) if l.strip())}
        out = []
        for l in open(p):
            if not l.strip(): continue
            r = json.loads(l); it = items[r["id"]]
            for j, c in enumerate(r["samples"]):
                fin = c["finish_reason"] == "stop"; out.append({"id": r["id"], "k": 8 + j, "finished": fin, "score": compute_score("x", c["text"], it["gt"]) if fin else 0.0, "len": c["len"]})
        return out, f"partial ({len(out) // 24} problems)"
    return None, "missing"
def side(model, ids):
    rows, st = rows_for(model)
    if rows is None: return None, st
    by = collections.defaultdict(list)
    for r in rows: by[r["id"]].append(r["score"] >= 1 - 1e-9)
    have = [i for i in ids if i in by]; sol = {i: sum(by[i]) for i in have}
    return {"targets": len(ids), "read": len(have), "ge1": sum(v >= 1 for v in sol.values()), "ge4": sum(v >= 4 for v in sol.values()), "ge12": sum(v >= 12 for v in sol.values()),
            "sample_rate": sum(sol.values()) / (24 * len(have)) if have else None, "status": st}, st
res = {}
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    rn, bn = rl.split("/")[-1], base.split("/")[-1]
    if rn not in T or bn not in T: continue
    R, sr = side(rn, T[rn]["ids"]); B, sb = side(bn, T[bn]["ids"])
    if R is None and B is None: continue
    res[name] = {"cap": T[rn]["cap"], "rl_on_U": R, "base_on_Uprime": B}
    f = lambda x: f"{x['ge1']:3d}/{x['read']:3d} >=1 ({x['ge1'] / max(x['read'], 1):.2f}), {x['ge4']:3d} >=4, rate {x['sample_rate']:.3f} [{x['status']}]" if x else "missing"
    print(f"{name:13s} {T[rn]['cap'] // 1024:>3d}k | RL on U:    {f(R)}\n{'':17s} | base on U': {f(B)}")
tot = {"rl": collections.Counter(), "base": collections.Counter()}
for v in res.values():
    for s, k in (("rl", "rl_on_U"), ("base", "base_on_Uprime")):
        x = v[k]
        if x and x["status"] == "done":
            for q in ("targets", "read", "ge1", "ge4", "ge12"): tot[s][q] += x[q]
print("pooled over finished sides:", {s: dict(c) for s, c in tot.items()})
json.dump({"pairs": res, "pooled": {s: dict(c) for s, c in tot.items()}}, open(OUT, "w"), indent=1)
