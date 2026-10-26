"""Add one RL/base pair to extra_pairs.json (rows for the appendix budget figure, fig2_gen.py) from <model>_cap<cap>.jsonl rows:
per cap the finishing/reliability split with aggregate rates (as in the paper), the four-way solve overlap and the within-budget wins-alone counts, physics only.
usage: python pair_extra.py extra_pairs.json rows_dir RL_MODEL BASE_MODEL "Name" "base tag" "cap label" """
import json, glob, sys, collections, os
OUT, ROWS, RL, BASE, NAME, BTAG, CLAB = sys.argv[1:8]
def load(model, cap):
    fs = glob.glob(f"{ROWS}/**/{model}_cap{cap}.jsonl", recursive=True); assert fs, (model, cap)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip():
            r = json.loads(l)
            if r["kind"] == "phys": by[r["id"]].append(r)
    return by
ok = lambda r: r["score"] >= 1 - 1e-9
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
ent = {"base": BTAG, "cap": CLAB}
for cap in (4096, 8192, 16384):
    B, R = load(BASE, cap), load(RL, cap); keys = [k for k in B if k in R]; n = len(keys)
    rb = [r for k in keys for r in B[k]]; rr = [r for k in keys for r in R[k]]
    fb = sum(r["finished"] for r in rb) / len(rb); fr = sum(r["finished"] for r in rr) / len(rr)
    ab = sum(ok(r) for r in rb) / max(1, sum(r["finished"] for r in rb)); ar = sum(ok(r) for r in rr) / max(1, sum(r["finished"] for r in rr))
    c = collections.Counter(); rl_only = []; base_only = []
    for k in keys:
        b, r = any(ok(x) for x in B[k]), any(ok(x) for x in R[k])
        c["both" if b and r else "base_only" if b else "rl_only" if r else "neither"] += 1
        if r and not b: rl_only.append(k)
        if b and not r: base_only.append(k)
    ent[str(cap)] = {"F": (fr - fb) * ar, "P": fb * (ar - ab), "d": sum(ok(r) for r in rr) / len(rr) - sum(ok(r) for r in rb) / len(rb), "n": n,
        "both": c["both"], "base_only": c["base_only"], "rl_only": c["rl_only"], "neither": c["neither"],
        "rl_only_base_all_finished": sum(all(x["finished"] for x in B[k]) for k in rl_only), "base_only_rl_all_finished": sum(all(x["finished"] for x in R[k]) for k in base_only),
        "rl_only_base_some_truncated": sum(any(not x["finished"] for x in B[k]) for k in rl_only), "base_only_rl_some_truncated": sum(any(not x["finished"] for x in R[k]) for k in base_only)}
    print(NAME, cap, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ent[str(cap)].items() if k in ("F", "P", "d", "rl_only", "base_only", "rl_only_base_all_finished", "base_only_rl_all_finished")})
out[NAME] = ent; json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT, "pairs:", list(out))
