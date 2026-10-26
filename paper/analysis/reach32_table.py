"""LaTeX rows for the reach-at-32 appendix table (Oct 21). The sets are the ones the reads were driven by (targets.json): for each
RL model, U = the physics problems its base finished eight of eight and never solved in the paper's read at the recipe's cap; for
each base, the union over its recipes of U' = the problems the RL model finished eight of eight and never solved, read once at the
largest of their caps. Each model wrote 24 new samples per problem. Rows: one per recipe (RL on U) and one per base (base on its
union U', listing the recipes it serves); pooled totals over finished reads. A model whose read is unfinished can be scored from
its partial file with --partials (needs the repo's matcher).
usage: python reach32_table.py targets.json rows32_dir out.json rows.tex [--partials]"""
import json, sys, os, collections
T = json.load(open(sys.argv[1])); T = T.get("targets", T); RD, OUT, TEX = sys.argv[2:5]; PART = "--partials" in sys.argv
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
SHORT = {"DeepSeek-R1-Distill-Qwen-1.5B": "R1-1.5B", "DeepSeek-R1-Distill-Qwen-7B": "R1-7B", "DeepSeek-R1-Distill-Qwen-14B": "R1-14B",
         "DeepSeek-R1-Distill-Qwen-32B": "R1-32B", "Qwen3-1.7B": "Qwen3-1.7B", "Qwen3-4B": "Qwen3-4B", "OpenMath-Nemotron-1.5B": "OpenMath-1.5B"}
def rows32(model):
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
                fin = c["finish_reason"] == "stop"; out.append({"id": r["id"], "k": 8 + j, "finished": fin, "score": compute_score("x", c["text"], it["gt"]) if fin else 0.0})
        return out, "partial"
    return None, "missing"
def side(model):
    ids = T[model]["ids"]; rows, st = rows32(model)
    if rows is None: return {"n": len(ids), "read": 0, "ge1": 0, "ge4": 0, "ge12": 0, "samples": 0, "solved": 0, "status": st}
    by = collections.defaultdict(int); have = set()
    for r in rows:
        if r["id"] in ids: have.add(r["id"]); by[r["id"]] += r["score"] >= 1 - 1e-9
    return {"n": len(ids), "read": len(have), "ge1": sum(by[p] >= 1 for p in have), "ge4": sum(by[p] >= 4 for p in have), "ge12": sum(by[p] >= 12 for p in have),
            "samples": 24 * len(have), "solved": sum(by.values()), "status": st}
def cell(x):
    if not x["read"]: return f"{x['n']} & -- & -- & --"
    tag = "" if x["status"] == "done" else f" ($\\mathit{{{x['read']}}}$ read)"
    return f"{x['n']}{tag} & {x['ge1']} ({x['ge1'] / x['read']:.2f}) & {x['ge4']} & {x['solved'] / x['samples']:.3f}"
res = {"recipes": {}, "bases": {}}; pooled = {"rl": collections.Counter(), "base": collections.Counter()}; rl_lines = []; base_lines = []
served = collections.defaultdict(list); base_cap = {}
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    rn, bn = rl.split("/")[-1], base.split("/")[-1]
    if rn not in T or bn not in T: continue
    R = side(rn); res["recipes"][name] = {"cap": cap, "base": SHORT.get(bn, bn), **R}; served[bn].append(name); base_cap[bn] = T[bn]["cap"]
    if R["status"] == "done":
        for q in ("n", "read", "ge1", "ge4", "ge12", "samples", "solved"): pooled["rl"][q] += R[q]
    rl_lines.append(f"{name} & {SHORT.get(bn, bn)} & {cap // 1024}k & {cell(R)} \\\\")
    print(f"{name:13s} {cap // 1024:>3d}k | RL on U:  {cell(R)}")
for bn, names in served.items():
    B = side(bn); res["bases"][SHORT.get(bn, bn)] = {"cap": base_cap[bn], "recipes": names, **B}
    if B["status"] == "done":
        for q in ("n", "read", "ge1", "ge4", "ge12", "samples", "solved"): pooled["base"][q] += B[q]
    base_lines.append(f"{SHORT.get(bn, bn)} & $U'$ of {len(names)} recipe{'s' if len(names) > 1 else ''} & {base_cap[bn] // 1024}k & {cell(B)} \\\\")
    print(f"{SHORT.get(bn, bn):13s} {base_cap[bn] // 1024:>3d}k | base on U': {cell(B)}  [{', '.join(names)}]")
pr, pb = pooled["rl"], pooled["base"]
def pooled_line(lab, p): return f"{lab} & & & {p['n']} & {p['ge1']} ({p['ge1'] / max(p['read'], 1):.2f}) & {p['ge4']} & {p['solved'] / max(p['samples'], 1):.3f} \\\\"
lines = rl_lines + ["\\midrule", pooled_line("all RL models", pr), "\\midrule"] + base_lines + ["\\midrule", pooled_line("all bases", pb)]
print("pooled RL:", dict(pr)); print("pooled base:", dict(pb))
open(TEX, "w").write("\n".join(lines) + "\n")
json.dump({**res, "pooled": {"rl": dict(pr), "base": dict(pb)}}, open(OUT, "w"), indent=1)
