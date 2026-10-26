"""Split-sample catch-up test. Problems are selected on samples 0-3 at budget B (RL finishes more often and scores higher
than its base); the rates are then read on samples 4-7 at B and on the base's samples from separate 16k and 64k generations,
so selection does not inflate the comparison. Catch-up = (base@big - base@B) / (RL@B - base@B) on the selected problems,
weighted by the selection half's finishing change. usage: python finishing_catchup.py PAIRS_ROWS_DIR"""
import json, glob, collections, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__))); from families import selected
import numpy as np
PR = sys.argv[1]
BIG = {"DeepSeek-R1-Distill-Qwen-1.5B": "pair_decomp6_cap64", "DeepSeek-R1-Distill-Qwen-7B": "pair_decomp6_cap64",
       "OpenMath-Nemotron-1.5B": "pair_decomp6_cap64", "Qwen3-1.7B": "pair_decomp10_native", "Qwen3-4B": "pair_decomp10_native"}
def rd(f):
    by = collections.defaultdict(list)
    for l in open(f):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return by
def load(m, cap):
    fs = glob.glob(f"*/rows/{m}_cap{cap}.jsonl"); assert len(fs) == 1, (m, cap, fs); return rd(fs[0])
def rate(rs, f, half=None):
    rs = [x for x in rs if half is None or (x["k"] < 4) == (half == 0)]
    return np.mean([x[f] for x in rs]) if rs else np.nan
cells = []
for base, recs in selected().items():
    B16 = load(base, 16384); B64 = rd(f"{PR}/{BIG[base]}/{base}_cap65536.jsonl")
    for B in (4096, 8192):
        Bb = load(base, B)
        for rec in recs:
            Rb = load(rec, B); R16 = load(rec, 16384)
            for k in Bb:
                if not (k in Rb and k in B16 and k in R16): continue
                d_f = rate(Rb[k], "finished", 0) - rate(Bb[k], "finished", 0)
                if d_f > 0 and rate(Rb[k], "score", 0) > rate(Bb[k], "score", 0):
                    c = dict(rec=rec, B=B, kind=k[0], w=d_f, x=rate(Rb[k], "score", 1), z=rate(Bb[k], "score", 1),
                             y=rate(B16[k], "score"), y2=rate(R16[k], "score"), yf=rate(B16[k], "finished"))
                    if k in B64: c.update(y64=rate(B64[k], "score"), yf64=rate(B64[k], "finished"))
                    cells.append(c)
def summ(cs, big="y"):
    cs = [c for c in cs if big in c]; w = np.array([c["w"] for c in cs]); g = lambda f: float((w * np.array([c[f] for c in cs])).sum() / w.sum())
    o = dict(n=len(cs), rl_B=g("x"), base_B=g("z"), base_big=g(big), base_fin_big=g("yf" if big == "y" else "yf64"))
    o["catchup"] = (o["base_big"] - o["base_B"]) / (o["rl_B"] - o["base_B"]); return o
res = {}; rng = np.random.default_rng(0); recs = sorted({c["rec"] for c in cells})
for big, lab in (("y", "16k"), ("y64", "64k")):
    for B in (4096, 8192):
        for kind in ("phys", "math"):
            res[f"{B}|{kind}|base@{lab}"] = summ([c for c in cells if c["B"] == B and c["kind"] == kind], big)
        bs = [summ([c for r in rng.choice(recs, len(recs)) for c in cells if c["rec"] == r and c["B"] == B], big)["catchup"] for _ in range(2000)]
        o = summ([c for c in cells if c["B"] == B], big); o["ci_recipes"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res[f"{B}|both|base@{lab}"] = o
json.dump(res, open("expansion/results/finishing_catchup.json", "w"), indent=1)
for k, v in res.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else ([round(x, 2) for x in b] if isinstance(b, list) else b)) for a, b in v.items()})
