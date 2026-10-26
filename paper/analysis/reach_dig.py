"""Which recipes expand reach, and how robustly. For each recipe at its training cap, one budget beyond, and the fixed 16k budget:
N = problems the base finished all eight times and solved none; count how many of N the RL model solves in >=1, >=2, >=4, 8 of 8 samples,
and the mirror (base on the RL model's never-solved set). Then recurrence: which robust-reach problems (>=4/8) are shared across recipes of a base.
usage: python reach_dig.py out.json cap_split.json rows_root..."""
import json, glob, sys, collections, os
OUT, CAPJ, ROOTS = sys.argv[1], sys.argv[2], sys.argv[3:]
H=os.path.dirname(os.path.abspath(__file__))
src=open(os.path.join(H,"cap_split.py")).read(); i=src.index("PAIRS = {"); d=0
for j in range(i,len(src)):
    d+=(src[j]=="{")-(src[j]=="}")
    if d==0 and src[j]=="}": break
exec(src[i:j+1])
def load(model,cap,roots):
    fs=sorted(f for r in roots for f in glob.glob(f"{r}/**/{model}_cap{cap}.jsonl",recursive=True)); assert fs,(model,cap)
    by=collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip():
            r=json.loads(l); by[(r["kind"],r["id"])].append(r)
    return by
ok=lambda r: r["score"]>=1-1e-9
def cell(B,R):
    keys=[k for k in B if k in R]
    N=[k for k in keys if all(r["finished"] for r in B[k]) and not any(ok(r) for r in B[k])]
    Nr=[k for k in keys if all(r["finished"] for r in R[k]) and not any(ok(r) for r in R[k])]
    hist=lambda M,ks: {t: sum(sum(ok(r) for r in M[k])>=t for k in ks) for t in (1,2,4,8)}
    return {"n":len(keys),"n_never_base":len(N),"n_never_rl":len(Nr),"rl_on_N":hist(R,N),"base_on_Nr":hist(B,Nr),
            "robust_ids":[f"{k[0]}:{k[1]}" for k in N if sum(ok(r) for r in R[k])>=4],"any_ids":[f"{k[0]}:{k[1]}" for k in N if any(ok(r) for r in R[k])],
            "mirror_robust_ids":[f"{k[0]}:{k[1]}" for k in Nr if sum(ok(r) for r in B[k])>=4]}
cs=json.load(open(CAPJ))["recipes"]; out={}
R64=[r for r in ROOTS if "cap64" in r or "pair_decomp5" in r]; R16=[r for r in ROOTS if "cap64" not in r and "pair_decomp5" not in r]
print(f"{'recipe':14s} {'base':9s} {'budget':14s} {'N':>4s} {'>=1':>4s} {'>=2':>4s} {'>=4':>4s} {'8/8':>4s} | {'Nr':>4s} {'m>=1':>5s} {'m>=4':>5s}")
for rec,c in cs.items():
    rl,base=PAIRS[rec][0],PAIRS[rec][1]; out[rec]={"base":PAIRS[rec][3]}
    reads=[("16k",16384,R16)]
    if not c.get("below_only"): reads+= [("at "+c["at_label"],c["at_cap"],R64)] + ([("beyond "+c["beyond_label"],c["beyond_cap"],R64)] if c.get("beyond_cap") else [])
    for name,cap,roots in reads:
        try: B,Rm=load(base,cap,roots),load(rl,cap,roots)
        except AssertionError: continue
        x=cell(B,Rm); out[rec][name]=x
        print(f"{rec:14s} {PAIRS[rec][3]:9s} {name:14s} {x['n_never_base']:4d} {x['rl_on_N'][1]:4d} {x['rl_on_N'][2]:4d} {x['rl_on_N'][4]:4d} {x['rl_on_N'][8]:4d} | {x['n_never_rl']:4d} {x['base_on_Nr'][1]:5d} {x['base_on_Nr'][4]:5d}")
print("\n--- robust reach (>=4/8 on base-never-solved, base finished 8/8), by base, at 16k and at the training cap: problem -> recipes")
for which in ("16k","at"):
    print("==", which)
    bybase=collections.defaultdict(lambda: collections.defaultdict(list))
    for rec,v in out.items():
        key=next((k for k in v if k.startswith(which)),None)
        if not key: continue
        for pid in v[key]["robust_ids"]: bybase[v["base"]][pid].append(rec)
    for base,m in bybase.items():
        shared={p:r for p,r in m.items() if len(r)>1}
        print(f"  base {base}: {sum(len(r) for r in m.values())} robust cells over {len(m)} problems; shared by >1 recipe: {len(shared)}")
        for p,r in sorted(shared.items(), key=lambda kv:-len(kv[1]))[:8]: print("     ", p, r)
json.dump(out, open(OUT,"w"), indent=1); print("wrote", OUT)
