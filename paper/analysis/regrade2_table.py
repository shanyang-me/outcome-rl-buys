"""Judge regrade of every released pair on the regenerated read (Oct 20). Reads the judged rows of regrade_pairs.py for the
pair_regen64 texts (one 64k generation per model at 0.6, read at any cap by truncation: a sample is finished at a cap when it
stopped within it) and, for every recipe in cap_split.PAIRS, at 16k and at its training cap, gives the physics cell under the
frozen matcher and under the judge: finishing, pass@1 per finished derivation, F and P (the two-part split of the whole change,
as in tab:regrade), the finishing share |F|/(|F|+|P|) and net reach@8 with the counts behind it. Writes a JSON and the LaTeX rows.
usage: python regrade2_table.py REGRADE2_DIR/pair_regen64 out.json rows.tex"""
import json, glob, sys, os, collections
RD, OUT, TEX = sys.argv[1:4]
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
rows = {}
for f in glob.glob(f"{RD}/*.jsonl"):
    rows[os.path.basename(f)[:-6]] = [json.loads(l) for l in open(f) if l.strip()]
def cell(rs, cap, key):
    ph = [r for r in rs if r["kind"] == "phys"]; fin = [r for r in ph if r["finished"] and r["len"] <= cap]
    f = len(fin) / len(ph); a = sum(r[key] for r in fin) / len(fin) if fin else 0.0
    return f, a
def reach(bs, rs, cap, key):
    def grp(rr):
        g = collections.defaultdict(list)
        for r in rr:
            if r["kind"] == "phys": g[r["id"]].append({**r, "fin": r["finished"] and r["len"] <= cap})
        return g
    B, R = grp(bs), grp(rs); ok = lambda r: r["fin"] and r[key] >= 1 - 1e-9; n = len(B)
    never = lambda M, k: all(r["fin"] for r in M[k]) and not any(ok(r) for r in M[k])
    U = [k for k in B if k in R and never(B, k)]; Up = [k for k in R if k in B and never(R, k)]
    return (sum(ok(r) for k in U for r in R[k]) / (8 * n) - sum(ok(r) for k in Up for r in B[k]) / (8 * n),
            len(U), sum(any(ok(r) for r in R[k]) for k in U), len(Up), sum(any(ok(r) for r in B[k]) for k in Up))
res = {}; lines = []
def s3(x): return f"{x:+.3f}"
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    b, r = base.split("/")[-1], rl.split("/")[-1]
    if b not in rows or r not in rows: print(f"{name:14s} missing {'base' if b not in rows else 'RL'} rows"); continue
    for bgt in sorted({16384, cap}):
        o = {}
        for key, klab in (("score", "matcher"), ("score_j", "judge")):
            fb, ab = cell(rows[b], bgt, key); fr, ar = cell(rows[r], bgt, key)
            F = (fr - fb) * ar; P = fb * (ar - ab); net, nU, rlU, nUp, bUp = reach(rows[b], rows[r], bgt, key)
            o[klab] = {"fin_base": fb, "fin_rl": fr, "acc_base": ab, "acc_rl": ar, "F": F, "P": P, "d": F + P,
                       "share": abs(F) / (abs(F) + abs(P)) if abs(F) + abs(P) > 0 else None, "reach_net": net, "U": nU, "rl_on_U": rlU, "Uprime": nUp, "base_on_Uprime": bUp}
        res[f"{name} @{bgt // 1024}k"] = o; m, jd = o["matcher"], o["judge"]
        sh = lambda x: f"{x['share']:.2f}" if x["share"] is not None else "--"
        lines.append(f"{name} & {bgt // 1024}k{' (cap)' if bgt == cap else ''} & ${m['acc_base']:.2f}\\to{jd['acc_base']:.2f}$ & ${m['acc_rl']:.2f}\\to{jd['acc_rl']:.2f}$ & "
                     f"${s3(m['F'])}\\to{s3(jd['F'])}$ & ${s3(m['P'])}\\to{s3(jd['P'])}$ & ${sh(m)}\\to{sh(jd)}$ & ${s3(m['reach_net'])}\\to{s3(jd['reach_net'])}$ \\\\")
        print(f"{name:14s} {bgt // 1024:>3d}k fin {m['fin_base']:.2f}->{m['fin_rl']:.2f} acc m {m['acc_base']:.3f}->{m['acc_rl']:.3f} j {jd['acc_base']:.3f}->{jd['acc_rl']:.3f} "
              f"F {s3(m['F'])}->{s3(jd['F'])} P {s3(m['P'])}->{s3(jd['P'])} reach {s3(m['reach_net'])}->{s3(jd['reach_net'])} | U {m['U']}->{jd['U']} RL>=1 {m['rl_on_U']}->{jd['rl_on_U']}")
json.dump(res, open(OUT, "w"), indent=1); open(TEX, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} rows -> {TEX}")
