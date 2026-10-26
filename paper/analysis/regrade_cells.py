"""Regrade analysis (Oct 19). Reads the judged rows of regrade_pairs.py and (1) checks the open judge against the Claude verdicts of
the matcher audit, (2) recomputes, for every pair whose base and RL texts were regraded, the physics cell at each budget under the
matcher and under the judge: fin@1(B), pass@1(B|fin), the finishing term F and the reliability term P, and the finishing share
|F|/(|F|+|P|). usage: python regrade_cells.py REGRADE_DIR AUDIT_DIR OUT.json"""
import json, sys, glob, os, collections
import numpy as np
RD, AD, OUT = sys.argv[1:4]
rows = {}
for f in glob.glob(f"{RD}/*/*.jsonl"):
    tag, model = f.split("/")[-2], os.path.basename(f)[:-6]
    rows[(tag, model)] = [json.loads(l) for l in open(f) if l.strip()]
# (1) agreement with the audit's Claude verdicts (uid -> src "tag|model|cap", id, k)
items = json.load(open(f"{AD}/items_with_uid.json")); ver = {}
for l in open(f"{AD}/verdicts.jsonl"):
    try: o = json.loads(l); ver[o["uid"]] = o["verdict"]
    except Exception: pass
idx = {(t, m): {(r["id"], r["k"]): r for r in rs} for (t, m), rs in rows.items()}
agree = collections.Counter()
for it in items:
    t, m, _ = it["src"].split("|"); r = idx.get((t, m), {}).get((it["id"], it["k"]))
    if r is None or it["uid"] not in ver: continue
    c = ver[it["uid"]] == "right"; o = r["score_j"] >= 1 - 1e-9
    agree[(c, o)] += 1
n = sum(agree.values())
res = {"agreement": {"n": n, "both_right": agree[(True, True)], "both_not": agree[(False, False)], "claude_only": agree[(True, False)],
                     "open_only": agree[(False, True)], "rate": (agree[(True, True)] + agree[(False, False)]) / n if n else None}}
# (2) cells
PAIRS = [("pair_decomp4", "OpenMath-Nemotron-1.5B", r) for r in ("JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B")] + \
        [("pair_decomp4", "Qwen3-1.7B", r) for r in ("e3-1.7B", "Polaris-1.7B-Preview")] + \
        [("pair_decomp10_native", "Qwen3-1.7B", "Polaris-1.7B-Preview"), ("pair_decomp10_native", "Qwen3-4B", "Polaris-4B-Preview"),
         ("pair_decomp9_32b", "DeepSeek-R1-Distill-Qwen-32B", "Skywork-OR1-32B")]
CAPS = {"pair_decomp4": (4096, 8192, 16384), "pair_decomp10_native": (16384, 32768, 49152, 53248, 65536), "pair_decomp9_32b": (16384, 24576, 28672, 32768, 49152)}
def cell(rs, cap, key):
    ph = [r for r in rs if r["kind"] == "phys"]; fin = [r for r in ph if r["finished"] and r["len"] <= cap]
    f = len(fin) / len(ph); a = sum(r[key] for r in fin) / len(fin) if fin else 0.0
    return f, a
def reach(bs, rs, cap, key):
    # net reach at this budget: RL's full solves on problems the base finished eight times and never fully solved, minus the mirror
    by = lambda rows: {k: v for k, v in collections.defaultdict(list, {}).items()} or None
    def grp(rows):
        g = collections.defaultdict(list)
        for r in rows:
            if r["kind"] == "phys": g[r["id"]].append({**r, "fin": r["finished"] and r["len"] <= cap})
        return g
    B, R = grp(bs), grp(rs); ok = lambda r: r["fin"] and r[key] >= 1 - 1e-9; n = len(B)
    never = lambda M, k: all(r["fin"] for r in M[k]) and not any(ok(r) for r in M[k])
    U = [k for k in B if never(B, k)]; Up = [k for k in R if never(R, k)]
    return sum(ok(r) for k in U for r in R[k]) / (8 * n) - sum(ok(r) for k in Up for r in B[k]) / (8 * n), len(U), sum(any(ok(r) for r in R[k]) for k in U), len(Up), sum(any(ok(r) for r in B[k]) for k in Up)
res["cells"] = {}
for tag, b, r in PAIRS:
    if (tag, b) not in rows or (tag, r) not in rows: continue
    for cap in CAPS[tag]:
        o = {}
        for key, lab in (("score", "matcher"), ("score_j", "judge")):
            fb, ab = cell(rows[(tag, b)], cap, key); fr, ar = cell(rows[(tag, r)], cap, key)
            F = (fr - fb) * ar; P = fb * (ar - ab)
            net, nU, rlU, nUp, bUp = reach(rows[(tag, b)], rows[(tag, r)], cap, key)
            o[lab] = {"fin_base": fb, "fin_rl": fr, "acc_base": ab, "acc_rl": ar, "F": F, "P": P, "d": F + P,
                      "share": abs(F) / (abs(F) + abs(P)) if abs(F) + abs(P) > 0 else None,
                      "reach_net": net, "U": nU, "rl_on_U": rlU, "Uprime": nUp, "base_on_Uprime": bUp}
        res["cells"][f"{r} vs {b} @{cap // 1024}k ({tag})"] = o
json.dump(res, open(OUT, "w"), indent=1)
print("agreement", res["agreement"])
for k, o in res["cells"].items():
    m, j = o["matcher"], o["judge"]
    print(f"{k:70s} acc base {m['acc_base']:.3f}->{j['acc_base']:.3f} rl {m['acc_rl']:.3f}->{j['acc_rl']:.3f} | F {m['F']:+.3f}->{j['F']:+.3f} P {m['P']:+.3f}->{j['P']:+.3f} | share {m['share'] or 0:.2f}->{j['share'] or 0:.2f} | reach {m['reach_net']:+.4f}->{j['reach_net']:+.4f} (U {m['U']}->{j['U']}, RL wins {m['rl_on_U']}->{j['rl_on_U']}; U' {m['Uprime']}->{j['Uprime']}, base wins {m['base_on_Uprime']}->{j['base_on_Uprime']})")
