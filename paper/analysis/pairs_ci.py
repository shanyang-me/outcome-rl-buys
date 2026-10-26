"""Problem-clustered bootstrap CIs for the public base->RL finishing decomposition (n=8 samples per problem, T=0.6).
Per cell (RL model, base, cap, set): paired-by-problem deltas of capped mean score, finishing rate, accuracy given finished
(sum of scores over finished samples / finished count), any-solve@8; 4000 problem resamples, percentile 95% CI.
usage: python pairs_ci.py rows_dir out.json out.md"""
import json, glob, sys, random, collections
from families import selected
FAMILIES = selected()
CAPS = (4096, 8192, 16384); KINDS = ("phys", "math"); B = 20000
D = sys.argv[1]
def load(model, cap):
    fs = glob.glob(f"{D}/*/{model}_cap{cap}.jsonl")
    if not fs: return None
    assert len(fs) == 1, (model, cap, fs)
    by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    # per problem: n, n_fin, sum_score, sum_score_fin, any_full
    return {k: (len(v), sum(r["finished"] for r in v), sum(r["score"] for r in v), sum(r["score"] for r in v if r["finished"]),
                1.0 if any(r["score"] >= 1 - 1e-9 for r in v) else 0.0, sum(r["len"] for r in v)) for k, v in by.items()}
def stats(S, ids):
    n = sum(S[i][0] for i in ids); nf = sum(S[i][1] for i in ids)
    return {"mean": sum(S[i][2] for i in ids) / n, "finished": nf / n, "sgf": (sum(S[i][3] for i in ids) / nf) if nf else 0.0,
            "any8": sum(S[i][4] for i in ids) / len(ids), "len": sum(S[i][5] for i in ids) / n}
rng = random.Random(0); out = {}; md = ["| RL model | base | set | cap | capped score base→RL (Δ, 95% CI) | finished (Δ, CI) | accuracy given finished (Δ, CI) | any-solve@8 (Δ) | tokens ratio | acc|finished on problems both finish ≥7/8 (n; Δ, CI) |", "|---|---|---|---|---|---|---|---|---|---|"]
for base, descs in FAMILIES.items():
    for cap in CAPS:
        Sb = load(base, cap)
        if not Sb: continue
        for d in descs:
            Sr = load(d, cap)
            if not Sr: continue
            for kind in KINDS:
                ids = sorted(k for k in Sb if k[0] == kind and k in Sr)
                if len(ids) < 20: continue
                pb, pr = stats(Sb, ids), stats(Sr, ids)
                boots = collections.defaultdict(list)
                for _ in range(B):
                    s = rng.choices(ids, k=len(ids)); b, r = stats(Sb, s), stats(Sr, s)
                    for q in ("mean", "finished", "sgf", "any8"): boots[q].append(r[q] - b[q])
                    if abs(r["mean"] - b["mean"]) > 1e-9: boots["share"].append((r["finished"] - b["finished"]) * r["sgf"] / (r["mean"] - b["mean"]))
                cell = {"base": base, "rl": d, "cap": cap, "set": kind, "n_problems": len(ids), "len_ratio": pr["len"] / pb["len"]}
                for q in ("mean", "finished", "sgf", "any8"):
                    bs = sorted(boots[q]); cell[q] = {"base": pb[q], "rl": pr[q], "delta": pr[q] - pb[q], "ci": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]}
                lo = lambda q: cell[q]["ci"][0]; hi = lambda q: cell[q]["ci"][1]
                cell["capped_gain"] = round(lo("mean"), 3) > 0; cell["capped_loss"] = round(hi("mean"), 3) < 0  # printed-interval rule
                cell["finishing_gain"] = round(lo("finished"), 2) > 0; cell["finishing_loss"] = round(hi("finished"), 2) < 0
                cell["perfinished_gain"] = round(lo("sgf"), 3) > 0; cell["perfinished_loss"] = round(hi("sgf"), 3) < 0
                both = [i for i in ids if Sb[i][1] >= 7 and Sr[i][1] >= 7]
                if len(both) >= 15:
                    sgf = lambda S, I: sum(S[i][3] for i in I) / sum(S[i][1] for i in I)
                    pt = sgf(Sr, both) - sgf(Sb, both); bs = sorted(sgf(Sr, s) - sgf(Sb, s) for s in (rng.choices(both, k=len(both)) for _ in range(B)))
                    cell["matched"] = {"n": len(both), "base": sgf(Sb, both), "rl": sgf(Sr, both), "delta": pt, "ci": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]}
                    cell["matched"]["gain"] = round(cell["matched"]["ci"][0], 3) > 0; cell["matched"]["loss"] = round(cell["matched"]["ci"][1], 3) < 0
                else: cell["matched"] = {"n": len(both)}
                # exact decomposition of the capped change (unfinished samples score 0): dmean = dfin*sgf_rl + fin_base*dsgf
                cell["finishing_term"] = cell["finished"]["delta"] * cell["sgf"]["rl"]; cell["perfinished_term"] = cell["finished"]["base"] * cell["sgf"]["delta"]
                cell["finishing_share"] = cell["finishing_term"] / cell["mean"]["delta"] if abs(cell["mean"]["delta"]) > 1e-9 else None
                cell["finishing_share_alt"] = (cell["finished"]["delta"] * cell["sgf"]["base"]) / cell["mean"]["delta"] if abs(cell["mean"]["delta"]) > 1e-9 else None
                bs = sorted(boots["share"]); cell["finishing_share_ci"] = [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if len(bs) > 100 else None
                out[f"{d} cap{cap} {kind}"] = cell
                f = lambda q, p=3: f"{cell[q]['base']:.{p}f}→{cell[q]['rl']:.{p}f} ({cell[q]['delta']:+.{p}f} [{cell[q]['ci'][0]:+.{p}f},{cell[q]['ci'][1]:+.{p}f}])"
                md.append(f"| {d} | {base} | {kind} | {cap//1024}k | {f('mean')} | {f('finished', 2)} | {f('sgf')} | {cell['any8']['delta']:+.3f} | {cell['len_ratio']:.2f}× | {('n=%d; %.3f→%.3f (%+.3f [%+.3f,%+.3f])' % (cell['matched']['n'], cell['matched']['base'], cell['matched']['rl'], cell['matched']['delta'], cell['matched']['ci'][0], cell['matched']['ci'][1])) if 'delta' in cell['matched'] else 'n=%d' % cell['matched']['n']} |")
json.dump(out, open(sys.argv[2], "w"), indent=1); open(sys.argv[3], "w").write("\n".join(md) + "\n")
c = collections.Counter(); c["cells"] = len(out)
for cell in out.values():
    if cell["capped_gain"] and not cell["perfinished_gain"]: c["capped gain, per-finished gain not established (CI covers 0 or below)"] += 1
    if cell["capped_gain"] and cell["perfinished_loss"]: c["  of which per-finished LOSS (CI below 0)"] += 1
    if cell["capped_gain"] and cell["perfinished_gain"]: c["capped gain AND per-finished gain"] += 1
    if not cell["capped_gain"] and cell["perfinished_gain"]: c["per-finished gain with no established capped gain"] += 1
    if cell["capped_loss"]: c["capped LOSS"] += 1
    if cell["capped_loss"] and cell["perfinished_gain"]: c["  capped loss with per-finished gain"] += 1
    if not cell["capped_gain"] and not cell["capped_loss"] and not cell["perfinished_gain"] and not cell["perfinished_loss"]: c["null on both"] += 1
    if cell["finishing_gain"]: c["finishing gain"] += 1
    if cell["finishing_loss"]: c["finishing loss"] += 1
for cell in out.values():
    m = cell["matched"]
    if "delta" in m:
        c["matched cells (n>=15 problems both finish >=7/8)"] += 1
        if round(m["ci"][0], 3) > 0: c["  matched per-finished gain"] += 1
        if round(m["ci"][1], 3) < 0: c["  matched per-finished loss"] += 1
gain_cells = [x for x in out.values() if x["capped_gain"]]
import statistics
shares = sorted(x["finishing_term"] / x["mean"]["delta"] for x in gain_cells)
c["capped-gain cells where the per-finished term exceeds the finishing term"] = sum(1 for x in gain_cells if x["perfinished_term"] > x["finishing_term"])
for k, v in c.items(): print(f"{v:3d}  {k}")
print(f"finishing term as a share of the capped gain over the {len(gain_cells)} capped-gain cells: median {statistics.median(shares):.3f}, min {shares[0]:.2f}, max {shares[-1]:.2f}; >=0.5 in {sum(s>=0.5 for s in shares)}, >=0.8 in {sum(s>=0.8 for s in shares)}, >=1 in {sum(s>=1 for s in shares)}; share CI includes 0.5 in {sum(1 for x in gain_cells if x['finishing_share_ci'] and x['finishing_share_ci'][0] < 0.5)}; alt-split share < 0.5 in {sum(1 for x in gain_cells if x['finishing_share_alt'] < 0.5)}")
for x in gain_cells: print(f"   {x['rl']:40s} {x['set']} {x['cap']//1024:2d}k  dmean {x['mean']['delta']:+.3f} = fin {x['finishing_term']:+.3f} + pf {x['perfinished_term']:+.3f}  share {x['finishing_term']/x['mean']['delta']:.2f}")
# per RL model across its cells
print("\nper model: cells / capped gain / finishing gain / per-finished gain / per-finished loss")
bym = collections.defaultdict(lambda: collections.Counter())
for cell in out.values():
    m = bym[cell["rl"]]; m["cells"] += 1; m["capped"] += cell["capped_gain"]; m["fin"] += cell["finishing_gain"]; m["pf+"] += cell["perfinished_gain"]; m["pf-"] += cell["perfinished_loss"]
for m, v in bym.items(): print(f"  {m:40s} {v['cells']} / {v['capped']} / {v['fin']} / {v['pf+']} / {v['pf-']}")
