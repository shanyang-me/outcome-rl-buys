"""Score the pre-registered predictions of the competition-math read (math/PREREG.md, P1-P5) from the analyses' JSON files.
P1 finishing: at each recipe's cap the base finishes less on the math set than on the paper's physics set (cap_cells.json, kind
phys, which at), for every pair; and for Open-RS3, ProRL v2, JustRL-D and e3 the finishing part is the larger part.
P2 reliability: among recipes whose base finishes at least 0.90 at the cap, P > 0 with the interval clear of zero in at least
three quarters (the registration wrote twelve of sixteen, assuming the physics finishing rates), and P > F in a majority.
P3 reach: net reach within [-0.02, +0.03] everywhere; its interval clear of zero for at most half the pairs; robust reach at most
0.02 everywhere; RL wins on U against base wins on U' between 0.5 and 2 where both exceed ten.
P4 contamination: the 2026-minus-2025 difference in net reach has an interval reaching zero or below for every recipe (math_p4.py).
P5 harness: our AIME 2025 pass@1 at the cap within 0.08 of the authors' number for every comparable model (p5_harness.py).
usage: python math_prereg.py math_reach.json cap_cells.json math_p4.json p5.json out.md [out.json]"""
import json, sys
HR, CC, P4, P5, OUT = sys.argv[1:6]; OUTJ = sys.argv[6] if len(sys.argv) > 6 else None
hr, cc, p4, p5 = (json.load(open(f)) for f in (HR, CC, P4, P5))
at = {}
for name, r in hr.items():
    L = f"{r['at_cap'] // 1024}k"; c = r["cells"].get(f"{L}|all")
    if c: at[name] = c
def clear(v): return v[1] > 0 or v[2] < 0
L = ["# Competition-math read: the pre-registered predictions", "", f"{len(at)} pairs at their caps.", ""]
res = {}
# P1
phys = {n: next((x["fin_b"] for x in cc.get(n, {}).get("cells", []) if x["which"] == "at" and x["kind"] == "phys"), None) for n in at}
below = {n: (at[n]["fin_b"][0] < phys[n]) for n in at if phys[n] is not None}
capbound = [n for n in ("Open-RS3", "ProRL v2", "JustRL-D", "e3") if n in at]
flarger = {n: abs(at[n]["F"][0]) > abs(at[n]["P"][0]) for n in capbound}
p1 = all(below.values()) and all(flarger.values())
L += ["## P1 finishing", "", "| recipe | base fin@1, physics | base fin@1, math | below | F | P | finishing larger |", "|---|---|---|---|---|---|---|"]
for n in at:
    L.append(f"| {n} | {phys[n]:.2f} | {at[n]['fin_b'][0]:.2f} | {'yes' if below.get(n) else 'NO'} | {at[n]['F'][0]:+.3f} | {at[n]['P'][0]:+.3f} | {('yes' if flarger[n] else 'NO') if n in flarger else ''} |" if phys[n] is not None else f"| {n} | -- | {at[n]['fin_b'][0]:.2f} | ? | | | |")
L += ["", f"- base finishes less on math than on physics: {sum(below.values())} of {len(below)}; finishing the larger part for the cap-bound recipes: {sum(flarger.values())} of {len(flarger)} ({', '.join(n for n in capbound if not flarger[n]) or 'none'} fail). **P1 {'holds' if p1 else 'FAILS'}.**", ""]
res["P1"] = dict(holds=p1, below=below, finishing_larger=flarger)
# P2
q = [n for n in at if at[n]["fin_b"][0] >= 0.90]
pos = {n: (at[n]["P"][0] > 0 and at[n]["P"][1] > 0) for n in q}; larger = {n: at[n]["P"][0] > at[n]["F"][0] for n in q}
k, m = sum(pos.values()), len(q); p2 = m > 0 and k >= 0.75 * m - 1e-9 and sum(larger.values()) > m / 2
L += ["## P2 reliability", "", f"Recipes whose base finishes at least 0.90 at the cap: {m} ({', '.join(q)}).", "",
      "| recipe | base fin@1 | P [CI] | P > 0, clear of zero | P > F |", "|---|---|---|---|---|"]
for n in q: L.append(f"| {n} | {at[n]['fin_b'][0]:.2f} | {at[n]['P'][0]:+.3f} [{at[n]['P'][1]:+.3f}, {at[n]['P'][2]:+.3f}] | {'yes' if pos[n] else 'NO'} | {'yes' if larger[n] else 'no'} |")
L += ["", f"- P > 0 with the interval clear of zero: {k} of {m} (the registration asked for twelve of sixteen, three quarters); reliability the larger part: {sum(larger.values())} of {m}. **P2 {'holds' if p2 else 'FAILS'}.**", ""]
res["P2"] = dict(holds=p2, qualifying=q, positive_clear=pos, larger=larger)
# P3
inr = {n: -0.02 <= at[n]["reach"][0] <= 0.03 for n in at}; cl = {n: clear(at[n]["reach"]) for n in at}; rob = {n: at[n]["robust"][0] <= 0.02 for n in at}
ratio = {n: (at[n]["rl_on_U"] / at[n]["base_on_Up"]) for n in at if at[n]["rl_on_U"] > 10 and at[n]["base_on_Up"] > 10}
rok = {n: 0.5 <= v <= 2 for n, v in ratio.items()}
p3 = all(inr.values()) and sum(cl.values()) <= len(at) / 2 and all(rob.values()) and all(rok.values())
L += ["## P3 reach", "", "| recipe | reach [CI] | in [-0.02, +0.03] | clear of zero | robust | RL wins on U / base wins on U' |", "|---|---|---|---|---|---|"]
for n in at:
    L.append(f"| {n} | {at[n]['reach'][0]:+.3f} [{at[n]['reach'][1]:+.3f}, {at[n]['reach'][2]:+.3f}] | {'yes' if inr[n] else 'NO'} | {'yes' if cl[n] else 'no'} | {at[n]['robust'][0]:+.3f} | {at[n]['rl_on_U']} / {at[n]['base_on_Up']}" + (f" = {ratio[n]:.2f}" if n in ratio else "") + " |")
L += ["", f"- within the band: {sum(inr.values())} of {len(at)}; interval clear of zero: {sum(cl.values())} of {len(at)} (at most half allowed); robust reach at most 0.02: {sum(rob.values())} of {len(at)}; "
      f"mirror ratio between 0.5 and 2 where both sides exceed ten: {sum(rok.values())} of {len(rok)}. **P3 {'holds' if p3 else 'FAILS'}.**", ""]
res["P3"] = dict(holds=p3, in_band=inr, clear=cl, robust_ok=rob, ratio=ratio)
# P4
p4ok = {n: v["p4"] for n, v in p4.items()}; p4h = len(p4ok) > 0 and all(p4ok.values())
L += ["## P4 contamination", "", "| recipe | reach 2025 | reach 2026 | 2026 - 2025 [CI] | holds |", "|---|---|---|---|---|"]
for n, v in p4.items(): L.append(f"| {n} | {v['reach_pre']:+.3f} | {v['reach_post']:+.3f} | {v['reach_diff'][0]:+.3f} [{v['reach_diff'][1]:+.3f}, {v['reach_diff'][2]:+.3f}] | {'yes' if v['p4'] else 'NO'} |")
L += ["", f"- the 2026 stratum shows no more reach than the 2025 one (interval reaching zero or below): {sum(p4ok.values())} of {len(p4ok)}. **P4 {'holds' if p4h else 'FAILS'}.**", ""]
res["P4"] = dict(holds=p4h, per_recipe=p4ok)
# P5
comp = {m: v for m, v in p5.get("rl", {}).items() if v.get("comparable")}
w = {m: v.get("within_0.08", abs(v["gap_at_cap"]) <= 0.08) for m, v in comp.items()}; p5h = len(w) > 0 and all(w.values())
L += ["## P5 harness", "", "| model | ours at cap | authors | gap | within 0.08 |", "|---|---|---|---|---|"]
for m, v in comp.items(): L.append(f"| {m} | {v['pass1_at_cap']:.3f} | {v['reported'] / 100:.3f} | {v['gap_at_cap']:+.3f} | {'yes' if w[m] else 'NO'} |")
nc = [(m, v) for m, v in p5.get("rl", {}).items() if v.get("reported") is not None and not v.get("comparable")]
L += ["", f"- comparable models within 0.08: {sum(w.values())} of {len(w)}. **P5 {'holds' if p5h else 'FAILS'}.**",
      "- not comparable (budget or temperature differ): " + "; ".join(f"{m} {v['gap_at_cap']:+.3f} (fin {v['fin_at_cap']:.2f} at {v['cap'] // 1024}k, authors at {(str(v['authors']['max_tokens'] // 1024) + 'k') if v['authors'].get('max_tokens') else 'an unstated cap'})" for m, v in nc), ""]
res["P5"] = dict(holds=p5h, within=w)
L += ["## Summary", "", "| prediction | result |", "|---|---|"] + [f"| {k} | {'holds' if v['holds'] else 'FAILS'} |" for k, v in res.items()]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-7:]))
if OUTJ: json.dump(res, open(OUTJ, "w"), indent=1)
