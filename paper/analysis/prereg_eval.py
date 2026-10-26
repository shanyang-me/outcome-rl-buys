"""Score the pre-registered hypotheses (expansion/PREREGISTRATION.md) for the six and for all ten added recipes.
usage: python prereg_eval.py pairs_ci_merged.json stats_added.json stats_added10.json stats_all.json out.md"""
import json, sys, statistics as st
from scipy import stats
sys.path.insert(0, "/private/tmp/claude-501/-Users-shan-Documents/9c490dc1-2850-45bc-9e03-8519ac7516a7/scratchpad")
from families import ADDED, AMEND1
C = json.load(open(sys.argv[1])); S6, S10, SA = (json.load(open(f)) for f in sys.argv[2:5]); OUT = sys.argv[5]
cells = list(C.values())
def changing(r): return any(c["rl"] == r and c["cap"] in (4096, 8192) and (c["capped_gain"] or c["capped_loss"]) for c in cells)
def dfin8(r): return st.mean(c["finished"]["delta"] for c in cells if c["rl"] == r and c["cap"] == 8192)
L = ["# Pre-registered hypotheses: results", ""]
for tag, recs, S in (("six (as first registered)", ADDED, S6), ("all ten added (with amendment 1)", ADDED + AMEND1, S10)):
    dom = S["dominance_by_capset"]["4k+8k"]; per = dom["per_recipe"]; ci = dom["per_recipe_ci_problems"]
    ch = [r for r in recs if changing(r)]; k = sum(per[r] > 0.5 for r in ch); n = len(ch)
    p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
    L += [f"## {tag}", "", "| recipe | changes where the budget binds | binding-cap dominance [problem CI] | matched mean | Δfinished at 8k |", "|---|---|---|---|---|"]
    for r in recs:
        mm = S["per_recipe"][r]["matched_mean"]
        L.append(f"| {r} | {'yes' if changing(r) else 'no'} | {per[r]:.3f} [{ci[r][0]:.3f}, {ci[r][1]:.3f}] | {('%+.4f' % mm) if mm is not None else '--'} | {dfin8(r):+.3f} |")
    mms = [S["per_recipe"][r]["matched_mean"] for r in recs if S["per_recipe"][r]["matched_mean"] is not None]
    h1 = (k == n and n > 0 and p < 0.05); h2 = (st.mean(mms) < 0.03 and max(mms) <= 0.10)
    d16 = S["dominance_by_capset"]["16k"]
    L += ["", f"- **H1**: {k} of {n} changing recipes have dominance > 1/2; one-sided sign test p = {p:.4f}; pooled binding-cap dominance "
          f"{dom['pooled']:.3f} [{dom['ci_crossed'][0]:.3f}, {dom['ci_crossed'][1]:.3f}] (crossed bootstrap). **{'Supported' if h1 else 'NOT supported'}.**",
          f"- **H2**: mean matched change {st.mean(mms):+.4f} over {len(mms)} recipes, max {max(mms):+.4f}; random effects {S['re_meta_matched']['mu']:+.4f} "
          f"HKSJ [{S['re_meta_matched']['hksj_ci'][0]:+.4f}, {S['re_meta_matched']['hksj_ci'][1]:+.4f}]. **{'Supported' if h2 else 'NOT supported'}.**",
          f"- **H3** (reported, not predicted): 16k pooled dominance {d16['pooled']:.3f} [{d16['ci_crossed'][0]:.3f}, {d16['ci_crossed'][1]:.3f}]; above 1/2 in {d16['k_above_half_changing']} of {d16['n_changing']} recipes that change at 16k.", ""]
pred = {"JustRL-Nemotron-1.5B": +1, "e3-1.7B": +1, "Polaris-1.7B-Preview": -1, "JustRL-DeepSeek-1.5B": +1, "AReaL-1.5B-Preview-Stage-3": +1}
L += ["## Registered directions (exploratory)", "", "| recipe | predicted Δfinished at 8k | observed (mean of physics, MATH) | held |", "|---|---|---|---|"]
for r, s in pred.items():
    v = dfin8(r); L.append(f"| {r} | {'up' if s > 0 else 'down'} | {v:+.3f} | {'yes' if v * s > 0 else 'no'} |")
lv = SA.get("length_vs_finishing_matched", {})
L += ["", f"- Length-finishing association over all {SA['n_recipes']} recipes: Spearman {lv.get('spearman', float('nan')):+.3f}, p {lv.get('p_exact_two_sided', float('nan')):.4f} ({lv.get('p_method', '')}); predicted negative: **{'held' if lv.get('spearman', 0) < 0 else 'did not hold'}**."]
same = [("JustRL-DeepSeek-1.5B", "JustRL-Nemotron-1.5B"), ("DeepMath-1.5B", "DeepMath-Omn-1.5B")]
L += ["", "## Same recipe on two bases (no prediction)", ""]
for a, b in same:
    L.append(f"- {a.split('-')[0]}: Δfinished at 8k {dfin8(a):+.3f} on R1-Distill-1.5B, {dfin8(b):+.3f} on OpenMath-Nemotron-1.5B.")
# AReaL's chat template does not append "<think>\n" after the assistant tag (registered: report it with its results)
import glob as _g, os as _o
_rows = _g.glob(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "pairs_rows", "pair_decomp4", "AReaL-1.5B-Preview-Stage-3_cap16384.jsonl"))
if _rows:
    _n = _f = _c = 0
    for _l in open(_rows[0]):
        if _l.strip():
            _r = json.loads(_l); _n += 1; _f += bool(_r["finished"]); _c += bool(_r.get("think_closed"))
    L += ["", "## AReaL template (reported as registered)", "",
          f"- AReaL-1.5B-Preview-Stage-3 at 16k: {_c}/{_n} samples close a think block ({_c/_n:.3f}); finished {_f/_n:.3f}. "
          "Its template does not append <think>\\n after the assistant tag; the model opens the block itself or answers without one."]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
