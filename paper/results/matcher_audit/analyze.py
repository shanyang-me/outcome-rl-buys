"""Matcher audit: from judged verdicts on 200 matcher-rejected finished physics samples per model, the share that are actually right,
each model's corrected accuracy given finished, and each pair's finishing (F) and reliability (P) terms before and after correction.
A 'right' sample is rescored to 1; 'partial' keeps the matcher's partial credit; intervals bootstrap the 200 sampled items."""
import json, sys, random, statistics as st, collections
D = sys.argv[1]
items = json.load(open(f"{D}/items_with_uid.json")); counts = json.load(open(f"{D}/counts.json"))
ver = {}
for l in open(f"{D}/verdicts.jsonl"):
    try: o = json.loads(l); ver[o["uid"]] = o["verdict"]
    except Exception: pass
by = collections.defaultdict(list)
for it in items: by[it["src"]].append(it)
def model_stats(src, sample=None):
    its = sample if sample is not None else by[src]; c = counts[src]
    jud = [it for it in its if it["uid"] in ver]
    inc = [(1 - it["score"]) if ver[it["uid"]] == "right" else 0.0 for it in jud]
    right = [ver[it["uid"]] == "right" for it in jud]
    add = c["wrong"] * (st.mean(inc) if inc else 0)
    return {"judged": len(jud), "right_share": st.mean(right) if right else 0, "fin_rate": c["fin"] / c["n"],
            "acc": c["score_sum"] / c["fin"], "acc_corr": (c["score_sum"] + add) / c["fin"]}
res = {src: model_stats(src) for src in by}
rng = random.Random(0)
for src in by:
    bs = sorted(model_stats(src, [rng.choice(by[src]) for _ in by[src]])["right_share"] for _ in range(2000))
    res[src]["right_share_ci"] = [bs[50], bs[1949]]
print(f"{'model @ budget':44s} judged  right share [95%]      acc|fin -> corrected")
for src, r in res.items():
    print(f"{src.split('|',1)[1]:44s} {r['judged']:4d}   {r['right_share']:.2f} [{r['right_share_ci'][0]:.2f}, {r['right_share_ci'][1]:.2f}]   {r['acc']:.3f} -> {r['acc_corr']:.3f}")
PAIRS = [("JustRL-N", "pair_decomp4|OpenMath-Nemotron-1.5B|16384", "pair_decomp4|JustRL-Nemotron-1.5B|16384"),
         ("KnowRL", "pair_decomp4|OpenMath-Nemotron-1.5B|16384", "pair_decomp4|KnowRL-Nemotron-1.5B|16384"),
         ("QuestA", "pair_decomp4|OpenMath-Nemotron-1.5B|16384", "pair_decomp4|QuestA-Nemotron-1.5B|16384"),
         ("DeepMath-Omn", "pair_decomp4|OpenMath-Nemotron-1.5B|16384", "pair_decomp4|DeepMath-Omn-1.5B|16384"),
         ("e3", "pair_decomp4|Qwen3-1.7B|16384", "pair_decomp4|e3-1.7B|16384"),
         ("Polaris-1.7B @16k", "pair_decomp4|Qwen3-1.7B|16384", "pair_decomp4|Polaris-1.7B-Preview|16384"),
         ("Polaris-1.7B @48k cap", "pair_decomp10_native|Qwen3-1.7B|49152", "pair_decomp10_native|Polaris-1.7B-Preview|49152"),
         ("Polaris-4B @52k cap", "pair_decomp10_native|Qwen3-4B|53248", "pair_decomp10_native|Polaris-4B-Preview|53248"),
         ("Skywork-32B @32k cap", "pair_decomp9_32b|DeepSeek-R1-Distill-Qwen-32B|32768", "pair_decomp9_32b|Skywork-OR1-32B|32768")]
out = {"models": res, "pairs": {}}
print(f"\n{'pair':24s}  F -> F'          P -> P'          finishing share -> corrected")
for name, b, r in PAIRS:
    B, R = res[b], res[r]
    F = (R["fin_rate"] - B["fin_rate"]) * R["acc"]; P = B["fin_rate"] * (R["acc"] - B["acc"])
    Fc = (R["fin_rate"] - B["fin_rate"]) * R["acc_corr"]; Pc = B["fin_rate"] * (R["acc_corr"] - B["acc_corr"])
    sh = abs(F) / (abs(F) + abs(P)); shc = abs(Fc) / (abs(Fc) + abs(Pc))
    out["pairs"][name] = {"F": F, "P": P, "F_corr": Fc, "P_corr": Pc, "share": sh, "share_corr": shc}
    print(f"{name:24s} {F:+.3f} -> {Fc:+.3f}   {P:+.3f} -> {Pc:+.3f}   {sh:.2f} -> {shc:.2f}")
json.dump(out, open(f"{D}/audit_results.json", "w"), indent=1)
