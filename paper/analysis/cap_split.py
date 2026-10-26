"""At-cap versus beyond-cap split for the released pairs, from the 64k-cap read (and the 36k read for the Qwen3 family).
For each recipe: the derived budget nearest at or above its final training cap ('at') and the next budget up ('beyond');
at each, the finishing term F=(fr-fb)*ar, the reliability term P=fb*(ar-ab), the change d, and the within-budget wins alone
(RL-only cells where the base finished all eight and was wrong, and the mirror). Pooled: mean F, P, d over the recipes read.
usage: python cap_split.py out.json rows_dir [rows_dir ...]"""
import json, glob, sys, os, collections, statistics as st
OUT, ROOTS = sys.argv[1], sys.argv[2:]
PAIRS = {"ProRL v2": ("Nemotron-Research-Reasoning-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B", 8192, "R1-1.5B", "8k"),
         "JustRL-D": ("JustRL-DeepSeek-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B", 16384, "R1-1.5B", "15k"),
         "DeepScaleR": ("DeepScaleR-1.5B-Preview", "DeepSeek-R1-Distill-Qwen-1.5B", 24576, "R1-1.5B", "8$\\to$24k"),
         "DeepMath": ("DeepMath-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B", 24576, "R1-1.5B", "24k$^\\dagger$"),
         "RLinf": ("RLinf-math-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B", 28672, "R1-1.5B", "27.6k"),
         "STILL-3": ("STILL-3-1.5B-preview", "DeepSeek-R1-Distill-Qwen-1.5B", 28672, "R1-1.5B", "29k$^\\dagger$"),
         "AReaL": ("AReaL-1.5B-Preview-Stage-3", "DeepSeek-R1-Distill-Qwen-1.5B", 24576, "R1-1.5B", "8$\\to$24k"),
         "Open-RS3": ("Open-RS3", "DeepSeek-R1-Distill-Qwen-1.5B", 4096, "R1-1.5B", "3.5k"),
         "JustRL-N": ("JustRL-Nemotron-1.5B", "OpenMath-Nemotron-1.5B", 16384, "OMN-1.5B", "15k"),
         "QuestA": ("QuestA-Nemotron-1.5B", "OpenMath-Nemotron-1.5B", 24576, "OMN-1.5B", "24k"),
         "KnowRL": ("KnowRL-Nemotron-1.5B", "OpenMath-Nemotron-1.5B", 24576, "OMN-1.5B", "24k"),
         "DeepMath-Omn": ("DeepMath-Omn-1.5B", "OpenMath-Nemotron-1.5B", 24576, "OMN-1.5B", "24k$^\\dagger$"),
         "e3": ("e3-1.7B", "Qwen3-1.7B", 16384, "Q3-1.7B", "8$\\to$16k"),
         "Polaris-1.7B": ("Polaris-1.7B-Preview", "Qwen3-1.7B", 49152, "Q3-1.7B", "40$\\to$48k"),
         "Polaris-4B": ("Polaris-4B-Preview", "Qwen3-4B", 53248, "Q3-4B", "40$\\to$52k"),
         "Skywork-7B": ("Skywork-OR1-7B", "DeepSeek-R1-Distill-Qwen-7B", 32768, "R1-7B", "16$\\to$32k"),
         "AceReason-14B": ("AceReason-Nemotron-14B", "DeepSeek-R1-Distill-Qwen-14B", 32768, "R1-14B", "8$\\to$32k"),
         "Skywork-32B": ("Skywork-OR1-32B", "DeepSeek-R1-Distill-Qwen-32B", 24576, "R1-32B", "16$\\to$24k"),  # released step 1000 is in the 24k stage (Skywork-OR1 report, Table 12)
         "Skywork-Math": ("Skywork-OR1-Math-7B", "DeepSeek-R1-Distill-Qwen-7B", 32768, "R1-7B", "8$\\to$32k"),
         "AceReason": ("AceReason-Nemotron-7B", "DeepSeek-R1-Distill-Qwen-7B", 32768, "R1-7B", "8$\\to$32k")}
LAB = {4096: "4k", 8192: "8k", 16384: "16k", 24576: "24k", 28672: "28k", 32768: "32k", 36864: "36k", 49152: "48k", 53248: "52k", 65536: "64k"}
def caps_for(model):
    return sorted({int(os.path.basename(f).split("_cap")[1].split(".")[0]) for d in ROOTS for f in glob.glob(f"{d}/{model}_cap*.jsonl")})
def load(model, cap):
    fs = [f for d in ROOTS for f in glob.glob(f"{d}/{model}_cap{cap}.jsonl")]; by = collections.defaultdict(list)
    for l in open(fs[0]):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return {k: v for k, v in by.items() if k[0] == "phys"}
solved = lambda rows: sum(r["score"] >= 1 - 1e-9 for r in rows) >= 1
def cell(B, R):
    keys = [k for k in B if k in R]; nb = sum(len(B[k]) for k in keys); nr = sum(len(R[k]) for k in keys)
    fb = sum(r["finished"] for k in keys for r in B[k]) / nb; fr = sum(r["finished"] for k in keys for r in R[k]) / nr
    ab = (sum(r["score"] for k in keys for r in B[k] if r["finished"]) / max(1, sum(r["finished"] for k in keys for r in B[k])))
    ar = (sum(r["score"] for k in keys for r in R[k] if r["finished"]) / max(1, sum(r["finished"] for k in keys for r in R[k])))
    mb = sum(r["score"] for k in keys for r in B[k]) / nb; mr = sum(r["score"] for k in keys for r in R[k]) / nr
    rl_only = [k for k in keys if solved(R[k]) and not solved(B[k])]; base_only = [k for k in keys if solved(B[k]) and not solved(R[k])]
    return {"F": (fr - fb) * ar, "P": fb * (ar - ab), "d": mr - mb, "fb": fb, "fr": fr, "ab": ab, "ar": ar, "base_mean": mb, "rl_mean": mr, "n": len(keys),
            "rl_only": len(rl_only), "base_only": len(base_only), "rl_only_base_all_finished": sum(1 for k in rl_only if all(r["finished"] for r in B[k])),
            "base_only_rl_all_finished": sum(1 for k in base_only if all(r["finished"] for r in R[k]))}
out = {"recipes": {}, "pooled": {}}
for name, (rl, base, cap, tag, caplabel) in PAIRS.items():
    cr, cb = caps_for(rl), caps_for(base); common = sorted(set(cr) & set(cb))
    if not common: continue
    at = min((c for c in common if c >= cap), default=None); beyond = min((c for c in common if at and c > at), default=None)
    rec = {"base": tag, "cap_label": caplabel, "train_cap": cap, "at_cap": at, "beyond_cap": beyond, "at_label": LAB.get(at), "beyond_label": LAB.get(beyond), "below_only": at is None}
    if at is None:  # read did not reach the training cap (Qwen3 family at 36k): report the highest budget read as 'below cap'
        at = max(common); rec.update({"at_cap": at, "at_label": LAB.get(at) + " (below cap)", "beyond_cap": None, "beyond_label": None})
    rec["at"] = cell(load(base, at), load(rl, at))
    if beyond: rec["beyond"] = cell(load(base, beyond), load(rl, beyond))
    out["recipes"][name] = rec
    print(f"{name:13s} at {rec['at_label']:>8s}: F {rec['at']['F']:+.3f} P {rec['at']['P']:+.3f} d {rec['at']['d']:+.3f} fin {rec['at']['fb']:.2f}->{rec['at']['fr']:.2f}" + (f" | beyond {rec['beyond_label']:>4s}: F {rec['beyond']['F']:+.3f} P {rec['beyond']['P']:+.3f} d {rec['beyond']['d']:+.3f} fin {rec['beyond']['fb']:.2f}->{rec['beyond']['fr']:.2f}" if beyond else " | beyond: not read"))
for which in ("at", "beyond"):
    rs = [r[which] for r in out["recipes"].values() if which in r and not r.get("below_only")]
    if rs: out["pooled"][which] = {"n": len(rs), "F_mean": st.mean(r["F"] for r in rs), "P_mean": st.mean(r["P"] for r in rs), "d_mean": st.mean(r["d"] for r in rs),
                                  "dominance": sum(abs(r["F"]) for r in rs) / max(1e-9, sum(abs(r["F"]) + abs(r["P"]) for r in rs)),
                                  "finish_more": sum(1 for r in rs if r["F"] > 0), "finishing_larger": sum(1 for r in rs if abs(r["F"]) > abs(r["P"])),
                                  "rl_only_nb": sum(r["rl_only_base_all_finished"] for r in rs), "base_only_nb": sum(r["base_only_rl_all_finished"] for r in rs)}
print("pooled:", json.dumps(out["pooled"]))
json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)
