"""Matcher audit sample (Oct 18): for each model with stored texts, the physics samples that finished within the budget and that the
frozen matcher scored below 1; draw 200 per model (seed 0) with the problem, gold, final boxed answer and the derivation's last 900
characters. Also writes each model's full counts at that budget, so judged rates can be scaled back. Output /workspace/w3/judge_audit/."""
import json, gzip, os, random, sys, collections
sys.path.insert(0, "/workspace/repo")
from rl.w3.outcome_reward import extract_boxed
W = "/workspace/w3"; OUT = f"{W}/judge_audit"; os.makedirs(OUT, exist_ok=True)
SRC = [("pair_decomp4", m, 16384) for m in ["OpenMath-Nemotron-1.5B", "JustRL-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "DeepMath-Omn-1.5B", "Qwen3-1.7B", "e3-1.7B", "Polaris-1.7B-Preview"]] + \
      [("pair_decomp10_native", "Qwen3-1.7B", 49152), ("pair_decomp10_native", "Polaris-1.7B-Preview", 49152), ("pair_decomp10_native", "Qwen3-4B", 53248), ("pair_decomp10_native", "Polaris-4B-Preview", 53248),
       ("pair_decomp9_32b", "DeepSeek-R1-Distill-Qwen-32B", 32768), ("pair_decomp9_32b", "Skywork-OR1-32B", 32768)]
gt = {}
for l in open(f"{W}/pair_decomp3_a/phys300.jsonl"):
    r = json.loads(l); gt[r["id"]] = (r["q"], json.loads(r["gt"])["golds"])
counts, items = {}, []
for d, m, B in SRC:
    f = f"{W}/{d}/texts/{m}.jsonl.gz"
    if not os.path.exists(f): print("MISSING", f, flush=True); continue
    allp = []; c = collections.Counter()
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        if r["kind"] != "phys": continue
        c["n"] += 1; fin = r["finished"] and r["len"] <= B
        if not fin: continue
        c["fin"] += 1; c["score_sum"] += r["score"]
        if r["score"] < 1 - 1e-9:
            c["wrong"] += 1; b = extract_boxed(r["text"])
            allp.append({"src": f"{d}|{m}|{B}", "id": r["id"], "k": r["k"], "score": r["score"], "answer": (" ; ".join(b) if b else "<no boxed answer>")[:400], "tail": r["text"][-900:]})
    rng = random.Random(0); pick = rng.sample(allp, min(200, len(allp)))
    for p in pick: q, g = gt[p["id"]]; p["problem"] = q[:1500]; p["gold"] = g
    items += pick; counts[f"{d}|{m}|{B}"] = dict(c)
    print(m, B, dict(c), "sampled", len(pick), flush=True)
json.dump(counts, open(f"{OUT}/counts.json", "w"), indent=1)
with open(f"{OUT}/items.jsonl", "w") as h:
    for it in items: h.write(json.dumps(it) + "\n")
print("EXTRACT_DONE", len(items), flush=True)
