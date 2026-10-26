"""Round-21 M21: the lone wins of Table tab:sides at 16k, recounted from the main read's rows (expansion/rows and public/rows, physics,
eight samples, full credit). A lone win is a problem one model solves at least once in eight while the other solves it in none; it is a
budget effect when the losing model leaves at least one sample unfinished, and otherwise the losing model finished all eight and got
all eight wrong. Pairs: the eighteen smaller recipes and their five bases (5,400 pair-problems).
usage: python3 -I lone_wins_16k.py   (run from probes/pair_decomp)"""
import collections, glob, json, os
files = {}
for d in ("expansion/rows", "public/rows"):
    for f in glob.glob(d + "/*_cap16384.jsonl"): files[os.path.basename(f)[:-len("_cap16384.jsonl")]] = f
B1, B7, NEM, Q17, Q4 = "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "OpenMath-Nemotron-1.5B", "Qwen3-1.7B", "Qwen3-4B"
PAIRS = {B1: ["DeepScaleR-1.5B-Preview", "Nemotron-Research-Reasoning-Qwen-1.5B", "Open-RS3", "STILL-3-1.5B-preview", "JustRL-DeepSeek-1.5B",
              "DeepMath-1.5B", "RLinf-math-1.5B", "AReaL-1.5B-Preview-Stage-3"],
         B7: ["Skywork-OR1-7B", "AceReason-Nemotron-7B", "Skywork-OR1-Math-7B"],
         NEM: ["JustRL-Nemotron-1.5B", "QuestA-Nemotron-1.5B", "KnowRL-Nemotron-1.5B", "DeepMath-Omn-1.5B"],
         Q17: ["e3-1.7B", "Polaris-1.7B-Preview"], Q4: ["Polaris-4B-Preview"]}
def load(m):  # problem -> [samples, finished, full-credit correct]
    g = collections.defaultdict(lambda: [0, 0, 0])
    for l in open(files[m]):
        r = json.loads(l)
        if r["kind"] != "phys": continue
        c = g[r["id"]]; c[0] += 1; c[1] += bool(r["finished"]); c[2] += bool(r["finished"] and r["score"] >= 1 - 1e-9)
    return g
C = collections.Counter(); cases = 0
for b, rs in PAIRS.items():
    B = load(b)
    for rl in rs:
        R = load(rl); ids = sorted(set(B) & set(R)); cases += len(ids)
        for q in ids:
            if R[q][2] > 0 and B[q][2] == 0: C["rl"] += 1; C["rl_U"] += (B[q][1] == 8)
            if B[q][2] > 0 and R[q][2] == 0: C["base"] += 1; C["base_U"] += (R[q][1] == 8)
print("pair-problems", cases)
for s in ("rl", "base"):
    n, u = C[s], C[s + "_U"]
    print("%s lone wins %d (%.1f%%); the other wrong in all eight %d; budget effects %d (%.0f%%)" % (s, n, 100 * n / cases, u, n - u, 100 * (n - u) / n))
