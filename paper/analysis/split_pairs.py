"""Write each discordant pair as three files for reading: question.md, rl.md, base.md. usage: python split_pairs.py trace_out <model>"""
import json, sys, os
O, M = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{O}/discordant_{M}.jsonl")]
for k, r in enumerate(rows):
    d = f"{O}/read/{M}/pair_{k:02d}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/question.md", "w").write(f"# Problem ({r['kind']}, id {r['id']})\n\n{r['question']}\n\n# Gold answer\n\n{r['gold']}\n")
    open(f"{d}/rl.md", "w").write(f"# Derivation A: {r['rl_model']} (RL), scored full credit, {r['rl_len']} tokens\n\n{r['rl_text']}\n")
    open(f"{d}/base.md", "w").write(f"# Derivation B: {r['base_model']} (base), scored {r['base_score']:.2f}, {r['base_len']} tokens\n\n{r['base_text']}\n")
print(len(rows), "pairs written to", f"{O}/read/{M}")
