"""For every read pair whose RL derivation states a step the base sample lacks, write all four base samples on that problem
(base_k0.md .. base_k3.md) beside the pair, so a checker can ask whether the step is in the base's repertoire at all.
usage: python exp2/export_base_samples2.py traces_dir trace_out [model,model,...]  (run from the scratchpad root)"""
import json, sys, os, glob, collections
T, O = sys.argv[1], sys.argv[2]
import sys as _s; _s.path.insert(0, '.')
from families import FAMILIES
BASE = {rl: b for b, rls in FAMILIES.items() for rl in rls}
ONLY = set(sys.argv[3].split(',')) if len(sys.argv) > 3 else None   # restrict to these RL models (reading files of others stay untouched)
cache = {}
def base_rows(b):
    if b not in cache:
        by = collections.defaultdict(list)
        for l in open(f"{T}/{b}.jsonl"):
            r = json.loads(l); by[(r["kind"], r["id"])].append(r)
        cache[b] = by
    return cache[b]
todo = []
for rp in sorted(glob.glob(f"{O}/reading_*.json")):
    R = json.load(open(rp)); m = R["model"]
    if ONLY and m not in ONLY: continue
    disc = [json.loads(l) for l in open(f"{O}/discordant_{m}.jsonl")]
    for p in R["pairs"]:
        if not p["rl_new_step"]: continue
        pdir = f"{O}/read/{m}/pair_{p['k']:02d}"
        import re as _re   # the read pair's own question.md is the source of truth for the problem id (discordant files can be reshuffled)
        kind, pid = _re.search(r"\((phys|math), id ([^)]+)\)", open(f"{pdir}/question.md").read(300)).groups()
        rows = sorted(base_rows(BASE[m])[(kind, pid)], key=lambda r: r["k"])
        for r in rows:
            open(f"{pdir}/base_k{r['k']}.md", "w").write(f"# Base sample k={r['k']} ({BASE[m]}), finished={r['finished']}, score={r['score']:.2f}, {r['len']} tokens\n\n{r['text']}\n")
        todo.append({"model": m, "k": p["k"], "dir": pdir, "quote": p["rl_new_step_quote"], "summary": p["summary"], "n_base": len(rows)})
json.dump(todo, open(f"{O}/repertoire_todo.json", "w"), indent=1)
print(len(todo), "new-step pairs exported:", collections.Counter(t["model"] for t in todo))
