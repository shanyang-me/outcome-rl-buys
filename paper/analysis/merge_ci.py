"""Merge per-cell CI JSONs (the published v5 cells keep their numbers; added cells come from a separate pairs_ci run on RECIPES=added10).
usage: python merge_ci.py out.json in1.json in2.json ..."""
import json, sys
out = {}
for f in sys.argv[2:]:
    d = json.load(open(f))
    clash = set(d) & set(out)
    assert not clash, ("cells computed twice", sorted(clash)[:5])
    out.update(d)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"{len(out)} cells, {len({c['rl'] for c in out.values()})} recipes, {len({c['base'] for c in out.values()})} bases")
