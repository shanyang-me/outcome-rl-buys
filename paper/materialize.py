"""Write every model x budget of data/rows as a plain JSON-lines file, the layout the paper's analysis scripts read.

Nested runs store one file per model at its generation cap; each tighter budget in MANIFEST.json is derived by the same rule
the paper uses (pairaudit.analysis.rows.read_at): a sample is finished at budget C iff it finished and its length is at most C.
usage: python paper/materialize.py [out_dir]   (default build/rows; out_dir/<run>/<model>_cap<C>.jsonl)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pairaudit.analysis.rows import RowStore

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "rows"
store = RowStore(ROOT / "data" / "rows")
manifest = json.loads((ROOT / "data" / "rows" / "MANIFEST.json").read_text())
n = 0
for run, spec in manifest.items():
    for model, m in spec["models"].items():
        for cap in m["caps"]:
            d = OUT / run
            d.mkdir(parents=True, exist_ok=True)
            hit = store.locate(model, cap)
            if hit is None or hit[0].name != run:
                rows = RowStore(ROOT / "data" / "rows", runs=[run]).rows(model, cap)
            else:
                rows = store.rows(model, cap)
            with open(d / f"{model}_cap{cap}.jsonl", "w") as h:
                for r in rows:
                    h.write(json.dumps(r) + "\n")
            n += 1
print(f"wrote {n} files under {OUT}")
