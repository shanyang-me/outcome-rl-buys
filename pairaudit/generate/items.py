"""Problem files and per-model generation specs."""

from __future__ import annotations

import json
from pathlib import Path


def load_items(path: str | Path) -> list[dict]:
    """Rows ``{id, q, gt, kind}``: the problem id, its text, the gold record (a JSON string for the matcher) and its benchmark."""
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for it in items:
        missing = [k for k in ("id", "q", "gt", "kind") if k not in it]
        if missing:
            raise ValueError(f"item {it.get('id')!r} lacks {missing}")
    return items


def load_math500(path: str | Path) -> list[dict]:
    """MATH-500's test.jsonl as items with symbolic golds."""
    items = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        items.append({"id": r.get("unique_id", str(len(items))), "q": r["problem"],
                      "gt": json.dumps({"equiv_mode": "sympy", "golds": [r["answer"]]}), "kind": "math"})
    return items


def load_spec(spec: str | Path | dict | None, default: tuple[float, float, int]) -> dict[str, tuple[float, float, int]]:
    """``{model_name: [temperature, top_p, generation_cap]}`` from a file, a JSON string or a dict; missing models get ``default``."""
    if spec is None:
        d = {}
    elif isinstance(spec, dict):
        d = spec
    else:
        s = str(spec)
        d = json.loads(s) if s.lstrip().startswith("{") else json.loads(Path(s).read_text())

    class _Spec(dict):
        def __missing__(self, key):
            return default

    return _Spec({k: (float(v[0]), float(v[1]), int(v[2])) for k, v in d.items()})
