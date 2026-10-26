"""Turn stored model outputs into rows: the step that makes the analysis independent of the task.

The analysis reads rows ``{id, kind, k, finished, score, len}`` and nothing else. Any task
whose outputs can be scored per sample fits: write the outputs as JSON lines

    {"id": ..., "k": ..., "text": ..., "finish_reason": "stop" | "length", "len": tokens}

(``finished: true|false`` may replace ``finish_reason``; ``kind`` may be given per line or
taken from the items file) and score them with the frozen matcher or with your own
``scorer(text, gold) -> float`` named as ``module:function``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from pairaudit.analysis.rows import ROW_FIELDS, iter_jsonl, read_at, register_run
from pairaudit.generate.queue import write_jsonl_atomic

Scorer = Callable[[str, str], float]
DEFAULT_SCORER = "pairaudit.scoring.matcher:score"


def load_scorer(spec: str | Scorer | None = None) -> Scorer:
    """``module:function`` -> callable; default the frozen matcher."""
    if callable(spec):
        return spec
    mod, _, fn = (spec or DEFAULT_SCORER).rpartition(":")
    if not mod or not fn:
        raise ValueError(f"scorer must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(mod), fn)


def score_texts(texts: Iterable[dict], items: dict[str, dict], scorer: Scorer) -> list[dict]:
    """Rows from output lines; ``items`` maps id -> {gt, kind}. Unfinished samples score 0 without calling the scorer."""
    rows = []
    for t in texts:
        it = items.get(t["id"])
        if it is None:
            raise KeyError(f"output for unknown item {t['id']!r}")
        fin = bool(t["finished"]) if "finished" in t else t.get("finish_reason") == "stop"
        if "len" not in t:
            raise ValueError(f"output {t['id']}#{t.get('k')} lacks 'len' (token length): needed to re-read at tighter budgets")
        rows.append({"id": t["id"], "kind": t.get("kind", it.get("kind", "task")), "k": int(t.get("k", 0)), "finished": fin,
                     "score": float(scorer(t["text"], it["gt"])) if fin else 0.0, "len": int(t["len"])})
    return rows


def write_run(root: Path, run: str, model: str, rows: list[dict], gen_cap: int, ladder: Iterable[int] = ()) -> Path:
    """Store one model's rows at ``gen_cap`` as a nested run and register it; tighter ``ladder`` budgets are derived on load."""
    root = Path(root)
    (root / run).mkdir(parents=True, exist_ok=True)
    rows = [{k: r[k] for k in ROW_FIELDS} for r in read_at(rows, gen_cap)]
    path = root / run / f"{model}_cap{gen_cap}.jsonl.gz"
    write_jsonl_atomic(path, rows, gz=True)
    caps = sorted({c for c in ladder if c < gen_cap} | {gen_cap})
    register_run(root, run, {model: {"generation_cap": gen_cap, "caps": caps}}, nested=True)
    return path


def rescore_file(texts_path: Path, items_path: Path, scorer: str | Scorer | None = None) -> list[dict]:
    items = {}
    for line in open(items_path, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            items[it["id"]] = it
    return score_texts(iter_jsonl(Path(texts_path)), items, load_scorer(scorer))
