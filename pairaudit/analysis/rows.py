"""Row files: one JSON line per sample, and the nested-budget read that turns one generation into every tighter budget.

A row is ``{"id", "kind", "k", "finished", "score", "len"}`` (``think_closed`` is optional):
the problem, its benchmark (``phys`` or ``math``), the sample index, whether the sample
stopped on its own within the budget, the matcher's partial-credit score (0 when
unfinished), and the token length (clipped to the budget).

Runs are directories under ``data/rows``. A *nested* run holds one file per model at the
generation cap; a tighter budget is the same samples re-read: a sample is finished at
budget C iff it finished and its length is at most C, otherwise it is unfinished and
scores 0. Separate-generation runs hold one file per budget. ``MANIFEST.json`` says which.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

Row = dict
Key = tuple[str, str]  # (kind, id)

ROW_FIELDS = ("id", "kind", "k", "finished", "score", "len")


def read_at(rows: Iterable[Row], cap: int) -> list[Row]:
    """Re-read samples generated at a larger budget as if the budget were ``cap``."""
    out = []
    for r in rows:
        fin = bool(r["finished"]) and r["len"] <= cap
        out.append({**r, "finished": fin, "score": r["score"] if fin else 0.0, "len": min(r["len"], cap)})
    return out


def iter_jsonl(path: Path) -> Iterator[Row]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def group_by_problem(rows: Iterable[Row], kinds: Iterable[str] | None = None) -> dict[Key, list[Row]]:
    keep = set(kinds) if kinds else None
    by: dict[Key, list[Row]] = defaultdict(list)
    for r in rows:
        if keep is None or r["kind"] in keep:
            by[(r["kind"], r["id"])].append(r)
    return dict(by)


@dataclass(frozen=True)
class RunSpec:
    name: str
    nested: bool
    models: dict  # model -> {"generation_cap": int, "caps": [int, ...]}


class RowStore:
    """Row files for a list of runs, searched in order; the first run holding a (model, budget) wins."""

    def __init__(self, root: str | Path, runs: list[str] | None = None):
        self.root = Path(root)
        manifest = json.loads((self.root / "MANIFEST.json").read_text())
        names = runs if runs is not None else list(manifest)
        missing = [r for r in names if r not in manifest]
        if missing:
            raise KeyError(f"runs not in {self.root / 'MANIFEST.json'}: {missing}")
        self.runs = [RunSpec(r, manifest[r]["nested"], manifest[r]["models"]) for r in names]

    def caps_for(self, model: str) -> list[int]:
        """Every budget some run in this store can read ``model`` at."""
        caps: set[int] = set()
        for run in self.runs:
            if model in run.models:
                caps.update(run.models[model]["caps"])
        return sorted(caps)

    def locate(self, model: str, cap: int) -> tuple[RunSpec, Path, bool] | None:
        """(run, file, derive?) for the first run that can read ``model`` at ``cap``."""
        for run in self.runs:
            spec = run.models.get(model)
            if not spec or cap not in spec["caps"]:
                continue
            file_cap = spec["generation_cap"] if run.nested else cap
            path = self.root / run.name / f"{model}_cap{file_cap}.jsonl.gz"
            if not path.exists():
                path = path.with_suffix("")  # plain .jsonl
            if path.exists():
                return run, path, run.nested and file_cap != cap
        return None

    def rows(self, model: str, cap: int) -> list[Row]:
        hit = self.locate(model, cap)
        if hit is None:
            raise FileNotFoundError(f"no rows for {model} at {cap} in runs {[r.name for r in self.runs]}")
        _, path, derive = hit
        rows = list(iter_jsonl(path))
        return read_at(rows, cap) if derive else rows

    def load(self, model: str, cap: int, kinds: Iterable[str] | None = None) -> dict[Key, list[Row]]:
        """Rows grouped by (kind, id), optionally restricted to some benchmarks."""
        return group_by_problem(self.rows(model, cap), kinds)

    def has(self, model: str, cap: int) -> bool:
        return self.locate(model, cap) is not None

    def common_caps(self, models: Iterable[str]) -> list[int]:
        """Budgets every one of ``models`` can be read at."""
        caps = None
        for m in models:
            c = set(self.caps_for(m))
            caps = c if caps is None else caps & c
        return sorted(caps or [])

    def kinds_for(self, model: str) -> tuple[str, ...]:
        """Benchmarks in a model's rows, in order of first appearance."""
        caps = self.caps_for(model)
        if not caps:
            raise FileNotFoundError(f"no rows for {model}")
        seen: dict[str, None] = {}
        for r in self.rows(model, caps[-1]):
            seen.setdefault(r["kind"])
        return tuple(seen)


def resolve_kinds(store: RowStore, reg, recipes: list, kinds: Iterable[str] | None = None) -> tuple[str, ...]:
    """The benchmarks to analyse: the caller's, else the config's, else those in the first pair's base rows."""
    if kinds:
        return tuple(kinds)
    if reg.kinds:
        return reg.kinds
    return store.kinds_for(recipes[0].base) if recipes else ()


def register_run(root: str | Path, run: str, models: dict, nested: bool = True, note: str = "") -> None:
    """Add a run to ``MANIFEST.json`` or extend its models; ``models`` maps model -> {generation_cap, caps}."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    entry = manifest.setdefault(run, {"nested": nested, "models": {}})
    entry["nested"] = nested
    entry["models"].update(models)
    if note:
        entry["files"] = note
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")


def summarize(rows: Iterable[Row]) -> dict[str, dict]:
    """Per-benchmark cell summary: mean score, finished rate, score given finished, any-solve@k, mean length."""
    by_kind: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    out = {}
    for kind, rs in sorted(by_kind.items()):
        fin = [r for r in rs if r["finished"]]
        by = defaultdict(list)
        for r in rs:
            by[r["id"]].append(r["score"])
        out[kind] = {
            "mean": sum(r["score"] for r in rs) / len(rs),
            "finished": len(fin) / len(rs),
            "score_given_finished": (sum(r["score"] for r in fin) / len(fin)) if fin else 0.0,
            "any_solve": sum(max(v) >= 1 - 1e-9 for v in by.values()) / len(by),
            "mean_len": sum(r["len"] for r in rs) / len(rs),
            "n_items": len(by),
            "n_samples": len(rs),
        }
    return out
