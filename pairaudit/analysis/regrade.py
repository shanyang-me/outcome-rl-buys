"""Cells under the judge: the three-part split recomputed from regraded rows (``score_j``) beside the matcher's."""

from __future__ import annotations

from pathlib import Path

from pairaudit.analysis.decomp import arrays, three_part
from pairaudit.analysis.rows import group_by_problem, iter_jsonl, read_at


def _judged(rows: list[dict], key: str) -> list[dict]:
    return [{**r, "score": r[key]} for r in rows]


def regrade_cells(regrade_dir: Path, pairs: list[tuple[str, str, str]], caps: list[int], kind: str | None = None) -> dict:
    """For each (run tag, base, rl) whose judged rows exist, the split at each budget under the matcher and under the judge.

    ``kind`` defaults to the first benchmark in the judged rows (the judge only regrades the primary one).
    """
    out = {}
    for tag, base, rl in pairs:
        fb, fr = Path(regrade_dir) / tag / f"{base}.jsonl", Path(regrade_dir) / tag / f"{rl}.jsonl"
        if not fb.exists() or not fr.exists():
            continue
        B, R = list(iter_jsonl(fb)), list(iter_jsonl(fr))
        kind = kind or B[0]["kind"]
        for cap in caps:
            cell = {}
            for key, label in (("score", "matcher"), ("score_j", "judge")):
                c = arrays(group_by_problem(read_at(_judged(B, key), cap), [kind]), group_by_problem(read_at(_judged(R, key), cap), [kind]), kind)
                if c is None:
                    continue
                p = three_part(c)
                p["share"] = abs(p["F"]) / (abs(p["F"]) + abs(p["P"])) if abs(p["F"]) + abs(p["P"]) > 0 else None
                cell[label] = p
            if cell:
                out[f"{rl} vs {base} @{cap // 1024}k ({tag})"] = cell
    return out
