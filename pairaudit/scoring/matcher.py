"""The frozen matcher: outcome reward as the fraction of a problem's gold answers matched by any boxed value.

This is the scorer every number in the audit was produced with. It is frozen on purpose: a
matcher change moves every cell of every model at once, so changes are reporting-only and
go through the judge protocol (``judge_protocol``) instead.

Ground truth is a JSON string ``{"equiv_mode": "numeric" | "sympy", "golds": [...]}``.
Scoring definition:
  * a multi-value gold credits the fraction of its components some boxed value matches;
  * prose golds are excluded from the denominator and counted;
  * unparseable golds stay in the denominator, score 0, and are counted.
``score_details`` exposes all of that per gold; ``compute_score`` is the bare float.
"""

from __future__ import annotations

import json
from typing import Any

from pairaudit.scoring.answer_equiv import gold_scorability, match_fraction
from pairaudit.scoring.boxed import extract_boxed

_MAX_GOLDS = 16
_MAX_CANDIDATES = 32

__all__ = ["compute_score", "extract_boxed", "load_golds", "score", "score_details"]


def load_golds(ground_truth: str) -> list[str] | None:
    """Parse the ground-truth record; None when it is not a valid gold record."""
    try:
        record = json.loads(ground_truth)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("equiv_mode") not in {"numeric", "sympy"}:
        return None
    golds = record.get("golds")
    if (
        not isinstance(golds, list)
        or not 1 <= len(golds) <= _MAX_GOLDS
        or any(not isinstance(gold, str) or not gold.strip() for gold in golds)
    ):
        return None
    return [gold.strip() for gold in golds]


def score_details(solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Per-gold credit under the scoring definition, plus the counts the definition requires."""
    golds = load_golds(ground_truth)
    if golds is None:
        return {
            "score": 0.0, "golds": [], "scorability": [], "credit": [], "scored_golds": 0,
            "prose_excluded": 0, "unparseable": 0, "candidates": [], "invalid_ground_truth": True,
        }
    boxed = extract_boxed(solution_str) if isinstance(solution_str, str) else []
    candidates = list(dict.fromkeys(boxed))[-_MAX_CANDIDATES:]
    scorability = [gold_scorability(gold) for gold in golds]
    credit: list[float] = []
    for gold, kind in zip(golds, scorability):
        if kind in ("prose", "unparseable") or not candidates:
            credit.append(0.0)
        else:
            credit.append(float(match_fraction(candidates, gold)))
    scored = [value for value, kind in zip(credit, scorability) if kind != "prose"]
    return {
        "score": sum(scored) / len(scored) if scored else 0.0,
        "golds": golds,
        "scorability": scorability,
        "credit": credit,
        "scored_golds": len(scored),
        "prose_excluded": sum(1 for kind in scorability if kind == "prose"),
        "unparseable": sum(1 for kind in scorability if kind == "unparseable"),
        "candidates": candidates,
        "invalid_ground_truth": False,
    }


def score(solution_str: str, ground_truth: str) -> float:
    """Fraction of scorable golds credited by the boxed values, in [0, 1]."""
    return float(score_details(solution_str, ground_truth)["score"])


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """verl-compatible reward signature (``data_source`` and ``extra_info`` are ignored)."""
    del data_source, extra_info
    return score(solution_str, ground_truth)
