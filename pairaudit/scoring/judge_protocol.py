"""Judge protocol v2: decide mechanically where possible, defer the rest with explicit rules.

The same 526 problems scored 40.1 under one judge and 24.0 under another, and the
disagreement was system-dependent: models trained on the benchmark's own split emit units
97% of the time, models that were not emit them 64%, so a strict grader penalised
formatting rather than physics. The protocol splits every item into three buckets:

  DECIDED_NO  no boxed answer at all; not a judgement call
  DECIDED     ``answer_norm.compare_any`` returned True/False on a scalar with units
  DEFER       symbolic, multi-valued, or otherwise not mechanically decidable

Only DEFER reaches a model, and it arrives with the same rules the deterministic path
applies, plus the parsed forms as an annotation. Two graders should return the same
verdict; where they cannot, the item should have been decided mechanically instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pairaudit.scoring.answer_norm import compare_any, parse

RULES = """You are grading one answer to a physics problem. Apply these rules exactly; they
are the same rules a deterministic checker applies to the mechanical cases, and your
verdicts must be consistent with them.

1. YES if ANY ONE of the model's boxed answers matches the gold. Boxing several
   intermediate results is expected and does not penalise the answer.
2. A BARE NUMBER MATCHES A GOLD CARRYING A UNIT when the magnitudes agree. The unit is
   fixed by the problem; the model is not being tested on transcription. `6` matches `6V`.
   `4` matches `4 m/s`.
3. A DIFFERENT unit matches only after conversion: `2.5 cm` matches `0.025 m`; `5 A` does
   not match `5 V`.
4. SIGN IS SIGNIFICANT. `-0.025` does not match `0.025 m`.
5. A direction or orientation word in the gold need NOT appear in the answer, but a
   CONTRADICTORY one fails: gold "1 m, west" is matched by `1`, not by `1 m east`.
6. Mathematically equivalent forms match: fractions vs decimals, algebraically rearranged
   but equal expressions, equivalent notation.
7. A numerically different value does NOT match, however close. ~3% apart is NO.
8. Multi-part or set-valued golds require EVERY part to match. A partial match is NO.
9. If the gold is corrupted and no answer is recoverable from it, answer NO.

Reply with exactly one word: YES or NO."""

TEMPLATE = """{rules}

GOLD ANSWER:
{gold}

THE MODEL'S BOXED ANSWERS (any one of them may be the attempt at this gold):
{preds}
{note}"""

_MAX_PREDS_SHOWN = 8


def annotate(gold: str, preds) -> str:
    """Show the parsed forms when available, so the model is not re-deriving them."""
    g = parse(gold)
    lines = []
    if g and g.value is not None:
        u = g.unit or "(none)"
        d = f", direction {g.direction}" if g.direction else ""
        lines.append(f"  gold parses as: magnitude {g.value:g}, unit {u}{d}")
    for p in (preds or [])[:_MAX_PREDS_SHOWN]:
        a = parse(str(p))
        if a and a.value is not None:
            u = a.unit or "(bare)"
            d = f", direction {a.direction}" if a.direction else ""
            lines.append(f"  {str(p)[:40]!r} parses as: magnitude {a.value:g}, unit {u}{d}")
    if not lines:
        return ""
    return "\nPARSED FORMS (advisory; the rules above still govern):\n" + "\n".join(lines)


def prompt_for(gold: str, preds) -> str:
    """The deferred-item prompt: rules, gold, the boxed candidates, parsed forms."""
    rendered = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(list(preds)[:_MAX_PREDS_SHOWN])) or "  (none)"
    return TEMPLATE.format(rules=RULES, gold=gold, preds=rendered, note=annotate(gold, preds))


def decide(gold: str, preds, abs_tol: float | None = None) -> bool | None:
    """The mechanical step: False with no candidates, True/False when the checker decides, None to defer."""
    if not preds:
        return False
    return compare_any(gold, preds, abs_tol=abs_tol) if abs_tol is not None else compare_any(gold, preds)


@dataclass
class Split:
    """Items partitioned by the protocol. ``deferred`` maps key -> prompt."""

    decided: dict[str, bool] = field(default_factory=dict)
    deferred: dict[str, str] = field(default_factory=dict)
    no_answer: int = 0

    @property
    def mechanical_share(self) -> float:
        total = len(self.decided) + len(self.deferred)
        return len(self.decided) / total if total else 0.0


def split(items) -> Split:
    """Apply the protocol to ``items`` (dicts with key, gold, preds and an optional abs_tol)."""
    out = Split()
    for it in items:
        key, gold, preds = it["key"], it["gold"], it.get("preds") or []
        verdict = decide(gold, preds, it.get("abs_tol"))
        if verdict is None:
            out.deferred[key] = prompt_for(gold, preds)
        else:
            out.decided[key] = bool(verdict)
            out.no_answer += not preds
    return out


def parse_verdict(text: str) -> str:
    """YES / NO / UNPARSED from a judge completion (a thinking model may emit <think> first)."""
    tail = (text.split("</think>")[-1] if "</think>" in text else text).strip().upper()
    if tail.startswith("YES"):
        return "YES"
    if tail.startswith("NO"):
        return "NO"
    head = tail[:40]
    return "YES" if "YES" in head else "NO" if "NO" in head else "UNPARSED"
