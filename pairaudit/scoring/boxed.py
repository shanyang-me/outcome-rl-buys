"""Boxed-answer extraction: every balanced ``\\boxed{...}`` in a model's text, in order.

A malformed box (unbalanced braces, usually because the budget cut the text off) is skipped
rather than failing the whole parse, so a truncated final box never erases credit for the
valid ones before it.
"""

from __future__ import annotations

_BOX_COMMAND = r"\boxed"
_MAX_SOLUTION_CHARS = 1_000_000


def _braced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth, index = 1, start + 1
    while index < len(text) and depth:
        character = text[index]
        if character == "{" and text[index - 1] != "\\":
            depth += 1
        elif character == "}" and text[index - 1] != "\\":
            depth -= 1
        index += 1
    if depth:
        return None
    return text[start + 1 : index - 1].strip(), index


def extract_boxed(text: str) -> list[str]:
    """Return the contents of every balanced boxed value in ``text``."""
    if not isinstance(text, str) or len(text) > _MAX_SOLUTION_CHARS:
        return []
    values: list[str] = []
    cursor = 0
    while True:
        command = text.find(_BOX_COMMAND, cursor)
        if command < 0:
            return values
        brace = command + len(_BOX_COMMAND)
        while brace < len(text) and text[brace].isspace():
            brace += 1
        parsed = _braced(text, brace)
        if parsed is None:
            cursor = command + len(_BOX_COMMAND)
            continue
        value, cursor = parsed
        if value:
            values.append(value)
