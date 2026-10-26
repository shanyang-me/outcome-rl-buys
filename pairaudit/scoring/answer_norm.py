"""Deterministic normalisation of physics answers, for a judge protocol that is stable
across graders.

Why this exists: the same 526 problems scored 40.1 under our reconstructed judge and 24.0
under a pinned Qwen3-32B, and the disagreements concentrated in one place --- whether a
bare number matches a gold carrying a unit. `6` against gold `6V` is correct and one judge
said no; `-0.025` against gold `0.025m` is a sign error and the other judge said yes. Left
to a model's discretion, that call moves benchmark scores by more than the effects being
measured, and it moves them *unevenly*: systems trained on the benchmark's own split emit
units 97.2% of the time, systems that were not emit them 64-69% of the time, so a strict
grader penalises the latter for formatting rather than physics.

The fix is to decide the mechanical cases mechanically and hand the model only what
genuinely needs judgement. This module does the first part.

Scope, stated honestly: this handles scalar numeric answers with optional units and an
optional direction word. Symbolic expressions, multi-valued families and set-valued golds
are explicitly NOT decided here --- `compare` returns None for them and the caller must
defer to a model judge.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# LaTeX scaffolding that carries no semantic content for matching
_STRIP = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\!|\\,|\\;|\\:|\\quad|\\qquad"), " "),
    (re.compile(r"~"), " "),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathbf\{([^}]*)\}"), r"\1"),
    (re.compile(r"[$\\]?\s*\\?approx\s*"), " "),
    (re.compile(r"\$+"), ""),
    (re.compile(r"\s+"), " "),
]

# unit aliases -> canonical, with SI scale relative to the canonical base
_UNITS: dict[str, tuple[str, float]] = {}


def _reg(canon: str, scale: float, *names: str) -> None:
    for n in names:
        _UNITS[n.lower()] = (canon, scale)


# length
_reg("m", 1.0, "m", "meter", "meters", "metre", "metres")
_reg("m", 1e-2, "cm", "centimeter", "centimetre")
_reg("m", 1e-3, "mm")
_reg("m", 1e3, "km")
# time
_reg("s", 1.0, "s", "sec", "second", "seconds")
_reg("s", 1e-3, "ms")
_reg("s", 60.0, "min", "minute", "minutes")
_reg("s", 3600.0, "h", "hr", "hour", "hours")
# mass
_reg("kg", 1.0, "kg")
_reg("kg", 1e-3, "g", "gram", "grams")
# electrical
_reg("V", 1.0, "v", "volt", "volts")
_reg("V", 1e3, "kv")
_reg("A", 1.0, "a", "amp", "ampere", "amperes")
_reg("A", 1e-3, "ma")
_reg("ohm", 1.0, "ohm", "ohms", "\\omega", "Ω")
_reg("C", 1.0, "c", "coulomb", "coulombs")
_reg("F", 1.0, "f", "farad")
_reg("T", 1.0, "t", "tesla")
# mechanics / energy
_reg("N", 1.0, "n", "newton", "newtons")
_reg("J", 1.0, "j", "joule", "joules")
_reg("W", 1.0, "w", "watt", "watts")
_reg("Pa", 1.0, "pa", "pascal")
_reg("Hz", 1.0, "hz", "hertz")
# energy, temperature, amount, angle, frequency, pressure, charge, field (Oct 19: added for the harder physics sets)
_reg("eV", 1.0, "ev", "electronvolt", "electronvolts")
_reg("eV", 1e3, "kev")
_reg("eV", 1e6, "mev")
_reg("eV", 1e9, "gev")
_reg("J", 1e3, "kj")
_reg("J", 1e6, "mj")
_reg("W", 1e3, "kw")
_reg("W", 1e6, "mw")
_reg("N", 1e3, "kn")
_reg("K", 1.0, "k", "kelvin")
_reg("mol", 1.0, "mol", "mole", "moles")
_reg("rad", 1.0, "rad", "radian", "radians")
_reg("deg", 1.0, "deg", "degree", "degrees", "°", "^\\circ", "^{\\circ}", "\\circ", "\\degree")
_reg("Hz", 1e3, "khz")
_reg("Hz", 1e6, "mhz")
_reg("Hz", 1e9, "ghz")
_reg("Pa", 1e3, "kpa")
_reg("Pa", 1e6, "mpa")
_reg("Pa", 101325.0, "atm")
_reg("m", 1e-6, "μm", "um", "micrometer", "micron")
_reg("m", 1e-9, "nm")
_reg("C", 1e-6, "μc", "uc")
_reg("C", 1e-9, "nc")
_reg("F", 1e-6, "μf", "uf")
_reg("F", 1e-12, "pf")
_reg("T", 1e-3, "mt")
_reg("A", 1e-6, "μa", "ua")
# composite, treated as opaque canonical strings
for _c in ("m/s", "m/s^2", "km/h", "n/c", "v/m", "rad/s", "kg*m/s", "j/k", "w/m^2", "m^2", "m^3", "kg/m^3", "g/cm^3", "j/mol", "j/(mol*k)", "n*m", "n/m", "m/s^3", "rad/s^2", "a/m", "t*m", "w/m", "kg/s", "mol/l", "m^-1", "s^-1", "m^2/s", "n*s"):
    _reg(_c.replace("*", ""), 1.0, _c, _c.replace("/", " / "), _c.replace("^", "^{") + ("}" if "^" in _c else ""))

# surface form -> canonical direction, so "upward" and "up" compare equal
_DIRECTIONS = {
    "north": "north", "south": "south", "east": "east", "west": "west",
    "up": "up", "upward": "up", "upwards": "up",
    "down": "down", "downward": "down", "downwards": "down",
    "left": "left", "right": "right",
    "clockwise": "clockwise", "counterclockwise": "counterclockwise",
    "anticlockwise": "counterclockwise",
    "inward": "inward", "inwards": "inward", "outward": "outward", "outwards": "outward",
    "positive": "positive", "negative": "negative",
    "horizontal": "horizontal", "vertical": "vertical",
}


@dataclass(frozen=True)
class Answer:
    value: float | None          # signed magnitude in canonical units
    unit: str | None             # canonical unit, None if bare
    direction: str | None        # a direction word if one was present
    raw: str


def _clean(s: str) -> str:
    for pat, rep in _STRIP:
        s = pat.sub(rep, s)
    return s.strip()


def _to_float(tok: str) -> float | None:
    """Parse a numeric token, including LaTeX scientific notation and simple fractions."""
    tok = tok.strip()
    if re.fullmatch(r"-?[\d.]+[eE][-+]?\d+", tok):
        try:
            return float(tok)
        except ValueError:
            return None
    m = re.fullmatch(r"(-?[\d.]+)\s*(?:\\times|\\cdot|x|\*)\s*10\^?\{?(-?\d+)\}?", tok)
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except ValueError:
            return None
    m = re.fullmatch(r"-?\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", tok)
    if m:
        try:
            v = float(m.group(1)) / float(m.group(2))
            return -v if tok.startswith("-") else v
        except (ValueError, ZeroDivisionError):
            return None
    m = re.fullmatch(r"(-?[\d.]+)\s*/\s*(-?[\d.]+)", tok)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(tok)
    except ValueError:
        return None


def parse(s: str) -> Answer | None:
    """Parse one scalar answer. Returns None when this is not a scalar we can decide."""
    if s is None:
        return None
    raw = str(s)
    t = _clean(raw)
    if not t:
        return None
    # multi-valued or set-valued golds are out of scope
    if t.count(",") >= 1 and re.search(r"\d\s*,\s*[-\d]", t):
        return None
    if re.search(r"\bor\b|\band\b|;", t, re.I):
        return None

    direction = None
    for w in sorted(_DIRECTIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", t, re.I):
            direction = _DIRECTIONS[w]
            t = re.sub(rf"\b{re.escape(w)}\b", " ", t, flags=re.I).strip(" ,")
            break

    m = re.match(
        r"^([+-]?(?:\\frac\{-?[\d.]+\}\{-?[\d.]+\}|[\d.]+\s*(?:\\times|\\cdot|x|\*)\s*10\^?\{?-?\d+\}?|[\d.]+[eE][-+]?\d+|[\d.]+\s*/\s*[\d.]+|[\d.]+))\s*(.*)$",
        t,
    )
    if not m:
        return None
    val = _to_float(m.group(1))
    if val is None:
        return None
    rest = m.group(2).strip()
    if rest.startswith("{") and rest.endswith("}"):   # a wrapping brace pair only; "m^{2}" keeps its braces
        rest = rest[1:-1].strip()
    rest = re.sub(r"^[.,;]+|[.,;]+$", "", rest).strip()

    unit = None
    if rest:
        key = rest.lower().replace(" ", "")
        if key in _UNITS:
            canon, scale = _UNITS[key]
            unit, val = canon, val * scale
        else:
            # unrecognised trailing text: not a scalar-with-unit we can decide
            return None
    return Answer(val, unit, direction, raw)


def compare(gold: str, pred: str, rel_tol: float = 0.01, abs_tol: float | None = None) -> bool | None:
    """True/False when decidable mechanically, None when a model judge is required.

    The unit rule, stated once so every grader applies it identically: a BARE prediction
    matches a gold carrying a unit when the magnitudes agree, because the unit is fixed by
    the problem statement and the model is not being tested on transcription. A prediction
    carrying a DIFFERENT unit does not match unless the values agree after conversion.
    Sign is always significant. A direction word in the gold is not required in the
    prediction, but a contradictory one fails.
    """
    g, p = parse(gold), parse(pred)
    if g is None or p is None or g.value is None or p.value is None:
        return None

    if g.unit and p.unit and g.unit != p.unit:
        return False
    if abs_tol is not None:
        # a dataset-supplied absolute tolerance replaces the relative one (OlympiadBench's per-problem `error`)
        if abs(g.value - p.value) > abs_tol + 1e-12:
            return False
    elif not math.isclose(g.value, p.value, rel_tol=rel_tol, abs_tol=1e-12):
        return False
    if g.direction and p.direction and g.direction != p.direction:
        opposites = {("up", "down"), ("north", "south"), ("east", "west"),
                     ("left", "right"), ("clockwise", "counterclockwise"),
                     ("positive", "negative"), ("inward", "outward")}
        pair = (g.direction, p.direction)
        if pair in opposites or pair[::-1] in opposites:
            return False
    return True


def compare_any(gold: str, preds, abs_tol: float | None = None) -> bool | None:
    """Apply `compare` across a prediction list. None if no prediction was decidable."""
    saw = False
    for p in preds or []:
        r = compare(gold, p, abs_tol=abs_tol)
        if r is True:
            return True
        if r is False:
            saw = True
    return False if saw else None


__all__ = ["Answer", "parse", "compare", "compare_any"]
