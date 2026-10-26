"""Deterministic, fail-closed answer equivalence for W3 rewards.

Scoring definition: design/SCORING_DEFINITION.md. The public seams are

    equivalent(pred, gold)            one boxed value against one gold (bool)
    match_fraction(candidates, gold)  D3: fraction of a gold's components any candidate matches
    gold_scorability(gold)            D4/D7: "numeric" | "symbolic" | "prose" | "unparseable"

Everything else is implementation. Malformed model output never raises; it fails closed.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import random
import re
import signal
import threading
import time
from typing import Sequence

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# --- units ------------------------------------------------------------------------------
#
# Atoms are matched as whole tokens, longest first. Prefixed units are their own atoms:
# values are compared, never converted, so "5 cm" against "5 mm" must fail closed and
# "1.5 GHz" against "1.5e9" stays a miss (SCORING_DEFINITION D1 excludes conversion).
_PREFIXED_ATOMS = (
    "km", "cm", "mm", "µm", "μm", "um", "nm", "pm", "fm", "dm",
    "mg", "µg", "μg", "ug", "ng",
    "ms", "µs", "μs", "us", "ns", "ps", "fs",
    "kN", "MN", "mN",
    "kJ", "MJ", "GJ", "mJ", "kcal",
    "kW", "MW", "GW", "mW", "µW", "μW",
    "kV", "MV", "mV", "µV", "μV",
    "mA", "µA", "μA", "kA", "nA", "pA",
    "mT", "µT", "μT", "nT",
    "kPa", "MPa", "GPa", "hPa",
    "kHz", "MHz", "GHz", "THz",
    "mmol", "kmol",
    "meV", "keV", "MeV", "GeV", "TeV",
    "mL", "µL", "μL",
    "µF", "μF", "uF", "nF", "pF", "mF",
    "mH", "µH", "μH", "nH",
    "kΩ", "MΩ", "mΩ", "kohm", "Mohm",
    "mWb", "µWb", "μWb",
    "mrad", "kWh", "mAh", "Ah", "Wh",
)
_BASE_ATOMS = (
    "eV", "Pa", "Hz", "Wb", "mol", "rad", "sr", "cd", "atm", "bar", "cal", "min", "rpm",
    "ohm", "lx", "lm", "Bq", "Gy", "Sv", "Gs", "dB", "kg",
    "°C", "°", "%",
    "Ω", "N", "J", "W", "V", "A", "T", "K", "C", "F", "H", "L", "l", "u", "g", "m", "s", "h",
)
_UNIT_ATOMS = tuple(sorted(set(_PREFIXED_ATOMS + _BASE_ATOMS), key=len, reverse=True))
_UNIT_ATOM = "(?:" + "|".join(re.escape(atom) for atom in _UNIT_ATOMS) + ")"
_UNIT_POWER = r"(?:\s*(?:\^|\*\*)\s*\{?[+-]?\d+\}?)?"
_UNIT_SUFFIX = rf"{_UNIT_ATOM}{_UNIT_POWER}(?:\s*(?:[*/·]|\s)\s*{_UNIT_ATOM}{_UNIT_POWER})*"
_NUMBER_WITH_UNIT_RE = re.compile(
    rf"^\s*(?P<value>{_NUMBER})(?:\s*(?P<unit>{_UNIT_SUFFIX}))?\s*$"
)
_NUMERIC_FRACTION_RE = re.compile(
    rf"^\s*\\[dtc]?frac\{{(?P<numerator>{_NUMBER})\}}"
    rf"\{{(?P<denominator>{_NUMBER})\}}\s*(?P<unit>{_UNIT_SUFFIX})?\s*$"
)
_SCIENTIFIC_RE = re.compile(
    rf"^\s*(?P<coefficient>{_NUMBER})\s*(?:\\times|\\cdot|×|·)\s*"
    r"10\s*\^\s*\{?\s*(?P<exponent>[+-]?\d+)\s*\}?\s*"
    rf"(?:(?P<unit>{_UNIT_SUFFIX}))?\s*$"
)
_RATIO_RE = re.compile(rf"^\s*(?P<left>{_NUMBER})\s*:\s*(?P<right>{_NUMBER})\s*$")
# A unit tail is only a unit when it follows a digit, whitespace, or a closing group:
# "mgh" ends in two unit atoms but is a product of symbols.
_TRAILING_UNIT_RE = re.compile(rf"(?<=[\d\s}})\]])\s*(?P<unit>{_UNIT_SUFFIX})\s*$")
_WRAPPED_UNIT_TAIL_RE = re.compile(
    r"\\(?:mathrm|text|textrm|mathsf|operatorname|mbox)\s*\{(?P<unit>[^{}]*)\}\s*$"
)
_UNIT_ALIASES = {"ohm": "Ω", "kohm": "kΩ", "Mohm": "MΩ", "l": "L", "μm": "µm", "um": "µm",
                 "μg": "µg", "ug": "µg", "μs": "µs", "us": "µs", "μW": "µW", "μV": "µV",
                 "μA": "µA", "μT": "µT", "μL": "µL", "μF": "µF", "uF": "µF", "μH": "µH",
                 "μWb": "µWb"}
# Unprefixed SI derived units expand to base dimensions so V/m ≡ N/C and J ≡ N·m.
# Scaled units (eV, cal, atm, min, h, °, every prefixed atom) stay opaque tokens.
_BASE_DIMENSIONS = {
    "N": {"kg": 1, "m": 1, "s": -2},
    "J": {"kg": 1, "m": 2, "s": -2},
    "W": {"kg": 1, "m": 2, "s": -3},
    "Pa": {"kg": 1, "m": -1, "s": -2},
    "V": {"kg": 1, "m": 2, "s": -3, "A": -1},
    "C": {"A": 1, "s": 1},
    "Ω": {"kg": 1, "m": 2, "s": -3, "A": -2},
    "T": {"kg": 1, "s": -2, "A": -1},
    "Wb": {"kg": 1, "m": 2, "s": -2, "A": -1},
    "F": {"kg": -1, "m": -2, "s": 4, "A": 2},
    "H": {"kg": 1, "m": 2, "s": -2, "A": -2},
    "Hz": {"s": -1},
    "rad": {},
    "sr": {},
}
_WRAPPER_COMMANDS = {
    "mathrm", "mathbf", "boldsymbol", "operatorname", "overline", "text", "textrm",
    "textit", "mathsf", "mathit", "mathcal", "mathbb", "mbox", "vec", "hat", "bar",
    "tilde", "widehat", "widetilde", "underline",
}
_NUMERIC_WRAPPER_RE = re.compile(r"\\(?:mathrm|text|textrm|mathsf|mathit|operatorname|mbox)\s*\{")
_SPACING_COMMANDS = {",", "!", ";", ":", " ", "quad", "qquad", "displaystyle"}
_LABEL_RE = re.compile(
    r"^\s*(?:\\text\{[^{}]*\}\s*[:：]?\s*)?"
    r"(?P<label>\\?[A-Za-z][A-Za-z0-9]*(?:_(?:\{[^{}]*\}|[A-Za-z0-9]))*(?:\^\{?\\prime\}?|')*"
    r"(?:\([A-Za-z0-9,\s]*\))?)\s*(?:=|\\approx|≈|\\simeq)\s*(?=\S)"
)
_TEXT_LABEL_RE = re.compile(r"^\s*\\text\{[^{}]*:\s*\}\s*(?=\S)")

_SAFE_INFIX_RE = re.compile(r"^[A-Za-z0-9_+*/^().,=\s-]+$")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication, convert_xor)
_COMMAND_NAMES = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    # var-forms denote the same physical symbol (\varepsilon_0 is \epsilon_0);
    # rl/symbolic_match canonicalises them the same way.
    "varepsilon": "epsilon",
    "zeta": "zeta",
    "eta": "eta",
    "theta": "theta",
    "vartheta": "theta",
    "kappa": "kappa",
    "lambda": "lam",
    "mu": "mu",
    "nu": "nu",
    "xi": "xi",
    "rho": "rho",
    "sigma": "sigma",
    "tau": "tau",
    "phi": "phi",
    "varphi": "phi",
    "chi": "chi",
    "psi": "psi",
    "omega": "omega",
    "Omega": "Omega",
    "Delta": "Delta",
    "Phi": "Phi",
    "Gamma": "Gamma",
    "Theta": "Theta",
    "Lambda": "Lambda",
    "Sigma": "Sigma",
    "Psi": "Psi",
    "pi": "pi",
    "hbar": "hbar",
    "ell": "l",
    "infty": "oo",
    "circ": "circ",
}
_UNICODE_SYMBOLS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "κ": "kappa",
    "λ": "lam",
    "μ": "mu",
    "µ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Ω": "Omega",
    "Δ": "Delta",
    "Φ": "Phi",
    "π": "pi",
}
_FUNCTION_COMMANDS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "ln": "log",
    "log": "log",
}
_KNOWN_FUNCTIONS = set(_FUNCTION_COMMANDS.values()) | {"sqrt"}
_KEEP_IDENTIFIERS = (
    set(_COMMAND_NAMES.values()) | _KNOWN_FUNCTIONS | {"prime", "deg", "oo", "E", "I"}
)
_FUNCTION_PREFIXES = sorted(_KNOWN_FUNCTIONS, key=len, reverse=True)
_GREEK_COMMANDS = sorted(
    (command for command in _COMMAND_NAMES if command not in {"hbar", "infty", "circ"}),
    key=len, reverse=True,
)
_GREEK_PREFIXES = sorted(
    (set(_COMMAND_NAMES.values()) - {"pi", "oo", "circ"}) | {"epsilon", "theta", "phi"},
    key=len, reverse=True,
)
_SYMBOLIC_TIMEOUT_SECONDS = 2.0
_SAMPLING_REL_TOL = 1e-6
_MAX_INPUT_CHARS = 4096
_MAX_COMPONENTS = 16
_MAX_SYMPY_OPS = 512
_MATCH_EPS = 1e-9


class _EquivalenceTimeout(TimeoutError):
    pass


@contextmanager
def _time_limit(seconds: float):
    """Hard-limit SymPy work on Unix main threads; preserve an outer timer."""

    can_interrupt = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    if not can_interrupt:
        started = time.monotonic()
        yield
        if time.monotonic() - started > seconds:
            raise _EquivalenceTimeout("symbolic comparison exceeded its deadline")
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def _raise_timeout(_signum, _frame):
        raise _EquivalenceTimeout("symbolic comparison exceeded its deadline")

    signal.signal(signal.SIGALRM, _raise_timeout)
    deadline = seconds
    if previous_timer[0] > 0:
        deadline = min(deadline, previous_timer[0])
    signal.setitimer(signal.ITIMER_REAL, deadline)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(previous_timer[0] - elapsed, 1e-6),
                previous_timer[1],
            )


# --- shared text normalisation ----------------------------------------------------------


def _braced(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise ValueError("expected braced LaTeX group")
    depth = 1
    index = start + 1
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth:
        raise ValueError("unbalanced LaTeX braces")
    return text[start + 1 : index - 1], index


def _group(text: str, start: int) -> tuple[str, int]:
    """A braced group, or -- as TeX allows -- the single following token (``\\frac12``)."""

    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text) and text[start] == "{":
        return _braced(text, start)
    command = re.match(r"\\[A-Za-z]+", text[start:])
    if command:
        return command.group(0), start + len(command.group(0))
    if start < len(text) and (text[start].isalnum() or text[start] == "."):
        return text[start], start + 1
    raise ValueError("expected a LaTeX argument")


def _strip_delimiters(text: str) -> str:
    stripped = text.strip()
    for opening, closing in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if (
            stripped.startswith(opening)
            and stripped.endswith(closing)
            and len(stripped) > len(opening) + len(closing)
        ):
            return stripped[len(opening) : -len(closing)].strip()
    return stripped


def _strip_wrappers(text: str) -> str:
    """Remove ``\\mathrm{...}``-style wrappers with balanced braces, keeping the content."""

    while True:
        match = _NUMERIC_WRAPPER_RE.search(text)
        if not match:
            return text
        try:
            content, end = _braced(text, match.end() - 1)
        except ValueError:
            return text
        # No padding: "\mathrm{v}_{0}" must stay "v_{0}", and adjacent wrappers
        # ("\mathrm{m}\mathrm{s}^{2}") read the same as the unwrapped form.
        text = text[: match.start()] + content + text[end:]


def _strip_label(text: str) -> str:
    """D5: ``v = 6 m/s`` names the quantity; the answer is the value."""

    match = _LABEL_RE.match(text)
    if match:
        return text[match.end():]
    match = _TEXT_LABEL_RE.match(text)
    if match:
        return text[match.end():]
    return text


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unit.replace("μ", "µ")
    normalized = re.sub(r"\s*(?:\*\*|\^)\s*\{?([+-]?\d+)\}?", r"^\1", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized.strip())
    normalized = re.sub(r"\s*[*·]\s*|\s+", "*", normalized)
    return normalized


def _unit_dimensions(unit: str | None) -> dict[str, int] | None:
    """Exponent vector of a normalised unit string, derived units expanded to SI base."""

    if unit is None:
        return None
    dimensions: dict[str, int] = {}
    sign = 1
    for piece in re.split(r"(/)", unit):
        if piece == "/":
            sign = -1
            continue
        for token in filter(None, piece.split("*")):
            match = re.fullmatch(rf"({_UNIT_ATOM})(?:\^([+-]?\d+))?", token)
            if not match:
                return None
            atom = _UNIT_ALIASES.get(match.group(1), match.group(1))
            power = sign * int(match.group(2) or 1)
            for base, exponent in _BASE_DIMENSIONS.get(atom, {atom: 1}).items():
                dimensions[base] = dimensions.get(base, 0) + power * exponent
    return {base: exponent for base, exponent in dimensions.items() if exponent}


def _units_compatible(left: str | None, right: str | None) -> bool:
    """Both units present and dimensionally distinct fails closed; a missing unit is permissive."""

    if left is None or right is None:
        return True
    if left == right:
        return True
    left_dimensions = _unit_dimensions(left)
    right_dimensions = _unit_dimensions(right)
    if left_dimensions is None or right_dimensions is None:
        return False
    return left_dimensions == right_dimensions


def _numeric_source(text: str) -> str:
    """Bring a numeric answer into the grammar the regex parsers read."""

    source = _strip_delimiters(text)
    source = _strip_wrappers(source)
    source = (
        source.replace("−", "-")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("℃", "°C")
        .replace(r"\%", "%")
        .replace(r"\Omega", "Ω")
        .replace(r"\mu", "µ")
        .replace(r"\cdot", "·")
        .replace(r"\degree", "°")
    )
    source = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", source)
    source = re.sub(r"\s*°\s*C\b", "°C", source)
    source = re.sub(r"\b(?:degrees|deg)\b", "°", source)
    source = re.sub(r"\\(?:,|;|!|:| |quad|qquad)", " ", source)
    source = source.replace("~", " ")
    return source.strip()


def _numeric_with_unit(text: str) -> tuple[float, str | None] | None:
    """Regex-fast numeric parse: number, fraction, scientific, or ratio, with an optional unit.

    LaTeX math delimiters wrap the value, they are not part of it. PhysReason golds
    arrive as ``$2\\mathrm{m}/\\mathrm{s}$``; with the ``$`` left on, the numeric path was
    skipped and correct bare-number answers fell through to a symbolic comparison that
    rejected them -- measured 2026-08-31 at ~61% of the base model's correct answers.
    """

    source = _numeric_source(text)
    scientific = _SCIENTIFIC_RE.fullmatch(source)
    if scientific:
        value = float(scientific.group("coefficient")) * 10 ** int(scientific.group("exponent"))
        if not math.isfinite(value):
            return None
        return value, _normalize_unit(scientific.group("unit"))
    fraction = _NUMERIC_FRACTION_RE.fullmatch(source)
    if fraction:
        denominator = float(fraction.group("denominator"))
        if denominator == 0:
            return None
        value = float(fraction.group("numerator")) / denominator
        return (value, _normalize_unit(fraction.group("unit"))) if math.isfinite(value) else None
    ratio = _RATIO_RE.fullmatch(source)
    if ratio:
        denominator = float(ratio.group("right"))
        if denominator == 0:
            return None
        return float(ratio.group("left")) / denominator, None
    # Braces that merely group a unit ("10{m}/{s}") are not part of the grammar.
    plain = re.sub(r"\{([A-Za-zµΩ°%/·*\s^\-\d]*)\}", r"\1", source).replace(",", "")
    number = _NUMBER_WITH_UNIT_RE.fullmatch(plain)
    if not number:
        return None
    value = float(number.group("value"))
    if not math.isfinite(value):
        return None
    return value, _normalize_unit(number.group("unit"))


def _numeric_value(text: str) -> float | None:
    parsed = _numeric_with_unit(text)
    return parsed[0] if parsed else None


def _split_trailing_unit(text: str) -> tuple[str, str | None]:
    match = _TRAILING_UNIT_RE.search(text)
    if match and text[: match.start()].strip():
        return text[: match.start()].strip(), _normalize_unit(match.group("unit"))
    return text.strip(), None


def _constant_with_unit(text: str) -> tuple[float, str | None] | None:
    """Constant expression with an optional trailing unit: ``2\\sqrt{5} m/s``, ``\\pi^2 W``."""

    source = _numeric_source(text)
    for expression, unit in (
        (source, None),
        _split_trailing_unit(source),
    ):
        if not expression or "__" in expression:
            continue
        try:
            with _time_limit(_SYMBOLIC_TIMEOUT_SECONDS):
                value = _constant_numeric_value(expression)
        except Exception:
            value = None
        if value is not None:
            return value, unit
    return None


def _parse_numeric(text: str) -> tuple[float, str | None] | None:
    labelled = _strip_label(_strip_delimiters(text))
    return _numeric_with_unit(labelled) or _constant_with_unit(labelled)


def _split_wrapped_unit(text: str) -> tuple[str, str | None]:
    """Symbolic answers shed an explicit upright unit tail (``\\frac{\\lambda L}{8d}\\mathrm{m}``).

    Only wrapper-marked tails are units here: a bare trailing ``m`` on a symbolic
    expression is far more often a mass than a metre.
    """

    text = _strip_delimiters(text)
    match = _WRAPPED_UNIT_TAIL_RE.search(text)
    if not match:
        return text, None
    unit = _numeric_source(match.group("unit"))
    if not re.fullmatch(rf"\s*{_UNIT_SUFFIX}\s*", unit):
        return text, None
    head = text[: match.start()].strip()
    if not head:
        return text, None
    return head, _normalize_unit(unit)


# --- LaTeX to SymPy ---------------------------------------------------------------------


def _split_identifier(name: str) -> str:
    """Physics writes products without operators: ``mgh`` is m·g·h, ``mv_0`` is m·v_0.

    Greek names, functions and subscripted labels survive whole; other purely alphabetic
    runs become single-letter symbols, function names peeled off the front (``Gcos``).
    Digits inside a subscript end it when letters follow (``v_0B`` is v_0·B).
    """

    if name in _KEEP_IDENTIFIERS or "__" in name or "_at_" in name:
        return name
    base, underscore, subscript = name.partition("_")
    if not re.fullmatch(r"[A-Za-z]+", base):
        return name
    if not underscore:
        # rl/symbolic_match fuses a Greek base with its subscript into one identifier
        # ("epsilon0", "thetan") so that "epsilon 0" cannot collapse to zero; read the
        # tail as the subscript it is rather than as glued symbols.
        for greek in _GREEK_PREFIXES:
            if base.startswith(greek) and len(base) > len(greek):
                tail = base[len(greek):]
                if re.fullmatch(r"[A-Za-z0-9]{1,4}", tail) and not tail.startswith(tuple(_FUNCTION_PREFIXES)):
                    return f"{greek}_{tail}"
                break
    trailing = ""
    if underscore:
        digits_then_letters = re.fullmatch(r"(\d+)([A-Za-z].*)", subscript)
        if digits_then_letters:
            subscript, trailing = digits_then_letters.group(1), digits_then_letters.group(2)
        if not re.fullmatch(r"[A-Za-z0-9]+", subscript):
            return name
    if base in _KEEP_IDENTIFIERS:
        pieces = [base]
    else:
        pieces = []
        index = 0
        while index < len(base):
            for function in _FUNCTION_PREFIXES:
                if base.startswith(function, index):
                    pieces.append(function)
                    index += len(function)
                    break
            else:
                pieces.append(base[index])
                index += 1
    if underscore:
        pieces[-1] = f"{pieces[-1]}_{subscript}"
    result = "*".join(pieces)
    if trailing:
        result = f"{result}*{_split_identifier(trailing)}"
    return result


def _latex_to_infix(text: str) -> str:
    """Convert the bounded PhysR1Corp LaTeX subset to safe SymPy infix."""

    source = (
        text.strip()
        .replace("−", "-")
        .replace("×", r"\times")
        .replace("·", r"\cdot")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace(r"\left", "")
        .replace(r"\right", "")
        .replace("~", " ")
    )
    for character, name in _UNICODE_SYMBOLS.items():
        source = source.replace(character, f" {name} ")
    source = re.sub(r"^\$+|\$+$", "", source).strip()
    source = re.sub(r"\^\s*\{?\s*\\prime\s*\}?|'", "_prime", source)
    output: list[str] = []
    index = 0
    while index < len(source):
        if source[index] != "\\":
            output.append(source[index])
            index += 1
            continue

        command_match = re.match(r"\\([A-Za-z]+|[,!;: ])", source[index:])
        if not command_match:
            raise ValueError("unsupported LaTeX command")
        command = command_match.group(1)
        index += len(command_match.group(0))
        # \dfrac, \tfrac and \cfrac are display/text/continued-fraction variants of
        # \frac with identical semantics. Models emit them constantly and gold authors
        # never do: measured 2026-08-31 over PhysReason, 17.5% of the base model's boxed
        # answers use \dfrac against 0% of golds. Unrecognised, every such answer scored
        # wrong however correct it was -- arm F learned to stop writing \dfrac.
        if command in {"dfrac", "tfrac", "cfrac"}:
            command = "frac"

        if command in _SPACING_COMMANDS:
            continue
        if command == "frac":
            numerator, index = _group(source, index)
            denominator, index = _group(source, index)
            output.append(
                f"(({_latex_to_infix(numerator)})/({_latex_to_infix(denominator)}))"
            )
            continue
        if command == "sqrt":
            while index < len(source) and source[index].isspace():
                index += 1
            if index < len(source) and source[index] == "[":
                close = source.find("]", index)
                if close < 0:
                    raise ValueError("unterminated root index")
                degree = source[index + 1 : close]
                index = close + 1
                radicand, index = _group(source, index)
                output.append(
                    f"(({_latex_to_infix(radicand)}))**(1/({_latex_to_infix(degree)}))"
                )
                continue
            radicand, index = _group(source, index)
            output.append(f"sqrt(({_latex_to_infix(radicand)}))")
            continue
        if command in _WRAPPER_COMMANDS:
            content, index = _group(source, index)
            output.append(_latex_to_infix(content))
            continue
        if command in ("times", "cdot"):
            output.append("*")
            continue
        if command in ("approx", "simeq"):
            output.append("=")
            continue
        if command in _FUNCTION_COMMANDS:
            output.append(f" {_FUNCTION_COMMANDS[command]}")
        elif command in _COMMAND_NAMES:
            output.append(f" {_COMMAND_NAMES[command]} ")
        else:
            # "\pir^2" is a Greek command glued to the symbol that follows it (gold
            # cal_problem_00652#2); an unknown command with a Greek prefix splits there.
            for greek in _GREEK_COMMANDS:
                if command.startswith(greek) and re.fullmatch(r"[A-Za-z]+", command[len(greek):]):
                    output.append(f" {_COMMAND_NAMES[greek]} {command[len(greek):]}")
                    break
            else:
                # No trailing space: a stray backslash on a symbol ("\v_{0}", gold
                # cal_problem_01562#0) must keep its subscript attached.
                output.append(f" {command}")

    infix = "".join(output)
    infix = re.sub(
        r"_\{([^{}]+)\}", lambda match: "_" + re.sub(r"\W", "", match.group(1)) + " ", infix
    )
    infix = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", infix)
    infix = re.sub(r"\^([+-]?\d+)", r"**\1", infix)
    infix = (
        infix.replace("{", "(")
        .replace("}", ")")
        .replace("[", "(")
        .replace("]", ")")
    )
    # A digit may precede an identifier ("2mgh"): no \b there, so use a lookbehind.
    infix = re.sub(
        r"(?<![A-Za-z_])[A-Za-z][A-Za-z0-9_]*",
        lambda match: _split_identifier(match.group(0)),
        infix,
    )
    # e raised to anything but an integer is the exponential; e squared is the charge.
    infix = re.sub(
        r"(?<![A-Za-z_])e\*\*\((?P<power>[^()]*)\)",
        lambda match: match.group(0)
        if re.fullmatch(r"\s*[+-]?\d+\s*", match.group("power"))
        else f"exp({match.group('power')})",
        infix,
    )
    functions = "|".join(_KNOWN_FUNCTIONS)
    start = r"(?<![A-Za-z_])"
    infix = re.sub(
        rf"{start}(?P<function>{functions})\s*\*\*\s*\((?P<power>[^()]*)\)\s*"
        r"(?P<argument>[A-Za-z][A-Za-z0-9_]*)",
        r"(\g<function>(\g<argument>))**(\g<power>)",
        infix,
    )
    infix = re.sub(
        rf"{start}(?P<function>{functions})\s*\*\*(?P<power>[+-]?\d+)\s*"
        r"(?P<argument>[A-Za-z][A-Za-z0-9_]*)",
        r"(\g<function>(\g<argument>))**(\g<power>)",
        infix,
    )
    infix = re.sub(rf"{start}(?P<function>{functions})\*(?=[A-Za-z(])", r"\g<function> ", infix)
    infix = re.sub(
        rf"{start}(?P<function>{functions})\s+(?P<argument>[A-Za-z][A-Za-z0-9_]*)",
        r"\g<function>(\g<argument>)",
        infix,
    )
    return infix.strip()


def _normalized_string(text: str) -> str:
    """Whitespace- and product-insensitive canonical form for string comparison.

    The identifier splitter writes products explicitly (``mgh`` -> ``m*g*h``) while a
    spaced source (``m g h``) never gets a ``*``; both must canonicalise to ``mgh``.
    """

    try:
        infix = _latex_to_infix(text).replace("**", "^")
        return re.sub(r"[\s*]+", "", infix)
    except Exception:
        return re.sub(r"\s+", "", text.strip())


def _top_level_components(text: str) -> tuple[str, ...]:
    source = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    components: list[str] = []
    start = 0
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    for index, character in enumerate(source):
        if character in pairs:
            depth += 1
        elif character in closing:
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced grouping")
        elif (
            character == ","
            and depth == 0
            and (index == 0 or source[index - 1] != "\\")
        ):
            components.append(source[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ValueError("unbalanced grouping")
    components.append(source[start:].strip())
    if any(not component for component in components):
        raise ValueError("empty answer component")
    return tuple(components)


def _parse_symbolic(text: str) -> sympy.Expr:
    infix = _latex_to_infix(text)

    def _atomic_call(match: re.Match) -> str:
        name = match.group("name")
        if name in _KNOWN_FUNCTIONS:
            return match.group(0)
        arguments = re.sub(r"\W+", "_", match.group("arguments")).strip("_")
        return f"{name}_at_{arguments}"

    infix = re.sub(
        r"\b(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*"
        r"\((?P<arguments>[A-Za-z0-9_,\s]+)\)",
        _atomic_call,
        infix,
    )
    if not infix or "__" in infix or not _SAFE_INFIX_RE.fullmatch(infix):
        raise ValueError("expression is outside the supported safe grammar")
    if "=" in infix:
        raise ValueError("equations are handled separately")
    symbols = {
        name: sympy.Symbol(name)
        for name in set(_COMMAND_NAMES.values()) - {"pi", "oo"}
    }
    local_dict = {
        **symbols,
        "pi": sympy.pi,
        "oo": sympy.oo,
        "E": sympy.Symbol("E"),
        "I": sympy.Symbol("I"),
        "sqrt": sympy.sqrt,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "cot": sympy.cot,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "sinh": sympy.sinh,
        "cosh": sympy.cosh,
        "tanh": sympy.tanh,
        "exp": sympy.exp,
        "log": sympy.log,
    }
    global_dict = {
        "Symbol": sympy.Symbol,
        "Integer": sympy.Integer,
        "Float": sympy.Float,
        "Rational": sympy.Rational,
        "Function": sympy.Function,
        "__builtins__": {},
    }
    expression = parse_expr(
        infix,
        local_dict=local_dict,
        global_dict=global_dict,
        transformations=_TRANSFORMATIONS,
        evaluate=True,
    )
    if not isinstance(expression, sympy.Expr):
        raise ValueError("parsed object is not a scalar expression")
    if int(sympy.count_ops(expression)) > _MAX_SYMPY_OPS:
        raise ValueError("expression exceeds the symbolic operation limit")
    return expression


def _parse_equation_lhs(text: str) -> sympy.Expr:
    """Treat indexed function notation on an answer label as one atomic lhs."""

    infix = _latex_to_infix(text)
    function_label = re.fullmatch(
        r"\s*(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*"
        r"\((?P<arguments>[A-Za-z0-9_,\s]+)\)\s*",
        infix,
    )
    if function_label:
        arguments = re.sub(r"\W+", "_", function_label.group("arguments")).strip("_")
        return sympy.Symbol(f"{function_label.group('name')}__{arguments}")
    return _parse_symbolic(infix)


def _constant_numeric_value(text: str) -> float | None:
    expression = _parse_symbolic(text)
    if expression.free_symbols:
        return None
    value = complex(sympy.N(expression, 15))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        return None
    if abs(value.imag) > 1e-12 * max(abs(value.real), 1.0):
        return None
    return float(value.real)


def _sampled_equivalent(predicted: sympy.Expr, expected: sympy.Expr) -> bool:
    """Numerical equivalence on the physical domain: symbols are positive quantities.

    Masses, lengths, speeds and field strengths are positive, so ``v_0\\sqrt{2hR/(x^2-h^2)}``
    and ``\\sqrt{2hv_0^2R/(x^2-h^2)}`` are the same answer (gold cal_problem_01367#1) even
    though they differ for negative ``v_0``. Sign-carrying answers still differ: a constant
    sign flip is a sign flip at every positive point.
    """

    symbols = sorted(predicted.free_symbols | expected.free_symbols, key=str)
    rng = random.Random(1729)
    successful = 0
    for _ in range(12):
        substitutions = {}
        for symbol in symbols:
            substitutions[symbol] = rng.uniform(0.2, 2.0)
        try:
            pred_value = complex(sympy.N(predicted.subs(substitutions), 15))
            gold_value = complex(sympy.N(expected.subs(substitutions), 15))
        except (ArithmeticError, TypeError, ValueError):
            continue
        components = (
            pred_value.real,
            pred_value.imag,
            gold_value.real,
            gold_value.imag,
        )
        if not all(math.isfinite(value) for value in components):
            continue
        if abs(pred_value - gold_value) > _SAMPLING_REL_TOL * max(
            abs(pred_value), abs(gold_value), 1e-12
        ):
            return False
        successful += 1
        if successful == 3:
            return True
    return False


def _sampled_ratio(
    numerator: sympy.Expr, denominator: sympy.Expr
) -> float | None:
    """Return the constant ratio numerator/denominator, or None.

    _sampled_equivalent's body generalized (additive; that function is
    untouched): same seeded RNG, same 12 draws, same 3-success requirement,
    same relative tolerance — but instead of testing equality it tests
    whether the ratio of the two expressions is the SAME finite real
    constant across successful draws. Built for the symbolic gold-leak
    filter, where "milestone = k * gold-side" is a leak for rational k and
    simplify() cannot always find the identity.
    """

    symbols = sorted(numerator.free_symbols | denominator.free_symbols, key=str)
    rng = random.Random(1729)
    successful = 0
    ratio: float | None = None
    for _ in range(12):
        substitutions = {}
        for symbol in symbols:
            value = rng.uniform(-2.0, 2.0)
            if abs(value) < 0.2:
                value = 0.2 if value >= 0 else -0.2
            substitutions[symbol] = value
        try:
            top = complex(sympy.N(numerator.subs(substitutions), 15))
            bottom = complex(sympy.N(denominator.subs(substitutions), 15))
        except (ArithmeticError, TypeError, ValueError):
            continue
        components = (top.real, top.imag, bottom.real, bottom.imag)
        if not all(math.isfinite(value) for value in components):
            continue
        if abs(bottom) < 1e-9:
            continue
        sample = top / bottom
        if abs(sample.imag) > 1e-9 * max(abs(sample.real), 1.0):
            return None
        if ratio is None:
            ratio = sample.real
        elif abs(sample.real - ratio) > _SAMPLING_REL_TOL * max(
            abs(ratio), 1e-12
        ):
            return None
        successful += 1
        if successful == 3:
            return ratio
    return None


def _expressions_equivalent(predicted: sympy.Expr, expected: sympy.Expr) -> bool:
    if sympy.simplify(predicted - expected) == 0:
        return True
    return _sampled_equivalent(predicted, expected)


def _strip_text_label(text: str) -> str:
    """Only the ``\\text{Sub-question 2: }`` form: a variable label is an equation's lhs
    and the equation logic below compares residuals with it."""

    match = _TEXT_LABEL_RE.match(_strip_delimiters(text))
    return _strip_delimiters(text)[match.end():] if match else text


def _symbolically_equivalent(predicted: str, expected: str) -> bool:
    predicted, pred_unit = _split_wrapped_unit(_strip_text_label(predicted))
    expected, gold_unit = _split_wrapped_unit(_strip_text_label(expected))
    if not _units_compatible(pred_unit, gold_unit):
        return False
    pred_infix = _latex_to_infix(predicted)
    gold_infix = _latex_to_infix(expected)
    if pred_infix.count("=") > 1 or gold_infix.count("=") > 1:
        raise ValueError("only scalar equalities are supported")
    pred_equation = pred_infix.split("=", 1) if "=" in pred_infix else None
    gold_equation = gold_infix.split("=", 1) if "=" in gold_infix else None

    if pred_equation and gold_equation:
        pred_residual = _parse_equation_lhs(pred_equation[0]) - _parse_symbolic(
            pred_equation[1]
        )
        gold_residual = _parse_equation_lhs(gold_equation[0]) - _parse_symbolic(
            gold_equation[1]
        )
        return _expressions_equivalent(
            pred_residual, gold_residual
        ) or _expressions_equivalent(pred_residual, -gold_residual)
    if gold_equation:
        return _expressions_equivalent(
            _parse_symbolic(pred_infix), _parse_symbolic(gold_equation[1])
        )
    if pred_equation:
        return _expressions_equivalent(
            _parse_symbolic(pred_equation[1]), _parse_symbolic(gold_infix)
        )
    return _expressions_equivalent(
        _parse_symbolic(pred_infix), _parse_symbolic(gold_infix)
    )


# --- public seams -----------------------------------------------------------------------


def equivalent(pred_boxed: str, gold: str) -> bool:
    """Return whether a boxed prediction is equivalent to one gold answer.

    This public seam never raises for malformed model output; unsupported or
    unparseable inputs fail closed.
    """

    try:
        if not isinstance(pred_boxed, str) or not isinstance(gold, str):
            return False
        if (
            not pred_boxed.strip()
            or not gold.strip()
            or len(pred_boxed) > _MAX_INPUT_CHARS
            or len(gold) > _MAX_INPUT_CHARS
        ):
            return False
        predicted_components = _top_level_components(pred_boxed)
        gold_components = _top_level_components(gold)
        if (
            len(predicted_components) > _MAX_COMPONENTS
            or len(gold_components) > _MAX_COMPONENTS
        ):
            return False
        if len(predicted_components) > 1 or len(gold_components) > 1:
            return len(predicted_components) == len(gold_components) and all(
                equivalent(predicted, expected)
                for predicted, expected in zip(
                    predicted_components, gold_components
                )
            )
        if "__" in pred_boxed or "__" in gold:
            return False
        expected_parsed = _parse_numeric(gold)
        predicted_parsed = _parse_numeric(pred_boxed) if expected_parsed is not None else None
        if (
            expected_parsed is not None
            and predicted_parsed is not None
            and _units_compatible(predicted_parsed[1], expected_parsed[1])
        ):
            # Units are compared, not converted: when BOTH sides carry a unit they must
            # agree dimensionally ("5 cm" vs "5 mm" fails closed, "N/C" vs "V/m" passes);
            # a unit on only one side stays permissive (unitless golds are common and
            # the model usually echoes the problem's unit).
            predicted, expected = predicted_parsed[0], expected_parsed[0]
            return abs(predicted - expected) <= 0.01 * max(abs(expected), 1e-9)
        # A numeric gold against a symbolic prediction, or a unit disagreement, drops to
        # the symbolic comparison: "\frac{5}{3}T" reads as 1.67 tesla numerically but is
        # the same expression as "5T/3", and "\sqrt{2} m g" against "\sqrt{2}\,mg" is one
        # product written two ways. Genuinely different units still fail there ("5 cm"
        # is 5*c*m, "5 mm" is 5*m**2).
        normalized_match = _normalized_string(pred_boxed) == _normalized_string(gold)
        try:
            with _time_limit(_SYMBOLIC_TIMEOUT_SECONDS):
                symbolic_match = _symbolically_equivalent(pred_boxed, gold)
        except Exception:
            return normalized_match
        return symbolic_match or normalized_match
    except Exception:
        return False


def _is_prose(text: str) -> bool:
    """D4: two or more consecutive alphabetic words that are neither LaTeX nor units."""

    source = re.sub(r"\\[A-Za-z]+", " ", _strip_delimiters(text))
    source = re.sub(r"[{}$]", " ", source)
    run = 0
    for token in source.split():
        word = token.rstrip(".,;:")
        if re.fullmatch(r"[A-Za-z]{2,}", word) and word not in _UNIT_ATOMS:
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def gold_scorability(gold: str) -> str:
    """D4/D7: how (whether) the automatic scorer can read a gold.

    ``prose`` golds carry a sign, a direction, or a statement the matcher cannot judge and
    are excluded from the automatic metric and counted; ``unparseable`` golds fail closed
    and are counted; ``numeric`` and ``symbolic`` are scorable.
    """

    try:
        if not isinstance(gold, str) or not gold.strip() or len(gold) > _MAX_INPUT_CHARS:
            return "unparseable"
        if _is_prose(gold):
            return "prose"
        components = _top_level_components(gold)
        if len(components) > _MAX_COMPONENTS:
            return "unparseable"
        kinds = set()
        for component in components:
            if _parse_numeric(component) is not None:
                kinds.add("numeric")
                continue
            if "__" in component:
                return "unparseable"
            try:
                with _time_limit(_SYMBOLIC_TIMEOUT_SECONDS):
                    expression, _ = _split_wrapped_unit(_strip_text_label(component))
                    infix = _latex_to_infix(expression)
                    if "=" in infix:
                        lhs, rhs = infix.split("=", 1)
                        _parse_equation_lhs(lhs)
                        _parse_symbolic(rhs)
                    else:
                        _parse_symbolic(infix)
            except Exception:
                return "unparseable"
            kinds.add("symbolic")
        return "symbolic" if "symbolic" in kinds else "numeric"
    except Exception:
        return "unparseable"


def match_fraction(candidates: Sequence[str], gold: str) -> float:
    """D3: the fraction of a gold's top-level components that some candidate matches.

    A candidate matching the whole gold positionally scores 1.0. Otherwise each gold
    component is credited if any candidate -- or any component of a multi-valued
    candidate -- is equivalent to it. Single-valued golds reduce to ``equivalent``.
    """

    try:
        if not isinstance(gold, str) or not gold.strip() or len(gold) > _MAX_INPUT_CHARS:
            return 0.0
        pool: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            if len(candidate) > _MAX_INPUT_CHARS:
                continue
            if equivalent(candidate, gold):
                return 1.0
            pool.append(candidate)
        gold_components = _top_level_components(gold)
        if len(gold_components) == 1 or len(gold_components) > _MAX_COMPONENTS:
            return 0.0
        pieces: list[str] = []
        for candidate in pool:
            try:
                pieces.extend(_top_level_components(candidate))
            except ValueError:
                pieces.append(candidate)
        pieces = list(dict.fromkeys(pieces))[: _MAX_COMPONENTS * 4]
        credited = sum(
            1 for component in gold_components
            if any(equivalent(piece, component) for piece in pieces)
        )
        return credited / len(gold_components)
    except Exception:
        return 0.0


__all__ = ["equivalent", "gold_scorability", "match_fraction"]
