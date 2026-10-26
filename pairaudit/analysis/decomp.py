"""The accounting: an RL model's change over its base at one budget, split exactly into parts.

Per cell (one recipe, one benchmark, one budget), with n problems and k samples each, a
sample is *finished* when it stopped on its own within the budget and *solved* when it
is finished and the matcher gives full credit. Two exact splits are used:

Two-part (``two_part``), on the capped score (partial credit, unfinished = 0):
    change = F + P,  F = (fin_RL - fin_base) * acc_RL,  P = fin_base * (acc_RL - acc_base)
with acc the mean score over a model's finished samples. F is the *finishing term* (the
RL model finishes more or fewer derivations), P the *reliability term* (its finished
derivations are right more or less often).

Three-part (``three_part``), on full-credit pass@1 (a sample counts when solved):
    change = F + P' + reach'
where *reach* is the RL model's solve rate on U, the problems the base finished every
sample of and never solved (nothing the base's samples contain), its *mirror* is the
base's solve rate on U', the RL model's never-solved set, F and P are the two-part split
on the problems outside U, P' = P + mirror and reach' = reach - mirror (the mirror's loss
sits inside the reliability term, so it is moved to reach as the control). Robust reach
counts only problems the RL model solves in at least four of eight samples.

Two more exact splits bound the arbitrariness of where the cross term goes: symmetric
(F_s = dfin * mean acc, P_s = mean fin * dacc) and matched (P_m takes the accuracy change
only on problems both models finish in at least seven of eight samples; F_m the rest).

Every function takes per-problem weights ``w`` so a bootstrap can resample problems by
counting them, and the problem-level arrays are built once per cell (``arrays``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Rows = dict  # (kind, id) -> list of rows, as RowStore.load returns

FULL = 1 - 1e-9  # a score at or above this is full credit
ROBUST_MIN = 4  # robust reach: solved in at least this many samples
MATCHED_MIN_FIN = 7  # matched split: both models finish at least this many of their samples


def solved(r) -> bool:
    return r["score"] >= FULL


@dataclass
class Cell:
    """Per-problem counts for one (base, RL, benchmark) cell; every statistic is a weighted sum over these."""

    ids: list[str]
    nb: np.ndarray  # samples per problem, base
    nr: np.ndarray  # samples per problem, RL
    bf: np.ndarray  # finished samples, base
    rf: np.ndarray  # finished samples, RL
    bs: np.ndarray  # solved samples (full credit), base
    rs: np.ndarray  # solved samples, RL
    bm: np.ndarray  # summed partial-credit score, base
    rm: np.ndarray  # summed partial-credit score, RL
    bsf: np.ndarray  # summed partial credit over finished samples, base
    rsf: np.ndarray  # summed partial credit over finished samples, RL
    bl: np.ndarray  # summed token length, base
    rl: np.ndarray  # summed token length, RL

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def never(self) -> np.ndarray:
        """U: the base finished every sample and solved none."""
        return (self.bf == self.nb) & (self.bs == 0)

    @property
    def never_rl(self) -> np.ndarray:
        """U': the RL model finished every sample and solved none."""
        return (self.rf == self.nr) & (self.rs == 0)

    @property
    def matched(self) -> np.ndarray:
        return (self.bf >= MATCHED_MIN_FIN) & (self.rf >= MATCHED_MIN_FIN)

    @property
    def any_solved_base(self) -> np.ndarray:
        return self.bs >= 1

    @property
    def any_solved_rl(self) -> np.ndarray:
        return self.rs >= 1

    def ones(self) -> np.ndarray:
        return np.ones(self.n)


def arrays(base: Rows, rl: Rows, kind: str | None = None) -> Cell | None:
    """Build the per-problem arrays over the problems both models were read on (None if there are none)."""
    keys = sorted(k for k in base if k in rl and (kind is None or k[0] == kind))
    if not keys:
        return None

    def agg(M, f):
        return np.array([sum(f(r) for r in M[k]) for k in keys], dtype=float)

    return Cell(
        ids=[k[1] for k in keys],
        nb=agg(base, lambda r: 1), nr=agg(rl, lambda r: 1),
        bf=agg(base, lambda r: bool(r["finished"])), rf=agg(rl, lambda r: bool(r["finished"])),
        bs=agg(base, solved), rs=agg(rl, solved),
        bm=agg(base, lambda r: r["score"]), rm=agg(rl, lambda r: r["score"]),
        bsf=agg(base, lambda r: r["score"] if r["finished"] else 0.0), rsf=agg(rl, lambda r: r["score"] if r["finished"] else 0.0),
        bl=agg(base, lambda r: r["len"]), rl=agg(rl, lambda r: r["len"]),
    )


def _W(w: np.ndarray, x: np.ndarray, mask: np.ndarray | None = None) -> float:
    return float((w * x).sum()) if mask is None else float((w * x * mask).sum())


def rates(c: Cell, w: np.ndarray | None = None) -> dict:
    """Whole-cell rates (partial credit): capped mean, finished rate, score given finished, any-solve, mean length."""
    w = c.ones() if w is None else w
    nb, nr = _W(w, c.nb), _W(w, c.nr)
    bf, rf = _W(w, c.bf), _W(w, c.rf)
    npb = float(w.sum())
    return {
        "base": {"mean": _W(w, c.bm) / nb, "finished": bf / nb, "sgf": (_W(w, c.bsf) / bf) if bf else 0.0,
                 "any": _W(w, c.any_solved_base.astype(float)) / npb, "len": _W(w, c.bl) / nb},
        "rl": {"mean": _W(w, c.rm) / nr, "finished": rf / nr, "sgf": (_W(w, c.rsf) / rf) if rf else 0.0,
               "any": _W(w, c.any_solved_rl.astype(float)) / npb, "len": _W(w, c.rl) / nr},
    }


def matched_sgf_delta(c: Cell, w: np.ndarray | None = None) -> tuple[float, int] | None:
    """Change in score given finished on problems both models finish >= 7 of 8 samples; (delta, count) or None."""
    w = c.ones() if w is None else w
    m = c.matched
    bf, rf = _W(w, c.bf, m), _W(w, c.rf, m)
    if bf <= 0 or rf <= 0:
        return None
    return _W(w, c.rsf, m) / rf - _W(w, c.bsf, m) / bf, int((w * m).sum())


def two_part(c: Cell, w: np.ndarray | None = None) -> dict:
    """The capped-score split over all problems: d = F + P with partial credit; plus the wins-alone counts."""
    w = c.ones() if w is None else w
    r = rates(c, w)
    fb, fr, ab, ar = r["base"]["finished"], r["rl"]["finished"], r["base"]["sgf"], r["rl"]["sgf"]
    mb, mr = r["base"]["mean"], r["rl"]["mean"]
    rl_only = c.any_solved_rl & ~c.any_solved_base
    base_only = c.any_solved_base & ~c.any_solved_rl
    return {
        "F": (fr - fb) * ar, "P": fb * (ar - ab), "d": mr - mb,
        "fb": fb, "fr": fr, "ab": ab, "ar": ar, "base_mean": mb, "rl_mean": mr, "n": int(w.sum()),
        "rl_only": int((w * rl_only).sum()), "base_only": int((w * base_only).sum()),
        "rl_only_base_all_finished": int((w * rl_only * (c.bf == c.nb)).sum()),
        "base_only_rl_all_finished": int((w * base_only * (c.rf == c.nr)).sum()),
    }


def three_part(c: Cell, w: np.ndarray | None = None, k: int = 8) -> dict:
    """The full-credit split with reach: F, P' (mirror added back), reach' (net of the mirror), and the variants."""
    w = c.ones() if w is None else w
    n = float(w.sum())
    C = ~c.never
    reach = _W(w, c.rs, c.never) / (k * n)
    mirror = _W(w, c.bs, c.never_rl) / (k * n)
    reach_robust = _W(w, c.rs * (c.rs >= ROBUST_MIN), c.never) / (k * n)
    mirror_robust = _W(w, c.bs * (c.bs >= ROBUST_MIN), c.never_rl) / (k * n)
    fb = _W(w, c.bf, C) / max(_W(w, c.nb, C), 1e-12)
    fr = _W(w, c.rf, C) / max(_W(w, c.nr, C), 1e-12)
    ab = _W(w, c.bs, C) / max(_W(w, c.bf, C), 1e-12)
    ar = _W(w, c.rs, C) / max(_W(w, c.rf, C), 1e-12)
    wc = _W(w, np.ones(c.n), C) / n
    F = (fr - fb) * ar * wc
    P = fb * (ar - ab) * wc
    Fs = (fr - fb) * (ar + ab) / 2 * wc
    Ps = (fb + fr) / 2 * (ar - ab) * wc
    Mm = C & c.matched
    if _W(w, c.bf, Mm) > 0 and _W(w, c.rf, Mm) > 0:
        Pm = fb * (_W(w, c.rs, Mm) / _W(w, c.rf, Mm) - _W(w, c.bs, Mm) / _W(w, c.bf, Mm)) * wc
    else:
        Pm = P
    Fm = F + P - Pm
    d = (_W(w, c.rs) - _W(w, c.bs)) / (k * n)
    return {
        "d": d, "F": F, "P": P + mirror, "reach": reach - mirror,
        "reach_raw": reach, "mirror": mirror, "reach_robust": reach_robust - mirror_robust,
        "reach_robust_raw": reach_robust, "mirror_robust": mirror_robust,
        "F_sym": Fs, "P_sym": Ps + mirror, "F_matched": Fm, "P_matched": Pm + mirror,
        "fin_b": _W(w, c.bf) / _W(w, c.nb), "fin_r": _W(w, c.rf) / _W(w, c.nr),
        "acc_b": _W(w, c.bs) / max(_W(w, c.bf), 1e-12), "acc_r": _W(w, c.rs) / max(_W(w, c.rf), 1e-12),
        "U": int((w * c.never).sum()), "Up": int((w * c.never_rl).sum()),
        "rl_solved_on_U": int((w * c.never * c.any_solved_rl).sum()),
        "base_solved_on_Up": int((w * c.never_rl * c.any_solved_base).sum()),
        "n": int(n),
    }


def pool_shares(parts: list[dict], split: str = "") -> dict:
    """Pooled shares over cells. size: |F|, |P'|, max(0, sum reach') over their total; signed: each over the net change."""
    fk, pk = ("F", "P") if not split else (f"F_{split}", f"P_{split}")
    F = np.array([p[fk] for p in parts])
    P = np.array([p[pk] for p in parts])
    Rn = np.array([p["reach"] for p in parts])
    net_reach = max(0.0, float(Rn.sum()))
    tot = float(abs(F).sum() + abs(P).sum()) + net_reach
    net = float(F.sum() + P.sum() + Rn.sum())
    robust = min(net_reach, max(0.0, float(sum(p["reach_robust"] for p in parts))))
    return {
        "size": (float(abs(F).sum()) / tot, float(abs(P).sum()) / tot, net_reach / tot),
        "signed": (float(F.sum()) / net, float(P.sum()) / net, float(Rn.sum()) / net),
        "reach_robust": robust / tot,
    }
