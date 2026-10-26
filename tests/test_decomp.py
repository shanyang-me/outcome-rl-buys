"""The accounting identities hold on synthetic rows, and the parts mean what the docstrings say."""

from __future__ import annotations

import random

import numpy as np
import pytest

from pairaudit.analysis.decomp import arrays, matched_sgf_delta, pool_shares, rates, three_part, two_part
from pairaudit.analysis.rows import group_by_problem, read_at


def synthetic(seed: int, n_problems: int = 60, k: int = 8, cap: int = 16384):
    """Two models on the same problems: random finishing, partial-credit scores, lengths; some problems never solved."""
    rng = random.Random(seed)
    rows = {"base": [], "rl": []}
    for i in range(n_problems):
        hard = rng.random() < 0.3
        for who, p_fin, p_ok in (("base", 0.93, 0.4), ("rl", 0.97, 0.5)):
            for j in range(k):
                length = rng.randint(500, int(1.05 * cap))
                fin = rng.random() < p_fin and length <= cap
                score = 0.0
                if fin and not hard:
                    score = rng.choice([0.0, 0.5, 1.0]) if rng.random() < p_ok else 0.0
                rows[who].append({"id": f"p{i}", "kind": "phys", "k": j, "finished": fin, "score": score, "len": min(length, cap)})
    return group_by_problem(rows["base"]), group_by_problem(rows["rl"])


@pytest.mark.parametrize("seed", range(5))
def test_two_part_is_exact_on_the_capped_score(seed):
    c = arrays(*synthetic(seed))
    p = two_part(c)
    assert p["F"] + p["P"] == pytest.approx(p["d"], abs=1e-12)
    assert p["d"] == pytest.approx(p["rl_mean"] - p["base_mean"], abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_three_part_is_exact_under_every_split(seed):
    c = arrays(*synthetic(seed))
    p = three_part(c)
    for F, P in (("F", "P"), ("F_sym", "P_sym"), ("F_matched", "P_matched")):
        assert p[F] + p[P] + p["reach"] == pytest.approx(p["d"], abs=1e-12), (F, P)
    assert p["reach"] == pytest.approx(p["reach_raw"] - p["mirror"])
    assert 0 <= p["reach_robust_raw"] <= p["reach_raw"] + 1e-12


def test_three_part_with_weights_equals_three_part_on_duplicated_problems():
    B, R = synthetic(7)
    c = arrays(B, R)
    rng = np.random.default_rng(3)
    w = np.bincount(rng.integers(0, c.n, c.n), minlength=c.n).astype(float)
    dup_B, dup_R = {}, {}
    for key, count in zip(sorted(B), w):
        for t in range(int(count)):
            dup_B[(key[0], f"{key[1]}#{t}")] = B[key]
            dup_R[(key[0], f"{key[1]}#{t}")] = R[key]
    a, b = three_part(c, w), three_part(arrays(dup_B, dup_R))
    for k in ("d", "F", "P", "reach", "F_sym", "P_sym", "F_matched", "P_matched"):
        assert a[k] == pytest.approx(b[k], abs=1e-12), k


def test_reach_counts_only_problems_the_base_finished_and_never_solved():
    B, R = synthetic(11)
    c = arrays(B, R)
    U = {c.ids[i] for i in np.flatnonzero(c.never)}
    for pid in U:
        rows = B[("phys", pid)]
        assert all(r["finished"] for r in rows) and all(r["score"] < 1 for r in rows)
    p = three_part(c)
    expected = sum(r["score"] >= 1 - 1e-9 for pid in U for r in R[("phys", pid)]) / (8 * c.n)
    assert p["reach_raw"] == pytest.approx(expected)


def test_matched_delta_uses_only_problems_both_finish_seven_of_eight():
    B, R = synthetic(5)
    c = arrays(B, R)
    m = matched_sgf_delta(c)
    assert m is not None
    both = [pid for pid in c.ids if sum(r["finished"] for r in B[("phys", pid)]) >= 7 and sum(r["finished"] for r in R[("phys", pid)]) >= 7]
    assert m[1] == len(both)


def test_pool_shares_sum_to_one_and_signed_shares_sum_to_one():
    parts = [three_part(arrays(*synthetic(s))) for s in range(4)]
    for split in ("", "sym", "matched"):
        p = pool_shares(parts, split)
        assert sum(p["size"]) == pytest.approx(1.0)
        assert sum(p["signed"]) == pytest.approx(1.0)


def test_read_at_tighter_budget_unfinishes_long_samples_and_zeroes_their_score():
    rows = [{"id": "a", "kind": "phys", "k": 0, "finished": True, "score": 1.0, "len": 5000},
            {"id": "a", "kind": "phys", "k": 1, "finished": True, "score": 0.5, "len": 3000},
            {"id": "a", "kind": "phys", "k": 2, "finished": False, "score": 0.0, "len": 16384}]
    at = read_at(rows, 4096)
    assert [r["finished"] for r in at] == [False, True, False]
    assert [r["score"] for r in at] == [0.0, 0.5, 0.0]
    assert [r["len"] for r in at] == [4096, 3000, 4096]
    assert rates(arrays(group_by_problem(at), group_by_problem(at)))["base"]["finished"] == pytest.approx(1 / 3)
