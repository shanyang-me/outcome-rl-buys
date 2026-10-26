"""The matcher wrapper and the judge protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pairaudit.scoring.answer_equiv as frozen
from pairaudit.scoring.boxed import extract_boxed
from pairaudit.scoring.judge_protocol import RULES, decide, parse_verdict, prompt_for, split
from pairaudit.scoring.matcher import compute_score, score, score_details

FROZEN_SHA = "200d5aa0141268dce366cda8c38c5ec2e4640738d7bafbd2fb50a07a04830ae5"


def gt(*golds, mode="numeric", **extra):
    return json.dumps({"equiv_mode": mode, "golds": list(golds), **extra})


def test_matcher_file_is_the_campaign_pin():
    # every number in the audit was scored by this exact file; a change here moves every cell at once
    assert hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() == FROZEN_SHA


def test_boxed_extraction_skips_a_truncated_final_box():
    assert extract_boxed(r"so \boxed{3.0} and then \boxed{\frac{1}{2}} and \boxed{cut off") == ["3.0", r"\frac{1}{2}"]
    assert extract_boxed("no boxes") == []


def test_matcher_credits_fraction_of_golds_and_is_verl_compatible():
    text = r"first \boxed{2.5 m} then \boxed{10}"
    assert score(text, gt("2.5", "10 s", "7")) == 2 / 3
    assert compute_score("any", text, gt("2.5", "10 s", "7")) == 2 / 3
    assert score("nothing boxed", gt("1")) == 0.0
    assert score_details(text, "not json")["invalid_ground_truth"] is True


def test_unfinished_text_scores_zero_only_by_the_callers_rule():
    # the matcher itself scores what is boxed; the finished rule lives in the sampler
    assert score(r"\boxed{4}", gt("4")) == 1.0


def test_protocol_decides_bare_numbers_mechanically_and_defers_symbolic():
    assert decide("6V", ["6"]) is True
    assert decide("0.025 m", ["-0.025"]) is False
    assert decide("anything", []) is False
    assert decide(r"\frac{mv^2}{2}", [r"\frac{1}{2} m v^2"]) is None


def test_split_buckets_and_prompt_carries_rules_and_parsed_forms():
    s = split([{"key": "a", "gold": "6V", "preds": ["6"]}, {"key": "b", "gold": "x^2", "preds": ["x*x"]}, {"key": "c", "gold": "1", "preds": []}])
    assert s.decided == {"a": True, "c": False}
    assert set(s.deferred) == {"b"} and s.no_answer == 1
    assert s.mechanical_share == 2 / 3
    p = prompt_for("6 V", ["6", "7"])
    assert RULES in p and "gold parses as: magnitude 6, unit V" in p


def test_verdict_parsing_handles_thinking_and_garbage():
    assert parse_verdict("<think>hmm</think>\nYES") == "YES"
    assert parse_verdict("No, the sign differs.") == "NO"
    assert parse_verdict("<think>ran out") == "UNPARSED"
