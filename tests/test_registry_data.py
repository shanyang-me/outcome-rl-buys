"""The registry and the shipped rows agree: every recipe can be read where the analyses need it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pairaudit.analysis.budget import cap_budgets
from pairaudit.analysis.registry import Registry
from pairaudit.analysis.rows import RowStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reg():
    return Registry.load()


def test_every_registered_recipe_has_fixed_budget_rows_for_both_models(reg):
    store = RowStore(ROOT / "data/rows", reg.views["fixed"])
    for rec in reg.select(["registered"]):
        for cap in reg.fixed_budgets:
            assert store.has(rec.model, cap), (rec.name, cap)
            assert store.has(rec.base, cap), (rec.name, rec.base, cap)


def test_every_recipe_reaches_its_cap_and_one_beyond_in_the_cap_view(reg):
    store = RowStore(ROOT / "data/rows", reg.views["cap"])
    for rec in reg.select():
        at, beyond, below_only = cap_budgets(store, rec)
        assert at is not None and at >= rec.train_cap and not below_only, (rec.name, at)
        assert beyond is not None and beyond > at, (rec.name, beyond)


def test_train_caps_sit_on_or_under_the_ladder_budget_used(reg):
    for rec in reg.select():
        assert rec.train_cap in reg.ladder, (rec.name, rec.train_cap)


def test_groups_partition_the_recipes(reg):
    groups = [r.group for r in reg.select()]
    assert set(groups) == {"original", "added", "amend1", "extra"}
    assert len(reg.select(["registered"])) == 18 and len(reg.select()) == 20


def test_manifest_files_exist_and_nested_runs_hold_one_file_per_model(reg):
    root = ROOT / "data/rows"
    man = json.loads((root / "MANIFEST.json").read_text())
    for run, v in man.items():
        for model, spec in v["models"].items():
            caps = [spec["generation_cap"]] if v["nested"] else spec["caps"]
            for cap in caps:
                assert (root / run / f"{model}_cap{cap}.jsonl.gz").exists(), (run, model, cap)
