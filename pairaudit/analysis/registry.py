"""The pair registry: which RL model is paired with which base, at what training cap.

A config is JSON with ``bases`` and ``pairs`` (``recipes`` is accepted as a synonym).
Everything else is optional and has a data-derived default:

* ``budgets``: ``fixed`` (budgets every pair is read at), ``ladder`` (budgets a
  generation is re-read at), ``labels`` (budget -> display label). Missing budgets are
  taken from the row store.
* ``protocol``: ``kinds`` (benchmark names in display order; the first is the primary
  one the cap study selects on), ``kind_labels``, ``binding`` (budgets where the budget
  binds, for the dominance statistic; default: every fixed budget but the largest).
* ``views``: named ordered lists of row runs; a missing view means every run in the manifest.
* ``groups``: named lists of pair groups; ``registered`` marks the pre-registered set.

The shipped ``configs/recipes.json`` is the paper's registry; ``configs/pairs.template.json``
is the minimal shape for your own pair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recipes.json"


@dataclass(frozen=True)
class Recipe:
    name: str
    model: str
    hf_id: str
    base: str
    base_label: str
    base_hf_id: str
    train_cap: int
    cap_label: str
    group: str
    authors_T: tuple[float, float] | None
    notes: str = ""


@dataclass(frozen=True)
class Registry:
    recipes: dict[str, Recipe]
    bases: dict[str, dict]
    budgets: dict
    views: dict[str, list[str]]
    protocol: dict
    groups: dict

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> Registry:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> Registry:
        pairs = d.get("pairs", d.get("recipes"))
        if not pairs or "bases" not in d:
            raise ValueError("config needs 'bases' and 'pairs'")
        recipes = {}
        for name, r in pairs.items():
            b = d["bases"][r["base"]]
            recipes[name] = Recipe(
                name=name, model=r["model"], hf_id=r.get("hf_id", r["model"]), base=r["base"], base_label=b.get("label", r["base"]),
                base_hf_id=b.get("hf_id", r["base"]), train_cap=int(r["train_cap"]), cap_label=r.get("cap_label", f"{int(r['train_cap']) // 1024}k"),
                group=r.get("group", "default"), authors_T=tuple(r["authors_T"]) if r.get("authors_T") else None, notes=r.get("notes", ""),
            )
        clean = lambda m: {k: v for k, v in (m or {}).items() if not k.startswith("_")}
        budgets = {"fixed": [], "ladder": [], "labels": {}, **clean(d.get("budgets"))}
        return cls(recipes, d["bases"], budgets, clean(d.get("views")), clean(d.get("protocol")), clean(d.get("groups")))

    # --- selection -----------------------------------------------------------------
    def select(self, groups: list[str] | None = None) -> list[Recipe]:
        """Pairs in registry order, optionally restricted to some groups (a named group list expands)."""
        if groups is None:
            return list(self.recipes.values())
        wanted = set()
        for g in groups:
            wanted.update(self.groups[g] if isinstance(self.groups.get(g), list) else [g])
        return [r for r in self.recipes.values() if r.group in wanted]

    @property
    def registered(self) -> list[Recipe]:
        """The pre-registered pairs if the config marks them, else every pair."""
        return self.select(["registered"]) if "registered" in self.groups else self.select()

    def by_model(self, model: str) -> Recipe:
        for r in self.recipes.values():
            if r.model == model:
                return r
        raise KeyError(model)

    def families(self, recipes: list[Recipe] | None = None) -> dict[str, list[Recipe]]:
        """Pairs grouped by base, in registry order."""
        fam: dict[str, list[Recipe]] = {}
        for r in recipes if recipes is not None else self.recipes.values():
            fam.setdefault(r.base, []).append(r)
        return fam

    # --- budgets and benchmarks --------------------------------------------------------
    def label(self, cap: int) -> str:
        return self.budgets["labels"].get(str(cap), f"{cap // 1024}k" if cap % 1024 == 0 else f"{cap}")

    @property
    def ladder(self) -> list[int]:
        return list(self.budgets["ladder"])

    @property
    def fixed_budgets(self) -> list[int]:
        return list(self.budgets["fixed"])

    @property
    def binding(self) -> list[int]:
        """Budgets where the budget binds (below the training caps); the dominance statistic pools these."""
        return list(self.protocol.get("binding") or self.fixed_budgets[:-1])

    @property
    def kinds(self) -> tuple[str, ...] | None:
        """Benchmarks in display order, or None to take them from the rows."""
        k = self.protocol.get("kinds")
        return tuple(k) if k else None

    def kind_label(self, kind: str) -> str:
        return self.protocol.get("kind_labels", {}).get(kind, kind)

    def view(self, name: str) -> list[str] | None:
        """The runs behind a view, or None (every run) when the config does not define it."""
        return self.views.get(name)
