"""Authors'-temperature check: the same pair read at the audit's T=0.6 and at the sampling its authors evaluate with."""

from __future__ import annotations

from pairaudit.analysis.cells import cap_cell_stats
from pairaudit.analysis.decomp import arrays
from pairaudit.analysis.registry import Recipe
from pairaudit.analysis.rows import RowStore, resolve_kinds


def temperature_cells(store_06: RowStore, store_authors: RowStore, rec: Recipe, kind: str | None = None, n_boot: int = 20000, reg=None) -> dict:
    """Per shared budget: the three-part split with intervals under both reads (``kind``: default the primary benchmark)."""
    kind = kind or (resolve_kinds(store_06, reg, [rec])[0] if reg else store_06.kinds_for(rec.base)[0])
    caps = sorted(set(store_06.caps_for(rec.model)) & set(store_authors.caps_for(rec.model)))
    out = {}
    for cap in caps:
        row = {}
        for label, store in (("T0.6", store_06), ("authors", store_authors)):
            if not (store.has(rec.base, cap) and store.has(rec.model, cap)):
                continue
            c = arrays(store.load(rec.base, cap), store.load(rec.model, cap), kind)
            if c is not None:
                row[label] = cap_cell_stats(c, n_boot)
        if len(row) == 2:
            out[cap] = row
    return out
