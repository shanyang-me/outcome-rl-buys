"""Command line: ``pairaudit <command>``.

``score`` and ``analyze`` are the tool path for your own pair: outputs -> rows -> every table.
The other analysis commands expose one table each and run on CPU; ``generate``, ``reach``
and ``regrade`` need a GPU and vLLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pairaudit.analysis.registry import DEFAULT_CONFIG, Registry
from pairaudit.analysis.rows import RowStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "data" / "rows"


def _reg(a) -> Registry:
    return Registry.load(a.config)


def _store(a, view: str) -> RowStore:
    """The rows behind a named view; a config without views reads every run in the manifest."""
    return RowStore(a.rows, _reg(a).view(view))


def _kinds(a):
    return tuple(a.kinds.split(",")) if getattr(a, "kinds", None) else None


def _dump(obj, path: str | None) -> None:
    text = json.dumps(obj, indent=1)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        print(f"wrote {path}", file=sys.stderr)
    else:
        print(text)


def cmd_summary(a):
    """Cell summaries for one model at one budget."""
    from pairaudit.analysis.rows import summarize

    store = _store(a, a.view)
    for kind, d in summarize(store.rows(a.model, a.cap)).items():
        print(f"{a.model} cap{a.cap} {kind:5s} mean {d['mean']:.3f} | finished {d['finished']:.2f} | score|finished {d['score_given_finished']:.3f} | any-solve {d['any_solve']:.3f} | tokens {d['mean_len']:.0f} | problems {d['n_items']}")


def cmd_budget(a):
    """At-cap and beyond-cap budgets and the two-part split per recipe (the cap study's cell table)."""
    from pairaudit.analysis.budget import cap_split
    from pairaudit.analysis.report import cap_split_markdown

    reg = _reg(a)
    out = cap_split(_store(a, "cap"), reg, reg.select(a.groups), a.kind)
    print(cap_split_markdown(out))
    print("pooled:", json.dumps(out["pooled"]))
    if a.out:
        _dump(out, a.out)


def cmd_ledger(a):
    """The ledger: pooled shares per budget with crossed-bootstrap intervals, and the TikZ figure."""
    from pairaudit.analysis.ledger import ledger
    from pairaudit.analysis.report import ledger_markdown, ledger_tikz

    reg = _reg(a)
    led = ledger(_store(a, "fixed"), _store(a, "cap"), reg, kinds=_kinds(a), n_boot=a.boot, with_ci=not a.no_ci)
    print(ledger_markdown(led, reg))
    if a.out:
        _dump(led, a.out)
    if a.tex:
        Path(a.tex).write_text(ledger_tikz(led, reg))
        print(f"wrote {a.tex}", file=sys.stderr)


def cmd_cells(a):
    """Fixed-budget cells with problem-clustered intervals (default: the registered recipes)."""
    from pairaudit.analysis.cells import fixed_budget_cells
    from pairaudit.analysis.report import fixed_cells_markdown

    reg = _reg(a)
    out = fixed_budget_cells(_store(a, "fixed"), reg, reg.select(a.groups) if a.groups else reg.registered, n_boot=a.boot)
    print(fixed_cells_markdown(out))
    if a.out:
        _dump(out, a.out)


def cmd_captable(a):
    """Every cell behind the cap rows, with paired bootstrap intervals; LaTeX on request."""
    from pairaudit.analysis.cells import cap_cell_table
    from pairaudit.analysis.report import cap_cells_latex

    reg = _reg(a)
    out = cap_cell_table(_store(a, "cap"), reg, reg.select(a.groups), n_boot=a.boot)
    for name, r in out.items():
        for s in r["cells"]:
            print(f"{name:14s} {s['kind']:5s} {s['label']:>8s}  F {s['F'][0]:+.3f}  P {s['P'][0]:+.3f} [{s['P'][1]:+.3f}, {s['P'][2]:+.3f}]  reach {s['reach'][0]:+.3f}  d {s['d'][0]:+.3f}")
    if a.out:
        _dump(out, a.out)
    if a.tex:
        Path(a.tex).write_text(cap_cells_latex(out, reg))
        print(f"wrote {a.tex}", file=sys.stderr)


def cmd_stats(a):
    """Recipe-level inference: dominance where the budget binds, sign test, matched change, length association."""
    from pairaudit.analysis.stats import recipe_stats

    reg = _reg(a)
    flags = json.loads(Path(a.flags).read_text()) if a.flags else None
    out = recipe_stats(_store(a, "fixed"), reg, reg.select(a.groups) if a.groups else reg.registered, flags, n_boot=a.boot)
    for k, v in out.items():
        if k not in ("dominance", "matched"):
            print(k, json.dumps(v))
    for n, d in out["dominance"].items():
        m = out["matched"][n]
        print(f"  {n:14s} dominance {d['point']:.3f} {d['ci_problems']}  changing {d['changing']}  matched {m['mean']}")
    if a.out:
        _dump(out, a.out)


def cmd_temperature(a):
    """A pair read at T=0.6 and at its authors' sampling, side by side."""
    from pairaudit.analysis.temperature import temperature_cells

    reg = _reg(a)
    rec = reg.recipes[a.recipe]
    out = temperature_cells(_store(a, "cap"), _store(a, "authors_T"), rec, n_boot=a.boot, reg=reg)
    for cap, row in out.items():
        print(f"--- {reg.label(cap)}")
        for lab, p in row.items():
            print(f"  {lab:8s} fin {p['fin_b']:.3f}->{p['fin_r']:.3f} acc|fin {p['acc_b']:.3f}->{p['acc_r']:.3f} | F {p['F'][0]:+.3f} P {p['P'][0]:+.3f} reach {p['reach'][0]:+.3f} d {p['d'][0]:+.3f} [{p['d'][1]:+.3f}, {p['d'][2]:+.3f}]")
    if a.out:
        _dump({str(k): v for k, v in out.items()}, a.out)


def cmd_score(a):
    """Score stored outputs into rows for one model and register the run (CPU)."""
    from pairaudit.scoring.rescore import rescore_file, write_run

    reg = _reg(a)
    rows = rescore_file(Path(a.texts), Path(a.items), a.scorer)
    gen_cap = a.cap or max(r["len"] for r in rows)
    path = write_run(Path(a.rows), a.run, a.model, rows, gen_cap, a.ladder and [int(c) for c in a.ladder.split(",")] or reg.ladder)
    fin = sum(r["finished"] for r in rows) / len(rows)
    print(f"wrote {path}: {len(rows)} rows, {len({r['id'] for r in rows})} problems, finished {fin:.2f}, cap {gen_cap}", file=sys.stderr)


def cmd_analyze(a):
    """Everything at once: cap split, ledger, per-cell intervals, cap table and recipe statistics, as JSON plus one Markdown report."""
    from pairaudit.analysis.budget import cap_split
    from pairaudit.analysis.cells import cap_cell_table, fixed_budget_cells
    from pairaudit.analysis.ledger import ledger
    from pairaudit.analysis.report import cap_split_markdown, fixed_cells_markdown, ledger_markdown
    from pairaudit.analysis.stats import recipe_stats

    reg = _reg(a)
    fixed, cap = _store(a, "fixed"), _store(a, "cap")
    recs = reg.select(a.groups) if a.groups else reg.select()
    registered = reg.select(a.groups) if a.groups else reg.registered
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = [f"# pairaudit report\n\nconfig `{a.config}`, rows `{a.rows}`, {len(recs)} pair(s)\n"]

    split = cap_split(cap, reg, recs)
    _dump(split, out / "cap_split.json")
    report += ["## Where each pair's cap falls, and the two-part split there\n", cap_split_markdown(split), ""]

    led = ledger(fixed, cap, reg, kinds=_kinds(a), n_boot=a.boot, fixed_groups=None if not a.groups else a.groups, cap_groups=a.groups, with_ci=not a.no_ci)
    _dump(led, out / "ledger.json")
    report += ["## Ledger: pooled shares of finishing, reliability and reach\n", ledger_markdown(led, reg), ""]

    cells = fixed_budget_cells(fixed, reg, registered, kinds=_kinds(a), n_boot=a.cell_boot)
    _dump(cells, out / "cells.json")
    report += ["## Fixed-budget cells with problem-clustered intervals\n", fixed_cells_markdown(cells), ""]

    table = cap_cell_table(cap, reg, recs, kinds=_kinds(a), n_boot=a.cell_boot)
    _dump(table, out / "cap_cells.json")

    stats = recipe_stats(fixed, reg, registered, cells, kinds=_kinds(a), n_boot=a.boot)
    _dump(stats, out / "stats.json")
    report += ["## Recipe-level statistics\n", "```json", json.dumps({k: v for k, v in stats.items() if k not in ("dominance", "matched")}, indent=1), "```", ""]

    (out / "report.md").write_text("\n".join(report))
    print(f"wrote {out / 'report.md'}", file=sys.stderr)


def cmd_generate(a):
    """Generate n samples per problem for each model (GPU)."""
    from pairaudit.generate.items import load_items, load_spec
    from pairaudit.generate.sample import run_queue

    items = load_items(a.items)
    if a.smoke:
        items = items[:4] + items[-4:]
    spec = load_spec(a.spec, (a.temperature, a.top_p, a.cap)) if a.spec else None
    models = [m for m in a.models.split(",") if m]
    kw = {} if spec else {"temperature": a.temperature, "top_p": a.top_p}
    run_queue(models, items, Path(a.out), spec, default_cap=1024 if a.smoke else a.cap, n=2 if a.smoke else a.n, seed=a.seed, tp=a.tp,
              cache_dirs=a.cache.split(",") if a.cache else None, download_dir=a.download_dir, **kw)


def cmd_reach(a):
    """Extra samples on each model's never-solved problems (GPU); targets: {model: {ids, cap}}."""
    from pairaudit.generate.items import load_items
    from pairaudit.generate.sample import run_queue

    targets = json.loads(Path(a.targets).read_text())
    run_queue([m for m in a.models.split(",") if m], load_items(a.items), Path(a.out), None, default_cap=0, n=a.n, seed=a.seed,
              targets=targets, k0=a.k0, tp=a.tp, cache_dirs=a.cache.split(",") if a.cache else None, download_dir=a.download_dir,
              temperature=a.temperature, top_p=a.top_p)


def cmd_regrade(a):
    """Judge regrade of stored texts (GPU for the deferred remainder)."""
    from pairaudit.generate.regrade import regrade

    regrade(Path(a.out), [Path(d) for d in a.src], Path(a.items), a.judge)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pairaudit", description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="pair registry JSON (default: the paper's configs/recipes.json; see configs/pairs.template.json)")
    p.add_argument("--rows", default=str(DEFAULT_ROWS), help="row data root with MANIFEST.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help=cmd_summary.__doc__)
    s.add_argument("model"), s.add_argument("cap", type=int), s.add_argument("--view", default="cap"), s.set_defaults(f=cmd_summary)
    s = sub.add_parser("budget", help=cmd_budget.__doc__)
    s.add_argument("--groups", nargs="*"), s.add_argument("--kind", help="benchmark the budgets are selected on (default: primary)"), s.add_argument("--out"), s.set_defaults(f=cmd_budget)
    s = sub.add_parser("ledger", help=cmd_ledger.__doc__)
    s.add_argument("--kinds", help="comma-separated benchmarks (default: the config's, else the rows')"), s.add_argument("--boot", type=int, default=2000), s.add_argument("--no-ci", action="store_true")
    s.add_argument("--out"), s.add_argument("--tex"), s.set_defaults(f=cmd_ledger)
    s = sub.add_parser("cells", help=cmd_cells.__doc__)
    s.add_argument("--groups", nargs="*"), s.add_argument("--boot", type=int, default=20000), s.add_argument("--out"), s.set_defaults(f=cmd_cells)
    s = sub.add_parser("captable", help=cmd_captable.__doc__)
    s.add_argument("--groups", nargs="*"), s.add_argument("--boot", type=int, default=20000), s.add_argument("--out"), s.add_argument("--tex"), s.set_defaults(f=cmd_captable)
    s = sub.add_parser("stats", help=cmd_stats.__doc__)
    s.add_argument("--groups", nargs="*"), s.add_argument("--flags", help="cells JSON from `pairaudit cells --out`")
    s.add_argument("--boot", type=int, default=4000), s.add_argument("--out"), s.set_defaults(f=cmd_stats)
    s = sub.add_parser("temperature", help=cmd_temperature.__doc__)
    s.add_argument("recipe"), s.add_argument("--boot", type=int, default=20000), s.add_argument("--out"), s.set_defaults(f=cmd_temperature)

    s = sub.add_parser("score", help=cmd_score.__doc__)
    s.add_argument("--texts", required=True, help="JSON lines {id, k, text, finish_reason|finished, len[, kind]}")
    s.add_argument("--items", required=True, help="JSON lines {id, gt[, kind]}: the golds the scorer sees")
    s.add_argument("--model", required=True, help="model name as in the config (row file prefix)"), s.add_argument("--run", required=True, help="run directory name under --rows")
    s.add_argument("--scorer", help="module:function taking (text, gt) -> float; default the frozen matcher")
    s.add_argument("--cap", type=int, help="generation cap in tokens (default: the longest output)"), s.add_argument("--ladder", help="comma-separated tighter budgets to expose (default: the config's)")
    s.set_defaults(f=cmd_score)
    s = sub.add_parser("analyze", help=cmd_analyze.__doc__)
    s.add_argument("--out", default="out"), s.add_argument("--groups", nargs="*"), s.add_argument("--kinds")
    s.add_argument("--boot", type=int, default=2000, help="crossed-bootstrap replicates for the ledger and statistics")
    s.add_argument("--cell-boot", type=int, default=20000, help="problem-bootstrap replicates per cell"), s.add_argument("--no-ci", action="store_true")
    s.set_defaults(f=cmd_analyze)

    g = sub.add_parser("generate", help=cmd_generate.__doc__)
    g.add_argument("--models", required=True, help="comma-separated Hugging Face ids, in queue order")
    g.add_argument("--items", required=True), g.add_argument("--out", required=True)
    g.add_argument("--spec", help="per-model [temperature, top_p, cap] JSON (file or string)")
    g.add_argument("--cap", type=int, default=16384), g.add_argument("--n", type=int, default=8)
    g.add_argument("--temperature", type=float, default=0.6), g.add_argument("--top-p", type=float, default=0.95), g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tp", type=int, default=1), g.add_argument("--cache", help="comma-separated HF caches to look in first"), g.add_argument("--download-dir", default="/root/hf")
    g.add_argument("--smoke", action="store_true", help="8 problems, 2 samples, 1k cap: exercises every code path"), g.set_defaults(f=cmd_generate)
    r = sub.add_parser("reach", help=cmd_reach.__doc__)
    r.add_argument("--models", required=True), r.add_argument("--items", required=True), r.add_argument("--out", required=True), r.add_argument("--targets", required=True)
    r.add_argument("--n", type=int, default=24), r.add_argument("--k0", type=int, default=8), r.add_argument("--seed", type=int, default=2)
    r.add_argument("--temperature", type=float, default=0.6), r.add_argument("--top-p", type=float, default=0.95), r.add_argument("--tp", type=int, default=1)
    r.add_argument("--cache"), r.add_argument("--download-dir", default="/root/hf"), r.set_defaults(f=cmd_reach)
    j = sub.add_parser("regrade", help=cmd_regrade.__doc__)
    j.add_argument("--out", required=True), j.add_argument("--src", nargs="+", required=True, help="run dirs holding texts/*.jsonl.gz"), j.add_argument("--items", required=True)
    j.add_argument("--judge", default="Qwen/Qwen3-32B"), j.set_defaults(f=cmd_regrade)

    a = p.parse_args(argv)
    a.f(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
