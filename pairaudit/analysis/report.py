"""Human-readable outputs: Markdown tables for the terminal and the README, LaTeX tables and the TikZ ledger for a paper."""

from __future__ import annotations

from pairaudit.analysis.registry import Registry


def pct(x: float) -> str:
    return "<1%" if 0 < x < 0.01 else f"{round(100 * x)}%"


def tex_pct(x: float) -> str:
    return pct(x).replace("%", "\\%")


def _z(x: float, p: int = 3) -> str:
    return f"{x:+.{p}f}" if abs(x) >= 0.5 * 10**-p else "0." + "0" * p


def ledger_markdown(led: dict, reg: Registry) -> str:
    """One row per budget: size shares of finishing, reliability and reach, with the cap rows' intervals."""
    L = ["| budget | recipes | cells | finishing | reliability | reach (net of mirror) | robust reach |", "|---|---|---|---|---|---|---|"]

    def row(name, e):
        p = e["pooled"]
        ci = e.get("ci")
        f = lambda j, v: pct(v) + (f" [{round(100 * ci['size'][j][0])}, {round(100 * ci['size'][j][1])}]" if ci else "")
        return f"| {name} | {p['n']} | {p['cells']} | {f(0, p['F'])} | {f(1, p['P'])} | {f(2, p['reach'])} | {pct(p['reach_robust'])} |"

    for cap, e in led["budget"].items():
        L.append(row(reg.label(int(cap)), e))
    for which, name in (("at", "at the training cap"), ("beyond", "one budget beyond")):
        if which in led["cap"]:
            L.append(row(name, led["cap"][which]))
    return "\n".join(L)


def cap_split_markdown(cs: dict) -> str:
    L = ["| recipe | base | cap | at: budget | F | P | change | finished base→RL | beyond: budget | F | P | change |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in cs["recipes"].items():
        a = r["at"]
        b = r.get("beyond")
        L.append(f"| {name} | {r['base']} | {r['cap_label']} | {r['at_label']} | {_z(a['F'])} | {_z(a['P'])} | {_z(a['d'])} | {a['fb']:.2f}→{a['fr']:.2f} | "
                 + (f"{r['beyond_label']} | {_z(b['F'])} | {_z(b['P'])} | {_z(b['d'])} |" if b else "not read | | | |"))
    return "\n".join(L)


def fixed_cells_markdown(cells: dict) -> str:
    L = ["| recipe | set | budget | capped score base→RL (Δ [95% CI]) | finished (Δ) | score given finished (Δ) | matched Δ (n) | finishing share | tokens |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cells.values():
        f = lambda q, p=3: f"{c[q]['base']:.{p}f}→{c[q]['rl']:.{p}f} ({_z(c[q]['delta'], p)} [{_z(c[q]['ci'][0], p)}, {_z(c[q]['ci'][1], p)}])"
        m = c["matched"]
        matched = f"{_z(m['delta'])} ({m['n']})" if "delta" in m else f"— ({m['n']})"
        share = pct(c["finishing_share"]) if c["finishing_share"] is not None and (c["capped_gain"] or c["capped_loss"]) else "—"
        L.append(f"| {c['recipe']} | {c['set']} | {c['cap'] // 1024}k | {f('mean')} | {f('finished', 2)} | {f('sgf')} | {matched} | {share} | {c['len_ratio']:.2f}× |")
    return "\n".join(L)


def cap_cells_latex(table: dict, reg: Registry) -> str:
    """The appendix table of every cap cell with intervals."""
    num = lambda x, p=3: ("$-$" if x < -0.5 * 10**-p else "") + f"{abs(x):.{p}f}" if abs(x) >= 0.5 * 10**-p else "0"
    ci = lambda v: f"{num(v[0])} [{num(v[1])}, {num(v[2])}]"
    sets = {k: reg.kind_label(k) for k in {s["kind"] for r in table.values() for s in r["cells"]}}
    rows = []
    for name, r in table.items():
        first = True
        for s in r["cells"]:
            rows.append(f"{name if first else ''} & {sets[s['kind']]}, {s['label'].replace(' (below cap)', '$^{b}$')} & ${s['fin_b']:.2f}\\to{s['fin_r']:.2f}$ & ${s['acc_b']:.2f}\\to{s['acc_r']:.2f}$ & {ci(s['F'])} & {ci(s['P'])} & {ci(s['reach'])} & {ci(s['d'])} \\\\")
            first = False
        rows.append("\\addlinespace[1pt]")
    head = ["\\begin{tabular}{llcccccc}", "\\toprule",
            "recipe & set, budget & fin@1$(B)$ & pass@1$(B\\,|\\,\\text{fin})$ & finishing $F$ & reliability $P$ & reach & change $\\Delta$ \\\\",
            " & & base$\\to$RL & base$\\to$RL & & & net of mirror & \\\\", "\\midrule"]
    return "\n".join(head + rows + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def ledger_tikz(led: dict, reg: Registry, width_cm: float = 5.0, half: float = 0.21) -> str:
    """The ledger as one TikZ picture: one bar per budget, segments finishing / reliability / reach, shares of the parts' total."""
    rows = [(reg.label(int(cap)), e["pooled"]) for cap, e in led["budget"].items()]
    rows += [(n, led["cap"][w]["pooled"]) for w, n in (("at", "at the training cap"), ("beyond", "beyond the cap")) if w in led["cap"]]
    L = [f"\\begin{{tikzpicture}}[x={width_cm:.2f}cm,y=1cm]"]
    y = 0.7 * (len(rows) - 1)
    for name, p in rows:
        F, P, Rr = p["F"], p["P"], p["reach"]
        L.append(f"\\fill[finishink] (0,{y - half:.2f}) rectangle ({F:.4f},{y + half:.2f});")
        L.append(f"\\fill[reliaink] ({F:.4f},{y - half:.2f}) rectangle ({F + P:.4f},{y + half:.2f});")
        if Rr > 0.003:
            L.append(f"\\fill[pattern=north east lines,pattern color=reachink] ({F + P:.4f},{y - half:.2f}) rectangle (1,{y + half:.2f});")
        if p.get("reach_robust", 0) > 0.002:
            L.append(f"\\fill[reachink] ({1 - p['reach_robust']:.4f},{y - half:.2f}) rectangle (1,{y + half:.2f});")
        L.append(f"\\node[anchor=east,font=\\footnotesize,text=white,inner sep=1pt] at ({F - 0.012:.4f},{y:.2f}) {{{tex_pct(F)}}};")
        L.append(f"\\node[font=\\footnotesize,inner sep=1pt] at ({F + P / 2:.4f},{y:.2f}) {{{tex_pct(P)}}};")
        L.append(f"\\node[anchor=west,font=\\footnotesize,inner sep=1pt] at (1.015,{y:.2f}) {{reach {('$0$' if Rr <= 0 else tex_pct(Rr))}}};")
        L.append(f"\\node[anchor=east,font=\\footnotesize,inner sep=1pt] at (-0.02,{y:.2f}) {{{name}}};")
        y -= 0.7
    L += [f"\\draw[black!60,thin] (0,{y + 0.2:.2f}) -- (0,{0.7 * (len(rows) - 1) + 0.3:.2f});", f"\\draw[black!60,thin] (0,{y + 0.2:.2f}) -- (1,{y + 0.2:.2f});",
          f"\\foreach \\x/\\l in {{0/0,0.5/half,1/all}} \\node[anchor=north,font=\\footnotesize] at (\\x,{y + 0.18:.2f}) {{\\l}};",
          "\\end{tikzpicture}"]
    return "\n".join(L) + "\n"
