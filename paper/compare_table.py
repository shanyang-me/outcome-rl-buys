"""Compare a LaTeX table body written by a paper script with the paper's CSV of the same table, cell by cell.

Header rows (first cell empty or 'recipe') are dropped on both sides; LaTeX markup is stripped before comparing.
usage: python paper/compare_table.py rows.tex table.csv"""
import csv
import re
import sys


def clean(s):
    s = re.sub(r"\\(textbf|emph|textit|mathrm|text|bm)\{([^{}]*)\}", r"\2", s)
    s = s.replace("$", "").replace("\\,", "").replace("{,}", ",").replace("\\to", "->").replace("\\geq", ">=").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?", "", s).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip().replace("−", "-")


def body(rows):
    rows = [r for r in rows if r and any(c for c in r)]
    while rows and rows[0][0] in ("", "recipe", "model"):  # leading header rows; continuation rows later keep an empty first cell
        rows = rows[1:]
    return [r for r in rows if r[0] not in ("recipe", "model") and any(re.search(r"\d", c) for c in r[1:])]


tex = [ln for ln in open(sys.argv[1]).read().replace("\\\\", "\n").splitlines() if "&" in ln]
got = body([[clean(c) for c in ln.split("&")] for ln in tex])
want = body([[clean(c) for c in row] for row in csv.reader(ln for ln in open(sys.argv[2]) if not ln.startswith("#"))])
bad = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w]
print(f"{len(got)} tex rows, {len(want)} csv rows, {len(bad)} rows differ")
for b in bad[:8]:
    print("  row", b[0], "\n    tex:", b[1], "\n    csv:", b[2])
sys.exit(1 if bad or len(got) != len(want) else 0)
