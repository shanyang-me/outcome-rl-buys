"""Compare two result JSON files number by number. usage: python paper/compare_json.py got.json want.json [tol]"""
import json
import sys

got, want = (json.load(open(p)) for p in sys.argv[1:3])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
bad, n = [], 0


def walk(x, y, p):
    global n
    if isinstance(y, dict):
        for k, v in y.items():
            walk(x.get(k) if isinstance(x, dict) else None, v, f"{p}/{k}")
    elif isinstance(y, list):
        for i, v in enumerate(y):
            walk(x[i] if isinstance(x, list) and i < len(x) else None, v, f"{p}[{i}]")
    elif isinstance(y, (int, float)) and not isinstance(y, bool):
        n += 1
        if not isinstance(x, (int, float)) or abs(x - y) > tol:
            bad.append((p, x, y))


walk(got, want, "")
print(f"{n} numbers compared, {len(bad)} differ (tol {tol})")
for b in bad[:12]:
    print("  ", *b)
sys.exit(1 if bad else 0)
