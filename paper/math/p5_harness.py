"""Prediction P5 of the competition-math read (PREREG.md): our read of each RL model's AIME 2025 pass@1 at its recipe's cap is within
0.08 of the number its authors report, where they report one at a comparable budget and temperature (authors_aime25.json, field
'comparable'). Also printed for every model with any reported number, and at the largest budget we read, for context; the bases
likewise. pass@1 = mean over the 8 samples of (finished and exactly right); judged scores where JUDGED=dir holds the model's file.
usage: [JUDGED=dir] python p5_harness.py rows_dir [out.json]"""
import json, glob, os, sys
H = os.path.dirname(os.path.abspath(__file__)); ROWS = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
A = json.load(open(os.path.join(H, "authors_aime25.json")))
src = open(os.path.join(H, "..", "analysis", "cap_split.py")).read(); i = src.index("PAIRS = {"); d = 0
for j in range(i, len(src)):
    d += (src[j] == "{") - (src[j] == "}")
    if d == 0 and src[j] == "}": break
exec(src[i:j + 1])
cap_of = {}  # model -> the recipe's cap (an RL model) or the largest cap among its recipes (a base)
for name, (rl, base, cap, tag, lab) in PAIRS.items():
    rn, bn = rl.split("/")[-1], base.split("/")[-1]; cap_of[rn] = cap; cap_of[bn] = max(cap_of.get(bn, 0), cap)
def caps_for(m): return sorted({int(os.path.basename(f).split("_cap")[1].split(".")[0]) for f in glob.glob(f"{ROWS}/{m}_cap*.jsonl")})
def pass1(m, cap):
    jm = None; J = os.environ.get("JUDGED")
    if J and os.path.exists(f"{J}/{m}.jsonl"):
        jm = {(r["id"], r["k"]): r["score_j"] for r in (json.loads(l) for l in open(f"{J}/{m}.jsonl") if l.strip())}
    n = ok = fin = 0
    for l in open(f"{ROWS}/{m}_cap{cap}.jsonl"):
        if not l.strip(): continue
        r = json.loads(l)
        if not r["id"].startswith("aime_2025|"): continue
        s = r["score"] if r["finished"] else 0.0
        if jm is not None and r["finished"]: s = jm.get((r["id"], r["k"]), s)
        n += 1; fin += bool(r["finished"]); ok += s >= 1 - 1e-9
    return (ok / n, fin / n, n // 8, jm is not None) if n else None
res = {"rl": {}, "base": {}}; scored = []
for side in ("rl", "base"):
    for m, a in A[side].items():
        caps = caps_for(m)
        if not caps: print(f"{m:40s} rows missing"); continue
        at = min((c for c in caps if c >= cap_of.get(m, 0)), default=max(caps)); top = max(caps)
        p_at, p_top = pass1(m, at), pass1(m, top)
        if p_at is None: print(f"{m:40s} no AIME 2025 rows"); continue
        row = {"cap": at, "pass1_at_cap": p_at[0], "fin_at_cap": p_at[1], "largest_cap": top, "pass1_at_largest": p_top[0], "problems": p_at[2], "judged": p_at[3],
               "reported": a["reported"], "comparable": a["comparable"], "authors": {k: a[k] for k in ("k", "max_tokens", "temperature", "top_p")}}
        if a["reported"] is not None:
            row["gap_at_cap"] = p_at[0] - a["reported"] / 100; row["gap_at_largest"] = p_top[0] - a["reported"] / 100
            if a["comparable"]: row["within_0.08"] = abs(row["gap_at_cap"]) <= 0.08; scored.append((m, row["within_0.08"]))
        res[side][m] = row
        rep = f"{a['reported']:5.1f}" if a["reported"] is not None else "  -  "
        gap = f"{row['gap_at_cap']:+.3f}" if "gap_at_cap" in row else "   -  "
        flag = "P5" if a["comparable"] else "  "
        print(f"{flag} {m:40s} ours@{at // 1024:>2d}k {p_at[0]:.3f} (fin {p_at[1]:.2f}{', judged' if p_at[3] else ''})  @{top // 1024:>2d}k {p_top[0]:.3f}  authors {rep}  gap@cap {gap}  "
              f"[{a['authors'] if False else ''}k={a['k']} {a['max_tokens']} t={a['temperature']}/{a['top_p']}]")
n_ok = sum(ok for m, ok in scored)
print(f"P5: {n_ok}/{len(scored)} comparable models within 0.08 at the recipe's cap" + (f"; outside: {', '.join(m for m, ok in scored if not ok)}" if n_ok < len(scored) else ""))
res["P5"] = {"within": n_ok, "comparable": len(scored), "outside": [m for m, ok in scored if not ok]}
if OUT: json.dump(res, open(OUT, "w"), indent=1)
