"""Section 4.2 'Why a recipe moves finishing or reliability': per-recipe physics split at the training cap against how much the base
leaves unfinished there. usage: python why_split.py cap_split_phys.json [out.json]"""
import json, sys, statistics as st
R = json.load(open(sys.argv[1]))["recipes"]
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def rho(a, b):
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
rows = []
for n, r in R.items():
    if r.get("below_only"): continue
    c = r["at"]; t = abs(c["F"]) + abs(c["P"])
    rows.append({"n": n, "cap": r["train_cap"], "fb": c["fb"], "ab": c["ab"], "F": c["F"], "P": c["P"], "d": c["d"], "share": abs(c["F"]) / t if t else 0.5, "head": 1 - c["fb"]})
share = [x["share"] for x in rows]; head = [x["head"] for x in rows]; cap = [x["cap"] for x in rows]
room = [x for x in rows if x["fb"] < 0.9]; full = [x for x in rows if x["fb"] >= 0.9]
out = {"n": len(rows), "rho_share_head": rho(share, head), "rho_share_cap": rho(share, cap),
       "room": [x["n"] for x in room], "room_share": st.mean(x["share"] for x in room), "room_P": st.mean(x["P"] for x in room),
       "full_n": len(full), "full_share": st.mean(x["share"] for x in full), "full_P": st.mean(x["P"] for x in full),
       "full_fb_range": [min(x["fb"] for x in full), max(x["fb"] for x in full)],
       "rho_P_ab": rho([x["P"] for x in rows], [x["ab"] for x in rows]), "rho_P_cap": rho([x["P"] for x in rows], cap),
       "finishing_leads": sum(x["share"] > 0.5 for x in rows)}
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}))
if len(sys.argv) > 2: json.dump(rows, open(sys.argv[2], "w"), indent=1)
