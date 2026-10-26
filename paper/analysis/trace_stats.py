"""Trace analysis: eight released RL models vs their bases on the same 300 problems (150 physics, 150 MATH-500), 4 samples, 16k cap.
Per model: finishing, accuracy, finished length, hesitation words per finished derivation (the paper's measure: wait / let me re / actually / hmm),
verification words, and why unfinished derivations are unfinished (runaway repetition vs still working).
Per pair: the same on problems both models finish (>= 3 of 4), and a set of discordant problems (RL finished-and-right, base finished-and-wrong)
written out for side-by-side reading. usage: python trace_stats.py traces_dir out_dir"""
import json, sys, os, re, zlib, random, collections, statistics as st
D, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
from families import selected
PAIRS = selected()
HES = re.compile(r"\b(wait|let me re|actually|hmm)\b", re.I)
VER = re.compile(r"\b(verify|double-check|double check|let me check|check (?:this|that|again|the)|plug(?:ging)? (?:back|in)|sanity check|confirm)\b", re.I)
def think(t):
    i = t.find("</think>"); return t if i < 0 else t[:i]
def loop_score(t):
    tail = t[-6000:]
    if len(tail) < 2000: return 0.0
    return 1 - len(zlib.compress(tail.encode(), 9)) / len(tail.encode())   # high = repetitive
def repeated_sentences(t):
    s = [x.strip() for x in re.split(r"[.!?\n]", t[-8000:]) if len(x.strip()) > 25]
    c = collections.Counter(s); return max(c.values()) if c else 0
def load(m):
    p = f"{D}/{m}.jsonl"
    if not os.path.exists(p): return None
    by = collections.defaultdict(list)
    for l in open(p):
        if l.strip(): r = json.loads(l); by[(r["kind"], r["id"])].append(r)
    return by
def per_model(by):
    rows = [r for v in by.values() for r in v]; fin = [r for r in rows if r["finished"]]; unf = [r for r in rows if not r["finished"]]
    out = {"n": len(rows), "finished": len(fin) / len(rows), "mean_score": st.mean(r["score"] for r in rows),
           "acc_given_finished": st.mean(r["score"] for r in fin) if fin else None,
           "tokens_finished": st.mean(r["len"] for r in fin) if fin else None,
           "hesitation_per_finished": st.mean(len(HES.findall(think(r["text"]))) for r in fin) if fin else None,
           "hesitation_per_1k_tokens": (sum(len(HES.findall(think(r["text"]))) for r in fin) / (sum(r["len"] for r in fin) / 1000)) if fin else None,
           "verification_per_finished": st.mean(len(VER.findall(think(r["text"]))) for r in fin) if fin else None,
           "unfinished_runaway_share": (sum(1 for r in unf if repeated_sentences(r["text"]) >= 5 or loop_score(r["text"]) > 0.85) / len(unf)) if unf else None,
           "n_unfinished": len(unf)}
    for kind in ("phys", "math"):
        rk = [r for (k, _), v in by.items() if k == kind for r in v]; fk = [r for r in rk if r["finished"]]
        out[f"{kind}_finished"] = len(fk) / len(rk); out[f"{kind}_mean_score"] = st.mean(r["score"] for r in rk)
        out[f"{kind}_acc_given_finished"] = st.mean(r["score"] for r in fk) if fk else None
    return out
Q = {}
for l in open(f"{D}/phys300.jsonl"):
    r = json.loads(l); Q[("phys", r["id"])] = (r["q"], r["gt"])
_m = []
for l in open(f"{D}/math500_test.jsonl"):
    r = json.loads(l); Q[("math", r.get("unique_id", str(len(_m))))] = (r["problem"], r["answer"]); _m.append(1)
summary = {}; rng = random.Random(3)
for base, recs in PAIRS.items():
    B = load(base)
    if B is None: continue
    summary[base] = per_model(B)
    for rec in recs:
        R = load(rec)
        if R is None: continue
        summary[rec] = s = per_model(R); s["base"] = base
        both = [k for k in B if k in R and sum(r["finished"] for r in B[k]) >= 3 and sum(r["finished"] for r in R[k]) >= 3]
        fb = [r for k in both for r in B[k] if r["finished"]]; fr = [r for k in both for r in R[k] if r["finished"]]
        s["matched_n"] = len(both)
        if both:
            s["matched_len_ratio"] = st.mean(r["len"] for r in fr) / st.mean(r["len"] for r in fb)
            s["matched_hes_base"] = st.mean(len(HES.findall(think(r["text"]))) for r in fb); s["matched_hes_rl"] = st.mean(len(HES.findall(think(r["text"]))) for r in fr)
            s["matched_ver_base"] = st.mean(len(VER.findall(think(r["text"]))) for r in fb); s["matched_ver_rl"] = st.mean(len(VER.findall(think(r["text"]))) for r in fr)
            s["matched_acc_base"] = st.mean(r["score"] for r in fb); s["matched_acc_rl"] = st.mean(r["score"] for r in fr)
        # discordant problems: RL has a finished full-credit sample, base has finished samples but none with full credit
        disc = []
        for k in B:
            if k not in R: continue
            rr = [r for r in R[k] if r["finished"] and r["score"] >= 1 - 1e-9]; bw = [r for r in B[k] if r["finished"] and r["score"] < 1 - 1e-9]
            if rr and bw and not any(r["finished"] and r["score"] >= 1 - 1e-9 for r in B[k]): disc.append((k, rr[0], bw[0]))
        s["discordant_rl_right_base_wrong"] = len(disc)
        rev = sum(1 for k in B if k in R and any(r["finished"] and r["score"] >= 1 - 1e-9 for r in B[k]) and not any(r["finished"] and r["score"] >= 1 - 1e-9 for r in R[k]) and any(r["finished"] for r in R[k]))
        s["discordant_base_right_rl_wrong"] = rev
        (random.Random("disc:" + rec) if os.environ.get("TRACE_RNG") == "per_recipe" else rng).shuffle(disc)
        with open(f"{O}/discordant_{rec}.jsonl", "w") as h:
            for (kind, pid), r1, r0 in disc[:12]:
                q, gt = Q.get((kind, pid), ("", ""))
                h.write(json.dumps({"kind": kind, "id": pid, "rl_model": rec, "base_model": base, "question": q, "gold": gt,
                                    "rl_text": r1["text"], "base_text": r0["text"], "rl_len": r1["len"], "base_len": r0["len"], "base_score": r0["score"]}) + "\n")
json.dump(summary, open(f"{O}/trace_summary.json", "w"), indent=1)
for m, s in summary.items():
    base = s.get("base")
    line = f"{m[:34]:34s} fin {s['finished']:.2f} score {s['mean_score']:.3f} acc|fin {s['acc_given_finished'] or 0:.3f} tok(fin) {s['tokens_finished'] or 0:6.0f} hes/fin {s['hesitation_per_finished'] or 0:5.1f} hes/1k {s['hesitation_per_1k_tokens'] or 0:4.1f} ver/fin {s['verification_per_finished'] or 0:4.1f} runaway-of-unfinished {s['unfinished_runaway_share'] if s['unfinished_runaway_share'] is not None else float('nan'):.2f} (n={s['n_unfinished']})"
    if base: line += f" | matched n={s.get('matched_n')} len x{s.get('matched_len_ratio', float('nan')):.2f} hes {s.get('matched_hes_base', float('nan')):.1f}->{s.get('matched_hes_rl', float('nan')):.1f} acc {s.get('matched_acc_base', float('nan')):.3f}->{s.get('matched_acc_rl', float('nan')):.3f} | discordant RL>base {s['discordant_rl_right_base_wrong']} base>RL {s['discordant_base_right_rl_wrong']}"
    print(line)
