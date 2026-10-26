"""Every quantity the paper's public-pair passages quote, from pairs_ci.json (printed-interval flags). usage: python pairs_counts.py pairs_ci.json"""
import json, sys, statistics
d = json.load(open(sys.argv[1])); C = list(d.values())
gain = [c for c in C if c["capped_gain"]]; loss = [c for c in C if c["capped_loss"]]
shares = sorted(c["finishing_share"] for c in gain)
print(f"cells {len(C)}; models {sorted({c['rl'] for c in C})}")
print(f"capped: rises {len(gain)}, falls {len(loss)} ({[(c['rl'][:10], c['set'], c['cap']) for c in loss]})")
print(f"finishing share over rising cells: min {shares[0]:.3f} median {statistics.median(shares):.3f} max {shares[-1]:.3f}; >=1 in {sum(s >= 1 for s in shares)}; share CI below 1/2 in {sum(c['finishing_share_ci'][0] < 0.5 for c in gain)}; alt split <1/2: {[(c['rl'][:10], c['set'], c['cap'], round(c['finishing_share_alt'], 2)) for c in gain if c['finishing_share_alt'] < 0.5]}; per-finished term >0 in {sum(c['perfinished_term'] > 0 for c in gain)}")
print(f"finishing: rises in {sum(c['finishing_gain'] for c in C)} (range {min(c['finished']['delta'] for c in C if c['finishing_gain']):+.3f}..{max(c['finished']['delta'] for c in C if c['finishing_gain']):+.3f}), falls in {sum(c['finishing_loss'] for c in C)} (range {min([c['finished']['delta'] for c in C if c['finishing_loss']] or [0]):+.3f}..{max([c['finished']['delta'] for c in C if c['finishing_loss']] or [0]):+.3f})")
pf_g = [c for c in C if c["perfinished_gain"]]; pf_l = [c for c in C if c["perfinished_loss"]]
print(f"full-set acc|fin: rises in {len(pf_g)}, falls in {len(pf_l)} {[(c['rl'][:10], c['set'], c['cap']) for c in pf_l]}; boundary cells (|bound|<0.0015): {[(c['rl'][:10], c['set'], c['cap'], [round(x, 4) for x in c['sgf']['ci']]) for c in C if min(abs(c['sgf']['ci'][0]), abs(c['sgf']['ci'][1])) < 0.0015]}")
hid = [c for c in C if c["perfinished_gain"] and not c["capped_gain"]]
print(f"hidden (pf gain, no capped gain): {len(hid)}: {[(c['rl'][:10], c['set'], c['cap'], round(c['len_ratio'], 2), round(c['finished']['delta'], 2)) for c in hid]}")
m_rep = [c for c in C if "delta" in c["matched"]]; m_g = [c for c in m_rep if c["matched"]["gain"]]; m_l = [c for c in m_rep if c["matched"]["loss"]]
print(f"matched: reported {len(m_rep)}, rises {len(m_g)}, falls {len(m_l)}; unreported n: {[(c['rl'][:10], c['set'], c['cap'], c['matched']['n']) for c in C if 'delta' not in c['matched']]}")
for c in sorted(m_g, key=lambda c: c["matched"]["delta"]): print(f"   matched gain {c['rl'][:22]:22s} {c['set']} {c['cap']//1024:2d}k {c['matched']['delta']:+.3f} [{c['matched']['ci'][0]:+.3f},{c['matched']['ci'][1]:+.3f}] n={c['matched']['n']}")
print("matched n at tight caps, physics:", sorted(c["matched"]["n"] for c in C if c["set"] == "phys" and c["cap"] == (4096 if "1.5B" in c["base"] else 8192)), "MATH:", sorted(c["matched"]["n"] for c in C if c["set"] == "math" and c["cap"] == (4096 if "1.5B" in c["base"] else 8192)))
print("matched base acc, MATH cells: %.2f..%.2f" % (min(c["matched"]["base"] for c in m_rep if c["set"] == "math"), max(c["matched"]["base"] for c in m_rep if c["set"] == "math")))
for m in sorted({c["rl"] for c in C}):
    cs = [c for c in C if c["rl"] == m]
    print(f"{m:40s} cells {len(cs)} capped+ {sum(c['capped_gain'] for c in cs)} fin+ {sum(c['finishing_gain'] for c in cs)} fin- {sum(c['finishing_loss'] for c in cs)} pf+ {sum(c['perfinished_gain'] for c in cs)} pf- {sum(c['perfinished_loss'] for c in cs)} matched+ {sum(c['matched'].get('gain', False) for c in cs)} | pf deltas {[round(c['sgf']['delta'], 3) for c in cs]} | tokens {[round(c['len_ratio'], 2) for c in cs]}")
# --- two-direction summary (added after the 7B 4k and Skywork-OR1-Math-7B cells) ---
mov = [c for c in C if c["capped_gain"] or c["capped_loss"]]
ok = [c for c in mov if c["finishing_share"] is not None and c["finishing_share"] >= 0.5]
print(f"\nMOVING cells (capped change outside its interval): {len(mov)} of {len(C)} ({len(gain)} rises, {len(loss)} falls); finishing term >= half of the move in {len(ok)}; exceptions: {[(c['rl'][:14], c['set'], c['cap'], round(c['finishing_share'], 2), round(c['mean']['delta'], 3), round(c['finished']['delta'], 3)) for c in mov if c not in ok]}")
print(f"  falls: finishing falls in {sum(c['finishing_loss'] for c in loss)} of {len(loss)}; per-finished term positive in {sum(c['perfinished_term'] > 0 for c in loss)}; tokens {min(c['len_ratio'] for c in loss):.2f}-{max(c['len_ratio'] for c in loss):.2f}x; caps {sorted(set(c['cap'] for c in loss))}")
ser = {}
for c in C: ser.setdefault((c["rl"], c["set"]), []).append(c)
flips = [(m, s) for (m, s), cs in ser.items() if any(c["capped_gain"] for c in cs) and any(c["capped_loss"] for c in cs)]
print(f"  series (model x set) whose capped change has both signs across caps: {len(flips)}: {flips}")
for m in sorted({c['rl'] for c in C}):
    cs = [c for c in C if c['rl'] == m]; print(f"  regime {m[:28]:28s} median tokens {statistics.median(c['len_ratio'] for c in cs):.2f}x  finishing +{sum(c['finishing_gain'] for c in cs)}/-{sum(c['finishing_loss'] for c in cs)}  capped +{sum(c['capped_gain'] for c in cs)}/-{sum(c['capped_loss'] for c in cs)}")
outside = [c["matched"]["delta"] for c in m_g if not c["rl"].startswith("Nemotron-Research")]; prorl = [c["matched"]["delta"] for c in m_g if c["rl"].startswith("Nemotron-Research")]
print(f"  matched gains outside ProRL: {min(outside):+.3f}..{max(outside):+.3f} ({len(outside)} cells); ProRL {min(prorl):+.3f}..{max(prorl):+.3f} ({len(prorl)}); matched falls {len(m_l)}; matched reported {len(m_rep)}")
pfg_full = [c for c in C if c['perfinished_gain']]
print(f"  full-set per-finished gains: {len(pfg_full)} cells, of which with finishing falling {sum(c['finishing_loss'] for c in pfg_full)} (selection inflates these)")
