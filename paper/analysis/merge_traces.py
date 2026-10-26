"""Merge the first eight's trace analysis (trace_out) with the ten added (trace_out_v2) for trace_tables.py.
usage: python exp2/merge_traces.py trace_out trace_out_v2 trace_out_all"""
import json, sys, os, shutil, glob
A, B, O = sys.argv[1:4]; os.makedirs(O, exist_ok=True)
sa, sb = json.load(open(f"{A}/trace_summary.json")), json.load(open(f"{B}/trace_summary.json"))
for k in set(sa) & set(sb): assert sa[k] == sb[k], k   # the shared base (R1-Distill-1.5B) comes from the same trace file
json.dump({**sa, **sb}, open(f"{O}/trace_summary.json", "w"), indent=1)
for src in (A, B):
    for f in glob.glob(f"{src}/reading_*.json"): shutil.copy(f, O)
ra, rb = json.load(open(f"{A}/repertoire_check.json")), json.load(open(f"{B}/repertoire_check.json"))
json.dump({"by_model": {**ra["by_model"], **rb["by_model"]}}, open(f"{O}/repertoire_check.json", "w"), indent=1)
print(len(sa) + len(sb) - len(set(sa) & set(sb)), "models;", len(glob.glob(f"{O}/reading_*.json")), "readings")
