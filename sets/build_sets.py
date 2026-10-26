#!/usr/bin/env python3
"""Rebuild the hard671 and math276 evaluation sets from their public sources.

The released files sets/hard671.jsonl and sets/math276.jsonl carry ids, golds
and q_sha256 (sha256 of the exact UTF-8 problem text the models saw) but no
third-party problem text. This script downloads each source at a pinned
revision, reconstructs the problem text q for every id, checks it against
q_sha256, and writes sets/build/{hard671,math276}.jsonl with the text filled in.

Golds (gt) and all other metadata come from our sets files, never from the
source. Any hash mismatch or missing id makes the script exit nonzero; the
hash check is exact and is never relaxed.

Usage:
    pip install datasets huggingface_hub
    python sets/build_sets.py [--cache-dir DIR] [--out-dir DIR]

See sets/SOURCES.md for dataset ids, revisions, licences and transforms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Pinned sources. Revisions are Hugging Face dataset commit hashes.
# ---------------------------------------------------------------------------
REV = {
    "Hothan/OlympiadBench": "91184b52131e7fc9455fef848035173aea8cc01a",
    "UGPhysics/ugphysics": "523e71bc4e33356bb19de4af38c128796e8a8770",
    "desimfj/PHYSICS": "6479757368c7d620ab826ba3ff7af44cb7eb3b5d",
    "Eureka-Lab/PHYBench": "d6d91c787b7abb865eb2490a328bf85a9f5095f0",
    "openai/frontierscience": "25ed67db7da8f4591484e764008ff585544f5a30",
    "MathArena/aime_2025": "c94da77eb22bbd6439e62a323bec18493a421302",
    "MathArena/aime_2026": "d61f486805306e6e6b5913504c07763a99e1b112",
    "MathArena/hmmt_feb_2025": "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
    "MathArena/hmmt_feb_2026": "589a53fe152febc0f4f99f3e561007bf8da8e664",
    "MathArena/hmmt_nov_2025": "118dbfb45c4c9467c672268ed55166642897aa46",
    "MathArena/brumo_2025": "6de9bc0987f5710c6376ee980e1483dbeb55cfda",
    "MathArena/cmimc_2025": "7b3ca7c4b417d31b1f97f781f39fc4f97f1b9185",
    "MathArena/smt_2025": "d8e99d83f1583842f46d4953cd960eea5d887e45",
}

UGPHYSICS_SUBJECTS = [
    "AtomicPhysics", "ClassicalElectromagnetism", "ClassicalMechanics",
    "Electrodynamics", "GeometricalOptics", "QuantumMechanics", "Relativity",
    "SemiconductorPhysics", "Solid-StatePhysics", "StatisticalMechanics",
    "TheoreticalMechanics", "Thermodynamics", "WaveOptics",
]

MATHARENA = [
    "aime_2025", "aime_2026", "hmmt_feb_2025", "hmmt_feb_2026",
    "hmmt_nov_2025", "brumo_2025", "cmimc_2025", "smt_2025",
]


class Fetcher:
    def __init__(self, cache_dir: str | None):
        self.cache_dir = cache_dir

    def file(self, repo: str, filename: str) -> str:
        return hf_hub_download(repo, filename, repo_type="dataset",
                               revision=REV[repo], cache_dir=self.cache_dir)

    def jsonl(self, repo: str, filename: str) -> list[dict]:
        with open(self.file(repo, filename), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def json(self, repo: str, filename: str):
        with open(self.file(repo, filename), encoding="utf-8") as f:
            return json.load(f)

    def parquet(self, repo: str, filename: str, columns: list[str]) -> list[dict]:
        ds = Dataset.from_parquet(self.file(repo, filename), columns=columns,
                                  cache_dir=self.cache_dir)
        return ds.to_list()


# ---------------------------------------------------------------------------
# Per-source loaders: each returns {key: [q, ...]} where key is the part of
# our id after "<src>|" and q is the exact text the models saw.
# ---------------------------------------------------------------------------
def _index(pairs) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for k, q in pairs:
        out[k].append(q)
    return out


def load_olympiadbench(fx: Fetcher):
    # Text-only, open-ended, English, physics, competition split.
    # Transform: if the row has a non-empty shared `context`, the context is
    # prepended to the question with a blank line: context + "\n\n" + question.
    repo = "Hothan/OlympiadBench"
    rows = fx.parquet(repo, "OlympiadBench/OE_TO_physics_en_COMP/OE_TO_physics_en_COMP.parquet",
                      ["id", "context", "question"])
    def q(r):
        return r["context"] + "\n\n" + r["question"] if r["context"] else r["question"]
    return _index((str(r["id"]), q(r)) for r in rows)


def load_ugphysics(fx: Fetcher):
    # English split of every subject; key = "<subject>_<index>" where subject is
    # the row's `subject` field (with spaces, e.g. "Quantum Mechanics").
    # Transform: none (`problem` verbatim).
    repo = "UGPhysics/ugphysics"
    pairs = []
    for subj in UGPHYSICS_SUBJECTS:
        for r in fx.jsonl(repo, f"{subj}/en.jsonl"):
            pairs.append((f"{r['subject']}_{r['index']}", r["problem"]))
    return _index(pairs)


def load_physics_test(fx: Fetcher):
    # data/test.jsonl holds each problem twice (language "zh" and "en") under
    # the same id; we use the English row. Transform: none (`question`).
    repo = "desimfj/PHYSICS"
    rows = fx.jsonl(repo, "data/test.jsonl")
    return _index((str(r["id"]), r["question"]) for r in rows if r["language"] == "en")


def load_phybench(fx: Fetcher):
    # The 100 fully released problems (PHYBench-fullques_v1.json).
    # Transform: none (`content`).
    repo = "Eureka-Lab/PHYBench"
    rows = fx.json(repo, "PHYBench-fullques_v1.json")
    return _index((str(r["id"]), r["content"]) for r in rows)


def load_frontierscience(fx: Fetcher):
    # Olympiad (short-answer) track, physics subject; key = task_group_id.
    # Transform: none (`problem`).
    repo = "openai/frontierscience"
    rows = fx.jsonl(repo, "olympiad/test.jsonl")
    return _index((r["task_group_id"], r["problem"]) for r in rows if r["subject"] == "physics")


def make_matharena_loader(name: str):
    # key = problem_idx. Transform: str.strip() of `problem` (the HF rows carry
    # stray leading/trailing newlines on some problems).
    def load(fx: Fetcher):
        repo = f"MathArena/{name}"
        rows = fx.parquet(repo, "data/train-00000-of-00001.parquet", ["problem_idx", "problem"])
        return _index((str(r["problem_idx"]), r["problem"].strip()) for r in rows)
    return load


SETS = {
    "hard671": {
        "OB-phys-en": load_olympiadbench,
        "UGPhysics-en-1part": load_ugphysics,
        "PHYSICS-test": load_physics_test,
        "PHYBench": load_phybench,
        "FrontierSci-phys-short": load_frontierscience,
    },
    "math276": {name: make_matharena_loader(name) for name in MATHARENA},
}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_set(set_name: str, loaders: dict, fx: Fetcher, out_dir: Path) -> tuple[list, bool]:
    with open(HERE / f"{set_name}.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_src[r["src"]].append(r)
    unknown = sorted(set(by_src) - set(loaders))
    if unknown:
        raise SystemExit(f"{set_name}: no loader for sources {unknown}")

    table, out, ok_all = [], [], True
    for src, loader in loaders.items():
        items = by_src.get(src, [])
        print(f"[{set_name}] loading {src} ...", file=sys.stderr, flush=True)
        index = loader(fx)
        matched = mismatched = missing = 0
        problems = []
        for r in items:
            key = r["id"].split("|", 1)[1]
            cands = index.get(key, [])
            if not cands:
                missing += 1
                problems.append(f"  MISSING   {r['id']}: key {key!r} not found in source")
                continue
            if len(cands) > 1:
                mismatched += 1
                problems.append(f"  AMBIGUOUS {r['id']}: key {key!r} maps to {len(cands)} source rows")
                continue
            q = cands[0]
            if sha256(q) != r["q_sha256"]:
                mismatched += 1
                problems.append(f"  MISMATCH  {r['id']}: sha256 {sha256(q)[:16]}... != {r['q_sha256'][:16]}...")
                continue
            matched += 1
            rec = {"id": r["id"], "src": r["src"], "kind": r["kind"], "q": q, "gt": r["gt"]}
            rec.update({k: v for k, v in r.items() if k not in rec})
            out.append(rec)
        table.append((set_name, src, len(items), matched, mismatched, missing))
        for p in problems:
            print(p)
        if mismatched or missing:
            ok_all = False

    # Preserve the order of the sets file.
    order = {r["id"]: i for i, r in enumerate(recs)}
    out.sort(key=lambda r: order[r["id"]])
    if ok_all:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{set_name}.jsonl", "w", encoding="utf-8") as f:
            for rec in out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return table, ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", default=None, help="Hugging Face cache dir (default: HF default)")
    ap.add_argument("--out-dir", default=str(HERE / "build"), help="output dir (default: sets/build)")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    args = ap.parse_args()

    fx = Fetcher(args.cache_dir)
    rows, ok = [], True
    for name in args.sets:
        t, good = build_set(name, SETS[name], fx, Path(args.out_dir))
        rows += t
        ok &= good
        if not good:
            print(f"{name}: NOT written (mismatched or missing ids above)")

    hdr = ("set", "source", "ids", "matched", "mismatched", "missing")
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    line = lambda r: "  ".join(str(x).ljust(n) if i < 2 else str(x).rjust(n) for i, (x, n) in enumerate(zip(r, w)))
    print(line(hdr))
    for r in rows:
        print(line(r))
    tot = [sum(r[i] for r in rows) for i in range(2, 6)]
    print(line(("TOTAL", "", *tot)))
    print("OK: all ids reproduced exactly" if ok else "FAIL: some ids were not reproduced exactly")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
