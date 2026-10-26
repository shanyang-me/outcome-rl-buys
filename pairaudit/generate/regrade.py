"""Judge regrade of stored texts under protocol v2: the deterministic checker decides what it can, a pinned open judge the rest.

For every physics sample that finished within its generation cap and that the frozen
matcher scored below full credit, each gold part is graded: ``judge_protocol.decide``
first, and the deferred parts go to the judge model with the protocol's rules. A sample's
judged score is the share of its gold parts judged right, never below the matcher's score.
Rows come out as the input rows plus ``score_j``; a verdict file caches every judged part
so a re-run judges only what is new.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from pairaudit.analysis.rows import iter_jsonl
from pairaudit.generate import queue
from pairaudit.scoring.boxed import extract_boxed
from pairaudit.scoring.judge_protocol import decide, parse_verdict, prompt_for

FULL = 1 - 1e-9


def gold_parts(items_path: Path) -> dict[str, tuple[list[str], float | None]]:
    """id -> (golds, abs_tol) from an items file."""
    out = {}
    for it in iter_jsonl(Path(items_path)):
        rec = json.loads(it["gt"])
        out[it["id"]] = ([str(g) for g in rec["golds"]], rec.get("tol"))
    return out


def collect(text_files: list[tuple[str, Path]], golds: dict, kinds=("phys",)) -> tuple[dict, dict, dict]:
    """Split every rejected sample's gold parts into decided and deferred; returns (rows_by_file, decided, deferred prompts)."""
    rows_by_file, decided, deferred = {}, {}, {}
    for tag, f in text_files:
        model = f.name[: -len(".jsonl.gz")]
        rows = []
        for r in iter_jsonl(f):
            text = r.pop("text")
            rows.append(r)
            if r["kind"] not in kinds or not r["finished"] or r["score"] >= FULL:
                continue
            preds = extract_boxed(text) or []
            gs, tol = golds[r["id"]]
            for j, g in enumerate(gs):
                key = f"{tag}|{model}|{r['id']}|{r['k']}|{j}"
                v = decide(g, preds, tol)
                if v is None:
                    deferred[key] = prompt_for(g, preds)
                else:
                    decided[key] = bool(v)
        rows_by_file[(tag, model)] = rows
    return rows_by_file, decided, deferred


def judge(deferred: dict[str, str], verdict_file: Path, model: str = "Qwen/Qwen3-32B", batch: int = 4000, max_tokens: int = 4096,
          max_model_len: int = 16384, log=print) -> dict[str, str]:
    """Verdicts for every deferred key, reading the cache first and judging only what is missing."""
    done = {}
    if verdict_file.exists():
        for o in iter_jsonl(verdict_file):
            done[o["key"]] = o["verdict"]
    todo = [k for k in deferred if k not in done]
    log(f"deferred {len(deferred)}; cached {len(deferred) - len(todo)}; judging {len(todo)}")
    if todo:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model, gpu_memory_utilization=0.9, max_model_len=max_model_len, seed=0)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template([{"role": "user", "content": deferred[k][:24000]}], tokenize=False, add_generation_prompt=True) for k in todo]
        for s in range(0, len(todo), batch):
            outs = llm.generate(prompts[s : s + batch], SamplingParams(temperature=0.0, max_tokens=max_tokens))
            with open(verdict_file, "a", encoding="utf-8") as h:
                for k, o in zip(todo[s : s + batch], outs):
                    v = parse_verdict(o.outputs[0].text)
                    done[k] = v
                    h.write(json.dumps({"key": k, "verdict": v, "raw_tail": o.outputs[0].text[-80:]}) + "\n")
            log(f"judged {min(s + batch, len(todo))}/{len(todo)}")
    return done


def regrade(out: Path, src_dirs: list[Path], items_path: Path, judge_model: str = "Qwen/Qwen3-32B", log=print) -> dict:
    """Regrade every ``<src>/texts/*.jsonl.gz``; write ``<out>/<src name>/<model>.jsonl`` with score_j and a summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    golds = gold_parts(items_path)
    files = [(Path(d).name, f) for d in src_dirs for f in sorted(Path(d).glob("texts/*.jsonl.gz"))]
    log(f"{len(files)} text files")
    rows_by_file, decided, deferred = collect(files, golds)
    verdicts = judge(deferred, out / "judge_verdicts.jsonl", judge_model, log=log)
    parts = dict(decided)
    for k in deferred:
        parts[k] = verdicts.get(k) == "YES"
    unparsed = sum(1 for k in deferred if verdicts.get(k) == "UNPARSED")
    stats = {}
    for (tag, model), rows in rows_by_file.items():
        (out / tag).mkdir(exist_ok=True)
        c = collections.Counter()
        judged = []
        for r in rows:
            sj = r["score"]
            if r["kind"] == "phys" and r["finished"] and r["score"] < FULL:
                gs, _ = golds[r["id"]]
                v = [parts[f"{tag}|{model}|{r['id']}|{r['k']}|{j}"] for j in range(len(gs))]
                sj = max(r["score"], sum(v) / len(v))
                c["rejected"] += 1
                c["raised"] += sj > r["score"] + 1e-9
                c["now_full"] += sj >= FULL
            judged.append({**r, "score_j": sj})
        queue.write_jsonl_atomic(out / tag / f"{model}.jsonl", judged)
        stats[f"{tag}/{model}"] = dict(c)
    summary = {"files": stats, "decided": len(decided), "deferred": len(deferred), "unparsed": unparsed}
    queue.write_json(out / "summary.json", summary)
    log(f"REGRADE_DONE {json.dumps({'deferred': len(deferred), 'unparsed': unparsed})}")
    return summary
