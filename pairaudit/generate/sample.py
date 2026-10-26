"""Generation: n samples per problem for each model at one budget, saved restart-safely, then scored and read at every tighter budget.

One generation per model at its largest budget; every tighter budget is the same samples
re-read (``rows.read_at``). Models are read with their own released config: a model
whose config has no rope scaling and too few positions gets only ``max_position_embeddings``
raised to fit; a model that ships its own rope scaling (YaRN) is loaded unchanged.

Outputs under ``out``:
  partial_<model>.jsonl     every finished problem's raw samples, appended and fsynced (resume file)
  texts/<model>.jsonl.gz    every sample with its text, for later judging
  <model>_cap<C>.jsonl      the scored rows at the generation cap and at each tighter ladder budget
  meta_<model>.json         HF revision, GPU, vLLM version, settings, timings
  summary_<model>.json      per-benchmark cell summaries; its presence marks the model done

vLLM is imported lazily so the analysis side of the package installs without it.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import shutil
import socket
import time
from pathlib import Path

from pairaudit.analysis.rows import read_at, register_run, summarize
from pairaudit.generate import queue
from pairaudit.scoring.matcher import score

LADDER = (4096, 8192, 16384, 24576, 28672, 32768, 36864, 49152, 53248, 65536)
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def resolve_model(hf_id: str, cache_dirs: list[str], download_dir: str) -> str:
    """A local snapshot path: the first cache that holds the model, else a download."""
    org, name = hf_id.split("/")
    for cache in cache_dirs:
        hits = glob.glob(f"{cache}/hub/models--{org}--{name}/snapshots/*/config.json")
        if hits:
            return os.path.dirname(sorted(hits)[-1])
    from huggingface_hub import snapshot_download

    return snapshot_download(hf_id, cache_dir=f"{download_dir}/hub")


def native_overrides(path: str, gen_cap: int, margin: int = 2048) -> dict | None:
    """hf_overrides that keep the released config, raising only the position count when it is too small."""
    cfg = json.load(open(f"{path}/config.json"))
    need = gen_cap + margin
    if cfg.get("rope_scaling") or cfg.get("max_position_embeddings", 0) >= need:
        return None
    return {"max_position_embeddings": need}


def think_close_position(text: str, token_ids: list[int], tok, close_ids: list[int]) -> int:
    """Token position where the thinking block closes, or -1."""
    if "</think>" not in text:
        return -1
    if len(close_ids) == 1 and close_ids[0] in token_ids:
        return token_ids.index(close_ids[0])
    return len(tok.encode(text[: text.find("</think>")], add_special_tokens=False))


def generate_model(hf_id: str, items: list[dict], out: Path, *, gen_cap: int, n: int = 8, temperature: float = 0.6,
                   top_p: float = 0.95, seed: int = 0, tp: int = 1, k0: int = 0, derived: tuple[int, ...] | None = None,
                   cache_dirs: list[str] | None = None, download_dir: str = "/root/hf", margin: int = 2048,
                   instruction: str = INSTRUCTION, log=print) -> dict:
    """Generate, score and write one model. Returns the summary. Raises on load failure (the caller releases the lock)."""
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    name = hf_id.split("/")[-1]
    host = socket.gethostname()
    out = Path(out)
    (out / "texts").mkdir(parents=True, exist_ok=True)
    derived = tuple(c for c in LADDER if c < gen_cap) if derived is None else derived
    t0 = time.time()
    path = resolve_model(hf_id, cache_dirs or ["/workspace/.cache/huggingface"], download_dir)
    ov = native_overrides(path, gen_cap, margin)
    log(f"NATIVE {name}: override {ov}; {len(items)} problems at {gen_cap}, n={n}, T={temperature}, top_p={top_p}, seed={seed}")
    llm = LLM(model=path, tensor_parallel_size=tp, gpu_memory_utilization=0.9, max_model_len=gen_cap + margin,
              trust_remote_code=True, hf_overrides=ov, seed=seed, dtype="bfloat16")
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + instruction}], tokenize=False, add_generation_prompt=True) for it in items]
    log(f"PROMPT {name}: {prompts[0][:90]!r} ... {prompts[0][-70:]!r}")
    close_ids = tok.encode("</think>", add_special_tokens=False)
    t1 = time.time()
    part = queue.partial_path(out, name)
    done = queue.load_partial(part, items, n)
    log(f"RESUME {name}: {len(done)}/{len(items)} problems already saved")
    eng = llm.llm_engine
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=gen_cap, seed=seed, output_kind=RequestOutputKind.FINAL_ONLY)
    for i, p in enumerate(prompts):
        if i not in done:
            eng.add_request(str(i), p, sp)
    with open(part, "a", encoding="utf-8") as ph:
        while eng.has_unfinished_requests():
            for o in eng.step():
                if not o.finished:
                    continue
                i = int(o.request_id)
                samples = []
                for c in sorted(o.outputs, key=lambda c: c.index):
                    ids = list(c.token_ids)
                    samples.append({"text": c.text, "finish_reason": c.finish_reason, "len": len(ids),
                                    "pos": think_close_position(c.text, ids, tok, close_ids)})
                done[i] = {"i": i, "id": items[i]["id"], "samples": samples}
                queue.append_partial(ph, done[i])
                log(f"SAVED {name} {len(done)}/{len(items)} problems, {(time.time() - t1) / 3600:.2f} h")
    assert len(done) == len(items), (len(done), len(items))
    t2 = time.time()
    rows, texts = [], []
    for i, it in enumerate(items):
        for j, c in enumerate(done[i]["samples"]):
            fin = c["finish_reason"] == "stop"
            row = {"id": it["id"], "kind": it["kind"], "k": k0 + j, "finished": fin,
                   "score": score(c["text"], it["gt"]) if fin else 0.0, "len": c["len"], "think_closed": c["pos"] >= 0}
            rows.append(row)
            texts.append({**row, "text": c["text"]})
    queue.write_jsonl_atomic(out / "texts" / f"{name}.jsonl.gz", texts, gz=True)
    t3 = time.time()
    summary = {}
    for cap in sorted((*derived, gen_cap)):
        at = rows if cap == gen_cap else read_at(rows, cap)
        queue.write_jsonl_atomic(out / f"{name}_cap{cap}.jsonl", at)
        for kind, d in summarize(at).items():
            summary[f"{name} cap{cap} {kind}"] = d
            log(f"PAIR {name} cap{cap} {kind:5s} mean {d['mean']:.3f} | finished {d['finished']:.2f} | score|finished {d['score_given_finished']:.3f} | any-solve {d['any_solve']:.3f} | tokens {d['mean_len']:.0f}")
    meta = {"model": hf_id, "path": path, "revision": os.path.basename(path) if "/snapshots/" in path else None, "host": host,
            "gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__, "dtype": "bfloat16", "hf_overrides": ov,
            "gen_cap": gen_cap, "derived_caps": list(derived), "n": n, "k0": k0, "temperature": temperature, "top_p": top_p,
            "seed": seed, "close_ids": close_ids, "n_items": len(items),
            "seconds": {"load": round(t1 - t0), "generate": round(t2 - t1), "score": round(t3 - t2)}}
    queue.write_json(out / f"meta_{name}.json", meta)
    queue.write_json(out / f"summary_{name}.json", summary)
    register_run(out.parent, out.name, {name: {"generation_cap": gen_cap, "caps": sorted((*derived, gen_cap))}}, nested=True)
    log(f"DONE {name}: generate {(t2 - t1) / 3600:.2f} h, score {(t3 - t2) / 60:.1f} min")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if path.startswith(download_dir):
        org, nm = hf_id.split("/")
        shutil.rmtree(f"{download_dir}/hub/models--{org}--{nm}", ignore_errors=True)
    return summary


def run_queue(models: list[str], items: list[dict], out: Path, spec, *, default_cap: int, n: int = 8, seed: int = 0,
              targets: dict | None = None, k0: int = 0, log=print, **kw) -> list[str]:
    """Claim each model in turn through the lock queue and generate it. ``targets`` (reach mode) restricts each model to its own problem ids and cap."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    finished = []
    for hf_id in models:
        name = hf_id.split("/")[-1]
        if not queue.claim(out, name):
            log(f"skip {name} (done or claimed elsewhere)")
            continue
        its, cap = items, spec[name][2] if spec else default_cap
        temperature, top_p = (spec[name][0], spec[name][1]) if spec else (kw.pop("temperature", 0.6), kw.pop("top_p", 0.95))
        derived = None
        if targets is not None:
            t = targets[name]
            wanted = set(t["ids"])
            its, cap, derived = [it for it in items if it["id"] in wanted], int(t["cap"]), ()
        try:
            generate_model(hf_id, its, out, gen_cap=cap, n=n, temperature=temperature, top_p=top_p, seed=seed, k0=k0, derived=derived, log=log, **kw)
            finished.append(name)
        except Exception as e:  # noqa: BLE001 - a load failure must release the lock and let the queue continue
            log(f"SKIP {hf_id}: {type(e).__name__}: {str(e)[:300]}")
            queue.release(out, name)
    return finished
