"""A work queue on a shared filesystem, and restart-safe partial outputs.

Several pods share one network volume. A model is *claimed* by creating ``<out>/<model>.lock``
with O_EXCL (atomic on the volume); a finished model has ``<out>/summary_<model>.json``.
A pod that dies mid-model leaves its lock; the chain script clears a lock whose owner is
the current host before retrying, and an operator clears a dead pod's lock by hand.

While a model generates, every finished problem's samples are appended to
``<out>/partial_<model>.jsonl`` and fsynced, so a dead pod loses nothing: on restart the
partial file is re-read, lines cut off mid-write are dropped, and only the missing
problems are requested again.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def summary_path(out: Path, model: str) -> Path:
    return Path(out) / f"summary_{model}.json"


def lock_path(out: Path, model: str) -> Path:
    return Path(out) / f"{model}.lock"


def partial_path(out: Path, model: str) -> Path:
    return Path(out) / f"partial_{model}.jsonl"


def claim(out: Path, model: str, host: str | None = None) -> bool:
    """True if this process now owns ``model``; False if it is done or claimed elsewhere."""
    if summary_path(out, model).exists():
        return False
    try:
        fd = os.open(lock_path(out, model), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, (host or socket.gethostname()).encode())
    os.close(fd)
    return True


def release(out: Path, model: str) -> None:
    try:
        os.remove(lock_path(out, model))
    except FileNotFoundError:
        pass


def lock_owner(out: Path, model: str) -> str | None:
    p = lock_path(out, model)
    return p.read_text().strip() if p.exists() else None


def load_partial(path: Path, items: list[dict], n: int) -> dict[int, dict]:
    """Problems already saved, by index; rewrites the file without any cut-off or stale line."""
    done: dict[int, dict] = {}
    if not Path(path).exists():
        return done
    good = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue  # a line cut off by a dead pod
        i = r.get("i")
        if isinstance(i, int) and 0 <= i < len(items) and r.get("id") == items[i]["id"] and len(r.get("samples", [])) == n:
            done[i] = r
            good.append(line if line.endswith("\n") else line + "\n")
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("".join(good), encoding="utf-8")
    os.replace(tmp, path)
    return done


def append_partial(handle, record: dict) -> None:
    handle.write(json.dumps(record) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_jsonl_atomic(path: Path, rows: Iterable[dict], gz: bool = False) -> None:
    """Write rows to ``path`` through a temp file so a reader never sees a half-written file."""
    import gzip

    tmp = Path(str(path) + ".tmp")
    opener = gzip.open if gz else open
    with opener(tmp, "wt", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
