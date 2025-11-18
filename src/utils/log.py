import os, json, time
from typing import Any, Dict, Optional

def utc_rfc3339() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def ensure_parent(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def make_row(
    *,
    trial: int,
    outcome: str,
    seed: int,
    device: str,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    layer_id: Optional[str] = None,
    layer_idx: Optional[int] = None,
    p: Optional[float] = None,
    bit_mode: Optional[str | int] = None,
    elements_seen: Optional[int] = None,
    elements_flipped: Optional[int] = None,
    avg_latency_ms_per_sample: Optional[float] = None,
    changed: Optional[int] = None,
    changed_rate: Optional[float] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp": utc_rfc3339(),
        "trial": trial,
        "outcome": outcome,
        "seed": seed,
        "device": device,
    }
    for k, v in {
        "model": model, "dataset": dataset,
        "layer_id": layer_id, "layer_idx": layer_idx,
        "p": p, "bit_mode": bit_mode,
        "elements_seen": elements_seen, "elements_flipped": elements_flipped,
        "avg_latency_ms_per_sample": avg_latency_ms_per_sample,
        "changed": changed, "changed_rate": changed_rate,
        "error": error,
    }.items():
        if v is not None:
            row[k] = v
    return row
