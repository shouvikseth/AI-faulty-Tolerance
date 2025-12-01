from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from typing import List, Tuple

import yaml
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.inj.tensor.dethook import attach_deterministic_hook, SiteSelector
from src.utils.log import append_jsonl  # assumes you already have this
# optional: validator at the end
try:
    from src.utils.validate import validate_file
    HAVE_VALIDATOR = True
except Exception:
    HAVE_VALIDATOR = False


# ------------------ device + utils -------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@torch.no_grad()
def top1_and_margin(logits: torch.Tensor):
    top2v, top2i = torch.topk(logits, k=2, dim=1)
    pred = top2i[:, 0]
    margin = (top2v[:, 0] - top2v[:, 1]).detach()
    return pred, margin


# ----------------- model/dataset ---------------------

def build_dataloader(cfg: dict) -> Tuple[DataLoader, str]:
    """
    Supports: MNIST (split: train|test), batch_size.
    """
    kind = (cfg.get("kind") or "mnist").lower()
    split = (cfg.get("split") or "test").lower()
    bs = int(cfg.get("batch_size") or 64)

    if kind == "mnist":
        tfm = T.Compose([T.ToTensor()])
        is_train = (split == "train")
        ds = tv.datasets.MNIST(root="./data", train=is_train, download=True, transform=tfm)
        dl = DataLoader(ds, batch_size=bs, shuffle=False)
        return dl, f"MNIST:{split}"
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

def build_model(cfg: dict, device: str) -> nn.Module:
    """
    Supports: SmallCNN (from src.bench.mnist.SmallCNN)
    """
    kind = cfg.get("kind") or "SmallCNN"
    weights = cfg.get("weights")
    if kind == "SmallCNN":
        from src.bench.mnist import SmallCNN
        m = SmallCNN()
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    m.to(device)
    if weights:
        sd = torch.load(weights, map_location=device)
        m.load_state_dict(sd, strict=True)
    m.eval()
    return m


# ------------- target selection helpers -------------

def select_modules_by_specs(model: nn.Module, specs: List[str]) -> List[Tuple[str, nn.Module]]:
    """
    Supported specs:
      - 'Linear' / 'Conv2d' / 'ReLU'        -> all layers of that type
      - 'Linear[-1]' / 'Conv2d[-1]' / 'ReLU[-1]' -> last of that type
      - 'Linear[i]' / 'Conv2d[i]' / 'ReLU[i]'    -> i-th layer of that type (0-based)
    """
    type_map = {"Linear": nn.Linear, "Conv2d": nn.Conv2d, "ReLU": nn.ReLU}

    # build index lists for each type
    idx = {k: [] for k in type_map}
    for m in model.modules():
        for name, cls in type_map.items():
            if isinstance(m, cls):
                idx[name].append(m)

    out: List[Tuple[str, nn.Module]] = []
    for spec in specs:
        # parse e.g. "Linear", "Linear[-1]", "Conv2d[0]"
        if "[" in spec and spec.endswith("]"):
            base = spec[:spec.index("[")]
            sel  = spec[spec.index("[")+1:-1]
        else:
            base, sel = spec, None

        if base not in idx:
            raise ValueError(f"Unsupported target_layers spec '{spec}' (unknown type '{base}')")

        arr = idx[base]
        if not arr:
            raise RuntimeError(f"No modules of type {base} found in model")

        if sel is None:
            for i, m in enumerate(arr):
                out.append((f"{base}[{i}]", m))
        else:
            i = int(sel)
            if i < 0:
                i = len(arr) + i
            if not (0 <= i < len(arr)):
                raise IndexError(f"Index {sel} out of range for {base} (len={len(arr)}) in spec '{spec}'")
            out.append((f"{base}[{i}]", arr[i]))
    return out

def attach_all_target_hooks(model: nn.Module, targets: List[Tuple[str, nn.Module]],
                            seed: int, p: float, bit_width: int):
    """
    Create one selector shared by all attached hooks so RNG is deterministic
    across modules for a given trial seed.
    """
    selector = SiteSelector(seed=seed, p_event=p, bit_width=bit_width)
    handles = []
    for label, module in targets:
        h = attach_deterministic_hook(module, label=label, selector=selector)
        handles.append(h)
    return handles


# ---------------------- main loop -------------------

def run(plan_path: str, shard: int, shards: int, resume: int):
    with open(plan_path) as f:
        plan = yaml.safe_load(f)

    device = pick_device()
    dataset_cfg = plan.get("dataset", {})
    model_cfg   = plan.get("model", {})
    inj_cfg     = plan.get("inject", {})
    trials_cfg  = plan.get("trials", {})
    log_cfg     = plan.get("logging", {})
    eval_cfg    = plan.get("eval", {})  # optional

    dl, dataset_desc = build_dataloader(dataset_cfg)
    m = build_model(model_cfg, device)

    target_specs = inj_cfg.get("target_layers", ["Linear[-1]"])
    targets = select_modules_by_specs(m, target_specs)
    layer_id = "|".join(t[0] for t in targets)

    p          = float(inj_cfg.get("p", 1e-6))
    bit_width  = int(inj_cfg.get("bit_width", 32))
    trials_max = int(trials_cfg.get("max", 100))
    seed_base  = int(trials_cfg.get("seed_base", 10000))
    log_path   = log_cfg.get("path", "out/runs.jsonl")
    validate   = bool(log_cfg.get("validate", True))

    eval_mode  = (eval_cfg.get("mode") or "label").lower()  # "label" or "margin"
    delta      = float(eval_cfg.get("delta", 0.5))

    # Iterate batches (cap by trials_max)
    trial = 0
    for (x, y) in dl:
        if trial >= trials_max:
            break

        x = x.to(device); y = y.to(device)

        # CLEAN pass (no hooks)
        with torch.no_grad():
            clean_logits = m(x)
        clean_pred, clean_margin = top1_and_margin(clean_logits)

        # INJECTED pass (attach hooks, run once, remove)
        seed = seed_base + trial
        handles = attach_all_target_hooks(m, targets, seed, p, bit_width)
        crashed = False
        inj_pred = None; inj_margin = None; last_plan = {}
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                inj_logits = m(x)
            inj_pred, inj_margin = top1_and_margin(inj_logits)
        except Exception:
            crashed = True
        finally:
            for h in handles:
                try: h.remove()
                except Exception: pass
            # read metadata from the first handle if present
            try:
                lp = handles[0].last_plan
                if lp:
                    last_plan = dict(lp)
            except Exception:
                pass
        t1 = time.perf_counter()
        avg_latency_ms_per_sample = float((t1 - t0) * 1000.0 / max(1, x.size(0)))

        # classify outcome
        if crashed:
            outcome = "CRASH"
            changed = None
            changed_rate = 0.0
            drop_avg = None
        else:
            changed = int((inj_pred != clean_pred).sum().item())
            changed_rate = float(((inj_pred != clean_pred).float().mean().item()))
            if changed > 0:
                outcome = "WRONG"
                drop_avg = None
            elif eval_mode == "margin":
                drop_avg = float((clean_margin - inj_margin).clamp_min(0).mean().item())
                outcome = "DEGRADED" if drop_avg >= delta else "CLEAN"
            else:
                outcome = "CLEAN"
                drop_avg = None

        # log row
        row = {
            "timestamp": now_utc(),
            "trial": trial,
            "outcome": outcome,
            "seed": seed,
            "device": device,
            "model": model_cfg.get("kind", "SmallCNN"),
            "dataset": dataset_desc,
            "layer_id": layer_id,
            "p": p,
            "bit_mode": bit_width,
            "avg_latency_ms_per_sample": avg_latency_ms_per_sample,
            "changed": changed,
            "changed_rate": changed_rate,
            "clean_margin_avg": float(clean_margin.mean().item()),
            "inj_margin_avg": float(inj_margin.mean().item()) if inj_margin is not None else None,
            "margin_drop_avg": drop_avg,
            "plan": plan.get("name", "plan"),
            "shard": shard,
            "shards": shards,
        }
        if last_plan:
            row.update({
                "injected": bool(last_plan.get("inject", True)),
                "bitpos": last_plan.get("bitpos"),
                "flat_index": last_plan.get("flat_index"),
                "k": last_plan.get("k"),
                "indices_count": len(last_plan.get("indices", [])) if last_plan.get("indices") is not None else None,
            })

        append_jsonl(log_path, row)
        trial += 1

    # optional validation summary
    if validate and HAVE_VALIDATOR:
        valid, invalid = validate_file(log_path, schema_path="schemas/run.schema.json")
        print(f"[SUMMARY] {log_path}: valid={valid}, invalid={invalid}")
        print(f"[TOTAL] valid={valid}, invalid={invalid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0)  # placeholder
    args = ap.parse_args()
    run(args.plan, args.shard, args.shards, args.resume)

if __name__ == "__main__":
    main()
