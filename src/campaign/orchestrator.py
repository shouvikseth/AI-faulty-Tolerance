# src/campaign/orchestrator.py
import argparse, os, time, json
import yaml
import torch, torch.nn as nn, torch.optim as optim
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.bench.mnist import SmallCNN
from src.inj.tensor.site_select import SiteSelector
from src.inj.tensor.dethook import attach_deterministic_hook
from src.utils.log import make_row, append_jsonl

# -------------------- helpers --------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_mnist_loader(split: str, batch_size: int):
    tfm = T.Compose([T.ToTensor()])
    is_train = (split.lower() == "train")
    ds = tv.datasets.MNIST(root="./data", train=is_train, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def ensure_weights(m: nn.Module, device, weights_path: str):
    """Load weights if present; otherwise train 1 epoch to produce a baseline."""
    if os.path.exists(weights_path):
        m.load_state_dict(torch.load(weights_path, map_location=device))
        return
    train = get_mnist_loader("train", 128)
    lossf = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=1e-3)
    m.train()
    torch.manual_seed(42)
    for x, y in train:
        x, y = x.to(device), y.to(device)
        loss = lossf(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(m.state_dict(), weights_path)

def last_linear(module: nn.Module):
    last = None
    for mod in module.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    return last

def select_modules_by_specs(model: nn.Module, specs):
    """
    Supported specs:
      - 'Linear[-1]'  -> last Linear
      - 'Linear'      -> all Linear layers
    """
    out = []
    for spec in specs:
        if spec == "Linear[-1]":
            mod = last_linear(model)
            if mod is None:
                raise RuntimeError("No nn.Linear found for spec 'Linear[-1]'")
            out.append(("Linear[-1]", mod))
        elif spec == "Linear":
            for i, mod in enumerate(m for m in model.modules() if isinstance(m, nn.Linear)):
                out.append((f"Linear[{i}]", mod))
        else:
            raise ValueError(f"Unsupported target_layers spec '{spec}'")
    return out


def select_modules_by_specs(model: nn.Module, specs):
    """
    Supported specs:
      - 'Linear' / 'Conv2d' / 'ReLU'        -> all layers of that type
      - 'Linear[-1]' / 'Conv2d[-1]' / 'ReLU[-1]' -> last layer of that type
      - 'Linear[i]' / 'Conv2d[i]' / 'ReLU[i]'    -> i-th layer of that type (0-based)
    """
    type_map = {"Linear": nn.Linear, "Conv2d": nn.Conv2d, "ReLU": nn.ReLU}

    # build index lists for each type
    idx = {k: [] for k in type_map}
    for m in model.modules():
        for name, cls in type_map.items():
            if isinstance(m, cls):
                idx[name].append(m)

    out = []
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
            # all
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

@torch.no_grad()
def clean_preds(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    y = model(x)
    return y.argmax(1)

def has_inf_nan(t: torch.Tensor) -> bool:
    return not torch.isfinite(t).all().item()

def read_completed_trials(log_path: str, plan_name: str, shard: int):
    done = set()
    if not os.path.exists(log_path):
        return done
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("plan") == plan_name and int(obj.get("shard", -1)) == int(shard):
                try:
                    done.add(int(obj["trial"]))
                except Exception:
                    pass
    return done

# -------------------- main orchestrator --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to YAML plan")
    ap.add_argument("--shard", type=int, default=0, help="This shard index")
    ap.add_argument("--shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--resume", type=int, default=1, help="1=skip already-logged trials for this plan+shard")
    args = ap.parse_args()

    with open(args.plan) as f:
        cfg = yaml.safe_load(f)

    plan_name = cfg.get("name", os.path.splitext(os.path.basename(args.plan))[0])
    ds_cfg   = cfg["dataset"]
    mdl_cfg  = cfg["model"]
    inj_cfg  = cfg["inject"]
    tri_cfg  = cfg["trials"]
    log_cfg  = cfg["logging"]
    resume   = bool(cfg.get("resume", True) and args.resume)

    # device + model
    device = pick_device()
    if mdl_cfg["kind"] != "SmallCNN":
        raise ValueError("This demo orchestrator only supports model.kind=SmallCNN for now.")
    model = SmallCNN().to(device).eval()
    ensure_weights(model, device, mdl_cfg.get("weights", "out/mnist_smallcnn.pt"))

    # dataset
    if ds_cfg["kind"].lower() != "mnist":
        raise ValueError("This demo orchestrator only supports dataset.kind=mnist for now.")
    dl = get_mnist_loader(ds_cfg.get("split", "test"), ds_cfg.get("batch_size", 64))

    # injection config
    enabled   = bool(inj_cfg.get("enabled", True))
    p_event   = float(inj_cfg.get("p", 1e-6))
    bit_width = int(inj_cfg.get("bit_width", 32))
    targets   = inj_cfg.get("target_layers", ["Linear[-1]"])
    targets_labeled = select_modules_by_specs(model, targets)
    if not enabled:
        p_event = 0.0

    # trials + logging
    max_trials = int(tri_cfg.get("max", 100))
    seed_base  = int(tri_cfg.get("seed_base", 2025))
    log_path   = log_cfg["path"]
    do_validate = bool(log_cfg.get("validate", True))

    # resume bookkeeping
    completed = read_completed_trials(log_path, plan_name, args.shard) if resume else set()

    processed = 0
    for global_idx, (x, y) in enumerate(dl):
        if (global_idx % max(args.shards, 1)) != args.shard:
            continue
        if processed >= max_trials:
            break
        if resume and (global_idx in completed):
            continue

        x = x.to(device)
        trial_seed = seed_base + global_idx
        outcome = "CLEAN"
        err_text = None
        changed = 0
        avg_ms = None

        # clean reference prediction
        try:
            clean = clean_preds(model, x)
        except Exception as e:
            outcome = "CRASH"
            err_text = f"clean pass error: {e}"
            row = make_row(
                trial=global_idx, outcome=outcome, seed=trial_seed, device=str(device),
                model=mdl_cfg["kind"], dataset=f"MNIST:{ds_cfg.get('split','test')}",
                layer_id="|".join(t for t, _ in targets_labeled),
                p=p_event, bit_mode=bit_width, error=err_text
            )
            row.update({"plan": plan_name, "shard": args.shard, "shards": args.shards})
            append_jsonl(log_path, row)
            processed += 1
            continue

        # attach deterministic hooks for THIS TRIAL
        handles = []
        selector = SiteSelector(seed=trial_seed, p_event=p_event, bit_width=bit_width)
        try:
            for label, module in targets_labeled:
                handles.append(attach_deterministic_hook(module, label, selector))

            # injected forward (timed)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(x)
                if device.type == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
            dt = time.perf_counter() - t0
            avg_ms = (dt * 1000.0) / max(x.size(0), 1)

            if has_inf_nan(logits):
                outcome = "DETECTED"
            else:
                pred_inj = logits.argmax(1)
                changed = int((pred_inj != clean).sum().item())
                outcome = "WRONG" if changed > 0 else "CLEAN"

        except Exception as e:
            outcome = "CRASH"
            err_text = str(e)
        finally:
            # collect flip metadata from the FIRST hook (all share seed/selector)
            flip_meta = {}
            if handles and getattr(handles[0], "last_plan", None):
                lp = handles[0].last_plan
                if lp:
                    flip_meta = {"bitpos": lp.get("bitpos"), "flat_index": lp.get("flat_index")}
            for h in handles:
                h.remove()

        # log row
        row = make_row(
            trial=global_idx, outcome=outcome, seed=trial_seed, device=str(device),
            model=mdl_cfg["kind"], dataset=f"MNIST:{ds_cfg.get('split','test')}",
            layer_id="|".join(t for t, _ in targets_labeled),
            p=p_event, bit_mode=bit_width,
            avg_latency_ms_per_sample=(None if avg_ms is None else round(avg_ms, 4)),
            changed=changed,
            changed_rate=(0.0 if x.size(0) == 0 else round(changed / x.size(0), 6)),
            error=err_text
        )
        row.update(flip_meta)
        row.update({"plan": plan_name, "shard": args.shard, "shards": args.shards})
        append_jsonl(log_path, row)
        processed += 1

    # optional: validate the resulting log
    if do_validate and os.path.exists(log_path):
        try:
            from src.utils.validate import main as validate_main
            import sys
            sys.argv = ["validate", "--in", log_path]
            validate_main()
        except Exception as e:
            print("[warn] validation step failed:", e)

if __name__ == "__main__":
    main()
