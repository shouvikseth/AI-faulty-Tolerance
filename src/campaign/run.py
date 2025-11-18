# campaign/run.py
import os, json, time, argparse, random
from statistics import mean
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from inj.tensor.flip import TensorBitFlipper, register_bert_layer_hook

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

def pick_device(name: str | None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_sst2_subset(samples: int, split="validation"):
    ds = load_dataset("glue", "sst2", trust_remote_code=True)[split]
    if samples and samples < len(ds):
        ds = ds.select(range(samples))
    return ds

def make_loader(ds, tok, batch_size: int):
    def collate(batch):
        texts = [x["sentence"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return {**enc, "labels": labels}
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

@torch.no_grad()
def predict(model, device, dl) -> Tuple[List[int], float]:
    model.eval()
    per_sample_latency = []
    preds_all: List[int] = []

    for batch in dl:
        for k in ("input_ids", "attention_mask"):
            batch[k] = batch[k].to(device)
        labels = batch["labels"].to(device)

        t0 = time.perf_counter()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        bsz = labels.size(0)
        per_sample_latency.extend([dt * 1000.0 / bsz] * bsz)

        preds = out.logits.argmax(dim=-1).cpu().tolist()
        preds_all.extend(preds)

    return preds_all, mean(per_sample_latency)

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="prajjwal1/bert-tiny")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--layer_idx", type=int, default=1)  # <= safer default for bert-tiny
    ap.add_argument("--p", type=float, default=1e-6)
    ap.add_argument("--bit_mode", default="random")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/bert_tiny_trials.jsonl")
    ap.add_argument("--clean_cache", default="reports/bert_tiny_clean_preds.pt")
    args = ap.parse_args()

    device = pick_device(args.device)
    set_seed(42)

    ds = load_sst2_subset(samples=args.samples, split="validation")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    dl = make_loader(ds, tok, args.batch_size)

    # figure out valid layer range and clamp
    try:
        n_layers = len(model.bert.encoder.layer)
    except Exception as e:
        raise RuntimeError("Model does not look like a BERT encoder with 'encoder.layer'.") from e

    layer_idx = max(0, min(args.layer_idx, n_layers - 1))
    if layer_idx != args.layer_idx:
        print(f"[info] Requested layer_idx={args.layer_idx} but model has {n_layers} layers; using {layer_idx}.")

    # baseline clean preds (cache)
    if os.path.exists(args.clean_cache):
        cache = torch.load(args.clean_cache)
        clean_preds = cache["preds"]
        assert cache["samples"] == args.samples, "Cache was built for a different sample size."
    else:
        clean_preds, clean_lat = predict(model, device, dl)
        os.makedirs(os.path.dirname(args.clean_cache), exist_ok=True)
        torch.save({"preds": clean_preds, "samples": args.samples}, args.clean_cache)
        print(f"Saved clean preds (avg latency {clean_lat:.3f} ms/sample) → {args.clean_cache}")

    # trials
    rows = []
    base_seed = 1000
    for t in range(args.trials):
        trial_seed = base_seed + t
        set_seed(trial_seed)

        flipper = TensorBitFlipper(p=args.p, bit_mode=args.bit_mode, generator=torch.Generator().manual_seed(trial_seed))
        handle = register_bert_layer_hook(model, layer_idx, flipper)

        preds, avg_lat = predict(model, device, dl)
        handle.remove()

        changed = sum(int(a != b) for a, b in zip(preds, clean_preds))
        changed_rate = changed / len(clean_preds)
        outcome = "WRONG" if changed > 0 else "CLEAN"

        rows.append({
            "trial": t,
            "seed": trial_seed,
            "device": str(device),
            "layer_idx": layer_idx,
            "p": args.p,
            "bit_mode": args.bit_mode,
            "elements_seen": flipper.total_elements_seen,
            "elements_flipped": flipper.total_elements_flipped,
            "avg_latency_ms_per_sample": round(avg_lat, 3),
            "changed_preds": changed,
            "changed_rate": round(changed_rate, 6),
            "outcome": outcome,
        })

        if (t + 1) % 10 == 0 or (t + 1) == args.trials:
            save_jsonl(args.out, rows)
            rows.clear()
            print(f"[{t+1}/{args.trials}] wrote logs to {args.out}")

    print("Done.")

if __name__ == "__main__":
    main()
