
# bench/bert_sst2_baseline.py
import os, json, time, argparse, math, random
from statistics import mean
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device(name: str | None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="prajjwal1/bert-tiny")
    ap.add_argument("--dataset", default="glue")
    ap.add_argument("--subset", default="sst2")
    ap.add_argument("--split", default="validation")  # SST-2 has validation as canonical eval
    ap.add_argument("--samples", type=int, default=500)  # run small, fast slice
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cpu|mps (default: auto)")
    ap.add_argument("--out", default="reports/baseline_bert_tiny_sst2.json")
    args = ap.parse_args()

    set_seed(42)
    device = pick_device(args.device)

    # 1) data
    ds = load_dataset(args.dataset, args.subset, trust_remote_code=True)[args.split]
    if args.samples and args.samples < len(ds):
        ds = ds.select(range(args.samples))

    # 2) tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    def collate(batch):
        texts = [x["sentence"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return {**enc, "labels": labels}

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 3) run + time
    n, correct = 0, 0
    per_sample_latencies_ms = []

    for batch in dl:
        # move to device
        for k in ("input_ids", "attention_mask"):
            batch[k] = batch[k].to(device)
        labels = batch["labels"].to(device)

        t0 = time.perf_counter()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # IMPORTANT for accurate MPS timing:
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0

        logits = out.logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        bsz = labels.size(0)
        n += bsz

        # per-sample latency from batch time
        per_sample = (dt * 1000.0) / bsz
        per_sample_latencies_ms.extend([per_sample] * bsz)

    accuracy = correct / n
    avg_latency_ms = mean(per_sample_latencies_ms)

    # 4) save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = {
        "model": args.model,
        "dataset": f"{args.dataset}/{args.subset}:{args.split}",
        "device": str(device),
        "num_samples": n,
        "batch_size": args.batch_size,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms_per_sample": round(avg_latency_ms, 3),
        "seed": 42,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
