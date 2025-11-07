# bootstrap.py — generates a minimal working starter in the current folder
import os, json, textwrap, pathlib

BASE = pathlib.Path(".").resolve()

FILES = {
    ".gitignore": """
__pycache__/
*.pyc
.venv/
.env/
data/
out/
.cache/
""",
    "requirements.txt": """
numpy
tqdm
pyyaml
jsonschema
torchvision  # will install with torch separately
""",
    "README.md": """
# Fault-Tolerance Quickstart (from scratch)

## Create & activate a virtual env
# Windows (PowerShell)
python -m venv .venv
.venv\\Scripts\\Activate.ps1
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

## Install deps
pip install -r requirements.txt
# CPU-only PyTorch (works everywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (CUDA 12.x GPU) use: --index-url https://download.pytorch.org/whl/cu121

## Train a tiny MNIST baseline
python -m src.bench.mnist --train 1 --eval 1

## Run a fault-injection sweep (tensor flips)
python -m src.campaign.run --p 1e-6 --trials 200 --jsonl out/runs.jsonl

## Summarize
python -m src.analysis.summarize out/runs.jsonl
""",
    "src/__init__.py": "",
    "src/inj/__init__.py": "",
    "src/inj/tensor/__init__.py": "",
    "src/campaign/__init__.py": "",
    "src/bench/__init__.py": "",
    "src/analysis/__init__.py": "",
    "src/inj/tensor/hook.py": r'''
import torch, random, re

def _flip_one_bit_inplace_float32(t: torch.Tensor, bitpos: int):
    """Flip one bit at a random scalar of a float32 tensor (in-place)."""
    assert t.dtype == torch.float32
    with torch.no_grad():
        flat = t.view(-1)
        if flat.numel() == 0:
            return
        i = random.randrange(flat.numel())
        v = flat[i].view(torch.int32)   # reinterpret
        v ^= (1 << bitpos)
        flat[i] = v.view(torch.float32)

def make_tensor_flip_hook(layer_name_regex: str, p: float = 1e-6, bit_positions=(0,7,15,31), seed: int = 1337):
    pat = re.compile(layer_name_regex)
    rng = random.Random(seed)
    def hook(module, inputs, output):
        name = module._get_name()
        if not pat.search(name):
            return output
        if rng.random() < p:
            bitpos = rng.choice(bit_positions)
            tgt = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tgt, torch.Tensor) and tgt.dtype == torch.float32:
                _flip_one_bit_inplace_float32(tgt, bitpos)
                setattr(module, "_last_injected_bitpos", bitpos)
            else:
                setattr(module, "_last_injected_bitpos", None)
        else:
            setattr(module, "_last_injected_bitpos", None)
        return output
    return hook
''',
    "src/campaign/log.py": r'''
import json, time, os

def write_run(jsonl_path, **fields):
    fields.setdefault("ts", time.time())
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fields) + "\n")
''',
    "src/bench/mnist.py": r'''
import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_data():
    tfm = T.Compose([T.ToTensor()])
    train = tv.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test = tv.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=128, shuffle=True), DataLoader(test, batch_size=256)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

def train_one_epoch(m, dl, device):
    m.train()
    opt = optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        logits = m(x); loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def evaluate(m, dl, device):
    m.eval()
    total = correct = 0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        logits = m(x); pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    print({"acc": correct/total})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--eval", type=int, default=1)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, test_dl = get_data()
    m = SmallCNN().to(device)
    if args.train > 0:
        for _ in range(args.train): train_one_epoch(m, train_dl, device)
    if args.eval: evaluate(m, test_dl, device)
    torch.save(m.state_dict(), "out/mnist_smallcnn.pt")

if __name__ == "__main__": main()
''',
    "src/campaign/run.py": r'''
import argparse, torch
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from src.bench.mnist import SmallCNN
from src.inj.tensor.hook import make_tensor_flip_hook
from src.campaign.log import write_run

def get_test_loader():
    tfm = T.Compose([T.ToTensor()])
    test = tv.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(test, batch_size=256)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=1e-6)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--jsonl", type=str, default="out/runs.jsonl")
    ap.add_argument("--weights", type=str, default="out/mnist_smallcnn.pt")
    ap.add_argument("--layers_regex", type=str, default="Conv2d|ReLU|Linear")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SmallCNN().to(device)
    try:
        m.load_state_dict(torch.load(args.weights, map_location=device))
    except Exception:
        pass
    m.eval()

    hook_fn = make_tensor_flip_hook(args.layers_regex, p=args.p, seed=args.seed)
    removers = []
    for mod in m.modules():
        if mod is m: continue
        removers.append(mod.register_forward_hook(hook_fn))

    dl = get_test_loader()
    for i,(x,y) in enumerate(dl):
        if i >= args.trials: break
        x,y = x.to(device), y.to(device)
        outcome, bitpos, layer = "OK", None, None
        try:
            logits = m(x); pred = logits.argmax(1)
            outcome = "SDC" if (pred != y).any().item() else "OK"
        except Exception:
            outcome = "CRASH"
        for mod in m.modules():
            if hasattr(mod, "_last_injected_bitpos") and getattr(mod, "_last_injected_bitpos") is not None:
                bitpos = getattr(mod, "_last_injected_bitpos"); layer = mod._get_name()
        write_run(args.jsonl, trial=i, p=args.p, outcome=outcome, seed=args.seed, bitpos=bitpos, layer=layer)

    for r in removers: r.remove()
    print({"batches_run": min(args.trials, len(dl)), "log": args.jsonl})

if __name__ == "__main__": main()
''',
    "src/analysis/summarize.py": r'''
import sys, json
from collections import Counter

def main(path):
    c = Counter(); total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line); c[rec.get("outcome","UNK")] += 1; total += 1
    sdc_rate = (c["SDC"]/total) if total else 0.0
    print({"total": total, "counts": dict(c), "sdc_rate": sdc_rate})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.summarize out/runs.jsonl"); raise SystemExit(1)
    main(sys.argv[1])
''',
}

# Create directories and write files
for rel, content in FILES.items():
    path = BASE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).lstrip("\n"))

print("Starter generated. Next steps:")
print("1) Create venv")
print("2) pip install -r requirements.txt")
print("3) pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
print("4) python -m src.bench.mnist --train 1 --eval 1")
print("5) python -m src.campaign.run --p 1e-6 --trials 200 --jsonl out/runs.jsonl")
print("6) python -m src.analysis.summarize out/runs.jsonl")
