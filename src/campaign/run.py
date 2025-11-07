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
