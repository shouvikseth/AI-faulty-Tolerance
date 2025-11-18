
# src/ci/drift_gate.py
import argparse, os
import torch, torch.nn as nn, torch.optim as optim
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader

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
            nn.Linear(128, 10),
        )
    def forward(self, x): return self.net(x)

def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def get_loaders():
    tfm = T.Compose([T.ToTensor()])
    train = tv.datasets.MNIST(root="./data", train=True,  download=True, transform=tfm)
    test  = tv.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=128, shuffle=True), DataLoader(test, batch_size=256, shuffle=False)

def train_one_epoch(m, dl, device):
    m.train(); opt = optim.Adam(m.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        loss = lossf(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def evaluate(m, dl, device) -> float:
    m.eval(); tot=0; correct=0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        pred = m(x).argmax(1)
        correct += (pred==y).sum().item(); tot += y.numel()
    return correct/tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="out/mnist_smallcnn.pt")
    ap.add_argument("--min_acc", type=float, default=0.97)       # pass if acc >= min_acc
    ap.add_argument("--train_if_missing", type=int, default=1)   # if no weights, train 1 epoch
    args = ap.parse_args()

    device = pick_device()
    train_dl, test_dl = get_loaders()
    m = SmallCNN().to(device)

    if os.path.exists(args.weights):
        m.load_state_dict(torch.load(args.weights, map_location=device))
    elif args.train_if_missing:
        torch.manual_seed(42)
        train_one_epoch(m, train_dl, device)
        os.makedirs(os.path.dirname(args.weights), exist_ok=True)
        torch.save(m.state_dict(), args.weights)
    else:
        raise FileNotFoundError(f"weights not found: {args.weights}")

    acc = evaluate(m, test_dl, device)
    print(f"[drift-gate] accuracy={acc:.4f} threshold={args.min_acc:.4f} -> {'PASS' if acc>=args.min_acc else 'FAIL'}")
    if acc < args.min_acc:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
