from __future__ import annotations
import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_loaders(bs_train=128, bs_test=256):
    tfm_train = T.Compose([T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor()])
    tfm_test  = T.Compose([T.Resize(224), T.ToTensor()])
    train = tv.datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm_train)
    test  = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)
    return DataLoader(train, batch_size=bs_train, shuffle=True), DataLoader(test, batch_size=bs_test)

class Head(nn.Module):
    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.fc(x)

def get_model():
    m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)  # ImageNet weights
    m.fc = Head(512, 10)
    return m

@torch.no_grad()
def evaluate(m, dl, device):
    m.eval(); tot=0; ok=0
    for x,y in dl:
        x,y=x.to(device),y.to(device)
        pred = m(x).argmax(1)
        ok += (pred==y).sum().item(); tot += y.numel()
    acc = ok/tot
    print({"acc": acc})
    return acc

def train(m, dl, device, epochs=1, lr=3e-4):
    m.train(); opt=optim.AdamW(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        for x,y in dl:
            x,y=x.to(device),y.to(device)
            logits=m(x); loss=lossf(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save", default="out/cifar_resnet18.pt")
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends,'mps') and torch.backends.mps.is_available()) else "cpu")
    tr,te = get_loaders()
    m = get_model().to(device)
    if args.epochs>0: train(m, tr, device, epochs=args.epochs)
    acc = evaluate(m, te, device)
    torch.save(m.state_dict(), args.save)
    print(f"[saved] {args.save}")
if __name__=="__main__": main()
