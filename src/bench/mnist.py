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
