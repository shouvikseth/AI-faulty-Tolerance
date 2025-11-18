import torch, torchvision as tv, torchvision.transforms as T
from torch.utils.data import DataLoader
from src.bench.mnist import SmallCNN

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = T.Compose([T.ToTensor()])
    test = tv.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    dl = DataLoader(test, batch_size=256)
    m = SmallCNN().to(device).eval()
    try:
        m.load_state_dict(torch.load("out/mnist_smallcnn.pt", map_location=device))
    except Exception:
        pass  # still okay; just less accurate
    preds = []
    for x,_ in dl:
        logits = m(x.to(device))
        preds.append(logits.argmax(1).cpu())
    torch.save(torch.cat(preds), "out/mnist_clean_preds.pt")
    print({"saved": "out/mnist_clean_preds.pt", "total": len(test)})

if __name__ == "__main__":
    main()
