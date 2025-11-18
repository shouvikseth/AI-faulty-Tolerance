# src/inj/tensor/demo_det_mnist.py
import torch, torchvision as tv, torchvision.transforms as T
from torch.utils.data import DataLoader
from src.bench.mnist import SmallCNN
from src.inj.tensor.site_select import SiteSelector
from src.inj.tensor.dethook import attach_deterministic_hook

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    m = SmallCNN().to(device).eval()

    tfm = T.Compose([T.ToTensor()])
    test = tv.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    dl = DataLoader(test, batch_size=64, shuffle=False)
    x, _ = next(iter(dl))
    x = x.to(device)

    # Same seed, same layer, same forward index → same flip → outputs equal
    sel = SiteSelector(seed=2025, p_event=1.0, bit_width=32)  # force inject to visualize determinism

    hook1 = attach_deterministic_hook(m.net[-1], layer_id="Linear[-1]", selector=sel)
    with torch.no_grad():
        out1 = m(x)
        if device.type == "mps": torch.mps.synchronize()
    hook1.remove()

    # Re-attach a NEW hook with the same seed; forward index resets to 1
    hook2 = attach_deterministic_hook(m.net[-1], layer_id="Linear[-1]", selector=sel)
    with torch.no_grad():
        out2 = m(x)
        if device.type == "mps": torch.mps.synchronize()
    hook2.remove()

    same_runs = torch.equal(out1, out2)
    print("same seed, same fwd index → identical outputs (expect True):", same_runs)

    # Change the seed → a different element/bit gets flipped → outputs differ
    sel_diff = SiteSelector(seed=999, p_event=1.0, bit_width=32)
    hook3 = attach_deterministic_hook(m.net[-1], layer_id="Linear[-1]", selector=sel_diff)
    with torch.no_grad():
        out3 = m(x)
        if device.type == "mps": torch.mps.synchronize()
    hook3.remove()

    diff = (out2 != out3).sum().item()
    print("different seed → different output (expect > 0 diffs):", diff)

if __name__ == "__main__":
    main()
