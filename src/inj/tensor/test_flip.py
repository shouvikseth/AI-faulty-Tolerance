# src/inj/tensor/test_flip.py
import torch
from src.inj.tensor.flip import TensorBitFlipper

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("device:", device)

    # Use a fixed tensor so we can inspect diffs reliably
    x = torch.randn(64, 64, dtype=torch.float32, device=device)

    # Force an obvious flip for the demo: flip the SIGN bit (31) so a value changes from + to -
    flipper = TensorBitFlipper(p_event=1.0, bit_mode=31, seed=123)

    y = flipper.flip(x)

    # 1) Exact equality (should be False if any element changed)
    print("torch.equal (expect False):", torch.equal(x, y))

    # 2) Count how many elements differ (exact, no tolerance)
    num_diff = (x != y).sum().item()
    print("num elements different (expect >= 1):", int(num_diff))

    # 3) Also show a numeric diff scale
    max_abs_diff = (x - y).abs().max().item()
    print("max |x - y|:", max_abs_diff)

    # Sanity: with p=0, tensor must be IDENTICAL
    flipper0 = TensorBitFlipper(p_event=0.0, seed=123)
    z = flipper0.flip(x)
    print("p=0 → torch.equal (expect True):", torch.equal(x, z))

if __name__ == "__main__":
    main()
