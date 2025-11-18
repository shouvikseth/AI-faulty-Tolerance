import hashlib
import torch

def _sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]

@torch.no_grad()
def checksum_indices(indices_1d: torch.Tensor) -> str:
    """
    SHA1 (16-hex) of a 1D int tensor (e.g., top-1 class indices).
    """
    a = indices_1d.detach().to(dtype=torch.int64, device="cpu").contiguous().numpy()
    return _sha1_of_bytes(a.tobytes())

@torch.no_grad()
def checksum_topk(indices_2d: torch.Tensor) -> str:
    """
    SHA1 (16-hex) of a 2D int tensor (N x k) of top-k indices.
    Order matters (per-row ranks then row order).
    """
    a = indices_2d.detach().to(dtype=torch.int64, device="cpu").contiguous().numpy()
    return _sha1_of_bytes(a.tobytes())
