
# src/inj/tensor/site_select.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import hashlib
import torch
import numpy as np

@dataclass(frozen=True)
class SitePlan:
    inject: bool
    flat_index: int
    bitpos: int

class SiteSelector:
    """
    Deterministic site/bit planner.
    Given (seed, layer_id, fwd_idx, tensor shape, p_event), produce:
      - inject? (Bernoulli with prob p_event via hash)
      - flat index in [0, numel)
      - bit position in [0, bit_width)
    Uses SHA-256 so results do not depend on RNG state or Python hash seeds.
    """

    def __init__(self, seed: int, p_event: float = 1e-6, bit_width: int = 32):
        assert 0 <= p_event <= 1
        assert bit_width in (16, 32)
        self.seed = int(seed)
        self.p_event = float(p_event)
        self.bit_width = int(bit_width)

    def _u64s(self, key: bytes) -> tuple[int, int, int]:
        h = hashlib.sha256(key).digest()
        u1 = int.from_bytes(h[0:8],  "little", signed=False)
        u2 = int.from_bytes(h[8:16], "little", signed=False)
        u3 = int.from_bytes(h[16:24],"little", signed=False)
        return u1, u2, u3

    def plan(self, shape: torch.Size, layer_id: str, fwd_idx: int) -> SitePlan:
        numel = int(torch.tensor(shape).prod().item()) if len(shape) else 0
        key = f"{self.seed}|{layer_id}|{fwd_idx}|{list(shape)}|{self.p_event}|{self.bit_width}".encode()
        u1, u2, u3 = self._u64s(key)

        # inject decision: u1 / 2^64 < p_event
        inject = u1 < int(self.p_event * (1 << 64))

        flat_index = 0 if numel == 0 else (u2 % numel)
        bitpos = int(u3 % self.bit_width)
        return SitePlan(inject=inject, flat_index=flat_index, bitpos=bitpos)


def flip_fp32_at_index_(t: torch.Tensor, flat_index: int, bitpos: int):
    """
    In-place flip of one fp32 element at a specific flat index/bit.
    Works on CPU/MPS/CUDA by round-tripping exactly one scalar through NumPy.
    """
    if t.dtype != torch.float32 or t.numel() == 0:
        return
    flat = t.view(-1)
    flat_index = int(flat_index % flat.numel())
    val_np = flat[flat_index].detach().cpu().numpy()   # 0-dim float32 array
    u32 = val_np.view(np.uint32)
    u32[...] ^= (1 << int(bitpos))
    flat[flat_index] = torch.from_numpy(val_np.view(np.float32)).to(flat.device)
