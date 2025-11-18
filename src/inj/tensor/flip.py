# src/inj/tensor/flip.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import random

@dataclass
class FlipStats:
    forwards_seen: int = 0
    injections: int = 0
    elements_seen: int = 0
    elements_flipped: int = 0

class TensorBitFlipper:
    """
    Safe FP32 bit-flip injector for CPU/MPS/CUDA tensors.

    Behavior:
      • On each call to .flip(x): with probability p_event, flips 1 scalar's chosen bit (default random 0..31).
      • No-op for non-float32 tensors (bf16/fp16 can be added later).
      • Tracks stats for logging.

    Notes:
      • Uses a tiny CPU round-trip for exactly ONE scalar to reinterpret bits (float32 <-> uint32).
      • Keeps your original device intact (writes the flipped scalar back to x's device).
    """

    def __init__(self, p_event: float = 1e-6, bit_mode: str | int = "random", seed: Optional[int] = None):
        assert 0.0 <= p_event <= 1.0
        self.p_event = p_event
        self.bit_mode = bit_mode
        self.rng = random.Random(seed) if seed is not None else random
        self.stats = FlipStats()

    def _choose_bit(self) -> int:
        if self.bit_mode == "random":
            return self.rng.randrange(32)  # 0..31 for fp32
        return int(self.bit_mode)

    def _flip_one_fp32_scalar_inplace(self, t: torch.Tensor, bitpos: int):
        # pick a random element in flattened view
        flat = t.view(-1)
        if flat.numel() == 0:
            return
        idx = self.rng.randrange(flat.numel())

        # move exactly ONE scalar to CPU as numpy, reinterpret, flip, write back
        val_np = flat[idx].detach().cpu().numpy()           # 0-dim float32 array
        u32 = val_np.view(np.uint32)                        # reinterpret
        u32[...] ^= (1 << bitpos)                           # flip that bit
        flipped = torch.from_numpy(val_np.view(np.float32)) # back to float32 tensor (CPU)
        flat[idx] = flipped.to(flat.device)                 # write back on original device

        # stats
        self.stats.elements_seen += t.numel()
        self.stats.elements_flipped += 1

    def flip(self, x: torch.Tensor) -> torch.Tensor:
        self.stats.forwards_seen += 1

        # only handle float32 for now—others are no-ops by design
        if x.dtype != torch.float32:
            self.stats.elements_seen += x.numel()
            return x

        # injection decision per-forward
        if self.rng.random() >= self.p_event:
            self.stats.elements_seen += x.numel()
            return x

        # perform single-scalar flip
        y = x.clone()
        self._flip_one_fp32_scalar_inplace(y, self._choose_bit())
        self.stats.injections += 1
        return y


def register_module_output_hook(module: torch.nn.Module, flipper: TensorBitFlipper):
    """
    Attaches a forward hook to any nn.Module that returns a Tensor output.
    Returns the handle so the caller can .remove() it later.
    """
    def _hook(_m, _inputs, output):
        if isinstance(output, torch.Tensor):
            return flipper.flip(output)
        # For tuple/dict outputs, you could extend this later; for now, pass through.
        return output

    return module.register_forward_hook(_hook)
