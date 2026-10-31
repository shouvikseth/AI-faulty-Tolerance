# inj/tensor/flip.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random

import torch
import numpy as np


@dataclass
class FlipStats:
    forwards_seen: int = 0
    injections: int = 0
    elements_seen: int = 0
    elements_flipped: int = 0


class TensorBitFlipper:
    """
    Safe FP32 bit-flip injector for CPU/MPS/CUDA tensors.

    Matches the API expected by src/campaign/run.py:
      TensorBitFlipper(p=..., bit_mode=..., generator=...)
      and exposes .total_elements_seen / .total_elements_flipped
    """

    def __init__(
        self,
        p: float = 1e-6,
        bit_mode: str | int = "random",
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.bit_mode = bit_mode
        self.gen = generator
        # Python RNG (simple, good enough); if you want you can
        # later integrate torch.Generator for stricter determinism.
        self.rng = random.Random(seed) if seed is not None else random

        self.stats = FlipStats()
        self.total_elements_seen = 0
        self.total_elements_flipped = 0

    def _choose_bit(self, nbits: int = 32) -> int:
        if self.bit_mode == "random":
            return self.rng.randrange(nbits)
        return int(self.bit_mode)

    def _flip_one_fp32_scalar_inplace(self, t: torch.Tensor, bitpos: int):
        flat = t.view(-1)
        if flat.numel() == 0:
            return

        idx = self.rng.randrange(flat.numel())

        # move ONE scalar to CPU as numpy, reinterpret bits, flip, and write back
        val_np = flat[idx].detach().cpu().numpy()           # 0-dim float32 array
        u32 = val_np.view(np.uint32)                        # reinterpret bits
        u32[...] ^= (1 << bitpos)                           # flip the bit
        flipped = torch.from_numpy(val_np.view(np.float32)) # back to float32 (CPU)
        flat[idx] = flipped.to(flat.device)                 # back to original device

        # stats (both detailed + convenience counters)
        self.stats.elements_seen += t.numel()
        self.stats.elements_flipped += 1
        self.total_elements_seen += t.numel()
        self.total_elements_flipped += 1

    def flip(self, x: torch.Tensor) -> torch.Tensor:
        self.stats.forwards_seen += 1
        self.total_elements_seen += x.numel()

        # only handle float32 for now
        if x.dtype != torch.float32:
            self.stats.elements_seen += x.numel()
            return x

        # inject or not on this forward
        if self.rng.random() >= self.p:
            self.stats.elements_seen += x.numel()
            return x

        # perform single-scalar flip
        y = x.clone()
        self._flip_one_fp32_scalar_inplace(y, self._choose_bit())
        self.stats.injections += 1
        return y


def register_module_output_hook(module: torch.nn.Module, flipper: TensorBitFlipper):
    """
    Generic hook: apply flipper.flip() to any module whose forward returns a Tensor.
    """
    def _hook(_m, _inputs, output):
        if isinstance(output, torch.Tensor):
            return flipper.flip(output)
        return output

    return module.register_forward_hook(_hook)


def register_bert_layer_hook(
    model: torch.nn.Module,
    layer_idx: int,
    flipper: TensorBitFlipper,
):
    """
    What src/campaign/run.py imports.

    Tries to hook model.bert.encoder.layer[layer_idx].output (BERT-style).
    If that structure isn't found, falls back to hooking the whole model.
    """
    target_module = None

    # Standard BERT layout
    if (
        hasattr(model, "bert")
        and hasattr(model.bert, "encoder")
        and hasattr(model.bert.encoder, "layer")
    ):
        layers = model.bert.encoder.layer
        if 0 <= layer_idx < len(layers):
            target_module = layers[layer_idx].output

    # Fallback: hook the full model output
    if target_module is None:
        target_module = model

    def _hook(_m, _inputs, output):
        if isinstance(output, torch.Tensor):
            return flipper.flip(output)
        return output

    return target_module.register_forward_hook(_hook)
