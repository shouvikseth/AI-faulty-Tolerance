# src/inj/tensor/dethook.py
from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch  # <-- must be imported before we reference torch in type hints

# ---------------------- helpers ----------------------

def _bit_pool(bit_width: int, mode: str) -> List[int]:
    mode = (mode or "any").lower()
    if bit_width == 32:
        if mode == "exp":       return list(range(23, 31))   # exponent
        if mode == "sign":      return [31]                  # sign
        if mode in ("signexp","exp+sign","sign+exp"):
            return list(range(23, 32))                       # exponent + sign
    return list(range(bit_width))                            # default: any

@torch.no_grad()
def _flip_scalar_f32_to_bits(val: torch.Tensor, bitpos: int) -> torch.Tensor:
    """
    val: 0-d float32 tensor on any device. Returns flipped scalar (same device).
    """
    import numpy as np
    v = val.detach().cpu().numpy()   # 0-d float32 array
    u = v.view(np.uint32); u[...] ^= (1 << bitpos)
    return torch.from_numpy(v.view(np.float32)).to(val.device)

@torch.no_grad()
def _flip_k_inplace_f32(t: torch.Tensor, indices: List[int], bitpos: int) -> None:
    assert t.dtype == torch.float32, "only float32 supported in this demo"
    flat = t.view(-1)
    for i in indices:
        flat[i] = _flip_scalar_f32_to_bits(flat[i], bitpos)

# ------------------ site selector --------------------

class SiteSelector:
    """Seeded selector that decides whether to inject and which bit to flip."""
    def __init__(self, seed: int, p_event: float, bit_width: int = 32):
        self.seed = seed
        self.rng  = random.Random(seed)
        self.p    = float(p_event)
        self.bit_width = int(bit_width)

    def plan(self, nelem: int) -> Dict:
        inject = (self.rng.random() < self.p)
        flat_index = self.rng.randrange(nelem) if inject and nelem > 0 else None
        bitpos = self.rng.randrange(self.bit_width) if inject else None
        return {"inject": inject, "flat_index": flat_index, "bitpos": bitpos}

# ----------------- deterministic hook ----------------

@dataclass
class DeterministicFlipHook:
    label: str
    selector: SiteSelector
    last_plan: Optional[Dict] = None

    def __call__(self, module, inputs, output):
        if not torch.is_tensor(output):
            self.last_plan = {"inject": False}
            return output

        t = output
        plan = self.selector.plan(t.numel())
        # record whether we injected at all
        self.last_plan = {"inject": plan["inject"], "flat_index": plan["flat_index"]}

        if not plan["inject"] or t.numel() == 0:
            return output

        # env knobs
        try:
            k = max(1, min(int(os.getenv("INJ_K", "1")), t.numel()))
        except Exception:
            k = 1
        bitmode = os.getenv("INJ_BITPOS", "any")

        pool   = _bit_pool(self.selector.bit_width, bitmode)
        rng    = self.selector.rng
        bitpos = rng.choice(pool)

        # choose K unique indices
        if t.numel() >= k:
            indices = rng.sample(range(t.numel()), k)
        else:
            indices = list(range(t.numel()))

        _flip_k_inplace_f32(t, indices, bitpos)

        self.last_plan.update({"bitpos": bitpos, "k": k, "indices": indices})
        return output

# ------------- attach + live handle proxy ------------

def attach_deterministic_hook(module: torch.nn.Module, label: str, selector: SiteSelector):
    hook = DeterministicFlipHook(label=label, selector=selector)
    handle = module.register_forward_hook(hook)

    class HandleProxy:
        def __init__(self, h, hk):
            self._h = h
            self._hook = hk
        def remove(self):
            self._h.remove()
        @property
        def last_plan(self):
            # always return most recent plan recorded by hook
            return self._hook.last_plan

    return HandleProxy(handle, hook)
