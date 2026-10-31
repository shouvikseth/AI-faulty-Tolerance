from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch  # must be imported before referencing in type hints


# ---------------------- helpers ----------------------

def _bit_pool(bit_width: int, mode: str) -> List[int]:
    """
    Choose a pool of bit positions based on the requested mode.
    Modes: any | exp | sign | signexp (exp+sign).
    Only float32 bit layout is specialized here.
    """
    mode = (mode or "any").lower()
    if bit_width == 32:
        if mode == "exp":       return list(range(23, 31))   # exponent bits
        if mode == "sign":      return [31]                  # sign bit
        if mode in ("signexp","exp+sign","sign+exp"):
            return list(range(23, 32))                       # exponent + sign
    return list(range(bit_width))                            # default: any

@torch.no_grad()
def _flip_scalar_f32_to_bits(val: torch.Tensor, bitpos: int) -> torch.Tensor:
    """
    Flip a single bit of a single float32 scalar (0-dim tensor).
    Returns a new scalar tensor on the same device.
    """
    import numpy as np
    v = val.detach().cpu().numpy()       # 0-d float32 array
    u = v.view(np.uint32); u[...] ^= (1 << bitpos)
    return torch.from_numpy(v.view(np.float32)).to(val.device)

@torch.no_grad()
def _flip_k_inplace_f32(t: torch.Tensor, indices: List[int], bitpos: int) -> None:
    """
    Flip the same bit position for K randomly chosen elements in t (float32).
    """
    assert t.dtype == torch.float32, "only float32 supported in this demo"
    flat = t.view(-1)
    for i in indices:
        flat[i] = _flip_scalar_f32_to_bits(flat[i], bitpos)


# ------------------ site selector --------------------

class SiteSelector:
    """
    Seeded selector that decides whether to inject and which bit to flip.
    """
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
        # Only handle Tensor outputs
        if not torch.is_tensor(output):
            self.last_plan = {"inject": False}
            return output

        t = output
        plan = self.selector.plan(t.numel())
        # record base plan
        self.last_plan = {"inject": plan["inject"], "flat_index": plan["flat_index"]}

        if not plan["inject"] or t.numel() == 0:
            return output

        # env knobs
        bitmode = os.getenv("INJ_BITPOS", "any")
        try:
            k = max(1, min(int(os.getenv("INJ_K", "1")), t.numel()))
        except Exception:
            k = 1

        pool   = _bit_pool(self.selector.bit_width, bitmode)
        rng    = self.selector.rng
        bitpos = rng.choice(pool)

        # choose K unique indices
        if t.numel() >= k:
            indices = rng.sample(range(t.numel()), k)
        else:
            indices = list(range(t.numel()))

        _flip_k_inplace_f32(t, indices, bitpos)

        # stash rich metadata for the orchestrator to log
        self.last_plan.update({"bitpos": bitpos, "k": k, "indices": indices})
        return output


# ------------- attach + live handle proxy ------------

def attach_deterministic_hook(module: torch.nn.Module, label: str, selector: SiteSelector):
    """
    Attach the deterministic flip hook and return a proxy handle whose
    `.last_plan` property always exposes the most recent plan.
    """
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
            return self._hook.last_plan

    return HandleProxy(handle, hook)
