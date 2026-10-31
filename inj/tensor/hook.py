import torch, random, re

def _flip_one_bit_inplace_float32(t: torch.Tensor, bitpos: int):
    """Flip one bit at a random scalar of a float32 tensor (in-place)."""
    assert t.dtype == torch.float32
    with torch.no_grad():
        flat = t.view(-1)
        if flat.numel() == 0:
            return
        i = random.randrange(flat.numel())
        v = flat[i].view(torch.int32)   # reinterpret
        v ^= (1 << bitpos)
        flat[i] = v.view(torch.float32)

def make_tensor_flip_hook(layer_name_regex: str, p: float = 1e-6, bit_positions=(0,7,15,31), seed: int = 1337):
    pat = re.compile(layer_name_regex)
    rng = random.Random(seed)
    def hook(module, inputs, output):
        name = module._get_name()
        if not pat.search(name):
            return output
        if rng.random() < p:
            bitpos = rng.choice(bit_positions)
            tgt = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tgt, torch.Tensor) and tgt.dtype == torch.float32:
                _flip_one_bit_inplace_float32(tgt, bitpos)
                setattr(module, "_last_injected_bitpos", bitpos)
            else:
                setattr(module, "_last_injected_bitpos", None)
        else:
            setattr(module, "_last_injected_bitpos", None)
        return output
    return hook
