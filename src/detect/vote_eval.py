from __future__ import annotations

import argparse, csv, os, time, json, random, math, os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from src.campaign.orchestrator import build_model, build_dataloader

# ---------------------------
# device & input preparation
# ---------------------------
def _select_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _prep_for_model(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Make a generic batch compatible with ResNet-18:
    - If batch is (N,1,*,*), repeat channels to 3.
    - Resize to 224x224 if needed.
    Safe no-op if already correct shape.
    """
    if x.dim() == 4 and x.shape[1] == 1:
        in_ch = getattr(getattr(m, "conv1", None), "in_channels", None)
        if in_ch in (None, 3):
            x = x.repeat(1, 3, 1, 1)
    if x.dim() == 4 and (x.shape[-2] != 224 or x.shape[-1] != 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x

# ---------------------------
# post-logit bit-flip injector
# ---------------------------
def _bit_mask(bit_mode: str, rng: random.Random) -> int:
    """
    Return a single-bit mask (1<<pos) for the requested mode.
    float32 layout: 0..22 mantissa, 23..30 exponent, 31 sign
    """
    if bit_mode == "sign":
        pos = 31
    elif bit_mode == "exp":
        pos = rng.randrange(23, 31)  # any exponent bit
    elif bit_mode == "signexp":
        # flip sign plus one exponent bit (combine masks)
        pos = 31
        exp_pos = rng.randrange(23, 31)
        return (1 << pos) | (1 << exp_pos)
    else:  # "any"
        pos = rng.randrange(0, 32)
    return (1 << pos)

def _inject_logits_inplace(
    logits: torch.Tensor,
    p: float,
    k: int,
    bit_mode: str,
    rng: random.Random,
) -> int:
    """
    With probability p per-sample, flip K logits (random classes) by toggling
    specific float32 bits ('sign' | 'exp' | 'signexp' | 'any').
    Returns number of samples injected.
    """
    if p <= 0 or k <= 0:
        return 0
    if logits.dtype != torch.float32:
        logits = logits.float()
    device = logits.device

    # operate on CPU via numpy for precise bit-level view
    arr = logits.detach().to("cpu").contiguous().numpy()  # shape [N,C]
    N, C = arr.shape
    injected = 0

    for i in range(N):
        if rng.random() < p:
            injected += 1
            # choose k distinct class indices
            idxs = rng.sample(range(C), k=min(k, C))
            for j in idxs:
                # view scalar as uint32, xor with mask, write back
                v = np.float32(arr[i, j])
                u = v.view(np.uint32)
                u ^= _bit_mask(bit_mode, rng)
                arr[i, j] = u.view(np.float32)

    # copy back to the original tensor
    logits.copy_(torch.from_numpy(arr).to(device))
    return injected

# ---------------------------
# main evaluation
# ---------------------------
@torch.no_grad()
def run(plan_path: str, passes: int, out_csv: str, p_override: float | None = None) -> None:
    plan = yaml.safe_load(open(plan_path))
    device = _select_device()

    # dataloader & model
    dl, dataset_desc = build_dataloader(plan["dataset"])
    m = build_model(plan["model"], device)
    wpath = plan["model"].get("weights")
    if wpath and os.path.exists(wpath):
        sd = torch.load(wpath, map_location=device)
        m.load_state_dict(sd, strict=False)
    m.eval()

    # injection knobs (env + CLI)
    p = float(p_override) if p_override is not None else float(os.environ.get("INJ_P", "0.5"))
    bit_mode = os.environ.get("INJ_BITPOS", "exp").strip().lower()  # exp|sign|signexp|any
    k = int(os.environ.get("INJ_K", "3"))
    rng = random.Random(os.environ.get("INJ_SEED", None))

    total = 0
    wrong_single = 0
    wrong_vote = 0
    disagree = 0
    total_injected_samples = 0

    t0 = time.time()
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to("cpu")

        x = _prep_for_model(m, x)

        # collect predictions across passes with fresh injections each pass
        preds = []
        injected_this_batch = 0
        for _ in range(passes):
            logits = m(x).clone()  # [B, num_classes]
            injected_this_batch += _inject_logits_inplace(logits, p=p, k=k, bit_mode=bit_mode, rng=rng)
            preds.append(logits.argmax(1).to("cpu"))
        total_injected_samples += injected_this_batch

        P = torch.stack(preds, dim=0)  # [S, B]
        wrong_single += (P[0] != y).sum().item()
        vote = P.mode(dim=0).values
        wrong_vote += (vote != y).sum().item()
        disagree += (P.max(dim=0).values != P.min(dim=0).values).sum().item()
        total += y.numel()

    t1 = time.time()
    wrong_single_rate = wrong_single / total if total else 0.0
    wrong_vote_rate   = wrong_vote   / total if total else 0.0
    disagr_rate       = disagree     / total if total else 0.0
    reduction_abs     = wrong_single_rate - wrong_vote_rate

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "samples","passes","wrong_single_rate","wrong_vote_rate","disagreement_rate",
            "reduction_abs","device","dataset","p","bit_mode","k","injected_samples","runtime_s"
        ])
        w.writerow([
            total, passes,
            f"{wrong_single_rate:.4f}", f"{wrong_vote_rate:.4f}", f"{disagr_rate:.4f}",
            f"{reduction_abs:.4f}", device, dataset_desc, p, bit_mode, k, total_injected_samples,
            f"{t1-t0:.2f}"
        ])

    print(f"[vote-eval] device= {device}")
    print(f"[vote-eval] dataset={dataset_desc} passes={passes} p={p} bit_mode={bit_mode} k={k}")
    print(f"[vote-eval] samples={total} injected_samples={total_injected_samples}")
    print(f"[vote-eval] 1-pass WRONG rate   = {wrong_single_rate:.4f}")
    print(f"[vote-eval] vote WRONG rate     = {wrong_vote_rate:.4f}")
    print(f"[vote-eval] disagreement rate   = {disagr_rate:.4f}")
    print(f"[vote-eval] reduction (absolute)= {reduction_abs:.4f}")
    print(f"[vote-eval] wrote {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--p_override", type=float, default=None, help="per-sample injection prob")
    args = ap.parse_args()
    run(args.plan, args.passes, args.out_csv, args.p_override)

if __name__ == "__main__":
    main()
