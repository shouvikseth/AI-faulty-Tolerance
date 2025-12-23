from __future__ import annotations

import argparse, csv, os, time, random
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from src.campaign.orchestrator import build_model, build_dataloader

# ---------------------------
# device & input prep
# ---------------------------
def _select_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _prep_for_model(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # If (N,1,H,W) and model wants 3ch (e.g., ResNet), repeat channels
    if x.dim() == 4 and x.shape[1] == 1:
        in_ch = getattr(getattr(m, "conv1", None), "in_channels", None)
        if in_ch in (None, 3):
            x = x.repeat(1, 3, 1, 1)
    # Resize to 224x224 for ResNet-18
    if x.dim() == 4 and (x.shape[-2] != 224 or x.shape[-1] != 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x

# ---------------------------
# post-logit injector
# ---------------------------
def _bit_mask(mode: str, rng: random.Random) -> int:
    mode = mode.lower()
    if mode == "sign":
        return 1 << 31                  # sign bit
    if mode == "exp":
        return 1 << rng.randrange(23, 31)   # exponent bits 23..30
    if mode == "signexp":
        return (1 << 31) | (1 << rng.randrange(23, 31))
    return 1 << rng.randrange(0, 32)    # "any" (could be mantissa, exp, or sign)

def _inject_logits_inplace(logits: torch.Tensor, p: float, k: int, bit_mode: str,
                           rng: random.Random) -> int:
    if p <= 0 or k <= 0:
        return 0
    if logits.dtype != torch.float32:
        logits = logits.float()
    arr = logits.detach().to("cpu").contiguous().numpy()  # [B,C]
    B, C = arr.shape
    injected = 0
    for i in range(B):
        if rng.random() < p:
            injected += 1
            cols = rng.sample(range(C), k=min(k, C))
            for j in cols:
                v = np.float32(arr[i, j])
                u = v.view(np.uint32)
                u ^= _bit_mask(bit_mode, rng)
                arr[i, j] = u.view(np.float32)
    logits.copy_(torch.from_numpy(arr).to(logits.device))
    return injected

def _mode_along_rows(preds: torch.Tensor) -> torch.Tensor:
    # preds: [S, B], returns per-sample majority vote (ties broken by earliest)
    S, B = preds.shape
    final = torch.empty(B, dtype=preds.dtype)
    for b in range(B):
        vals, counts = torch.unique(preds[:, b], return_counts=True)
        idx = torch.argmax(counts)  # tie-breaks by first max occurrence
        final[b] = vals[idx]
    return final

# ---------------------------
# main
# ---------------------------
@torch.no_grad()
def run(plan_path: str, passes: int, route_thresh: float, out_csv: str,
        p_override: float | None = None) -> None:
    plan = yaml.safe_load(open(plan_path))
    device = _select_device()

    # dataloader + model
    dl, dataset_desc = build_dataloader(plan["dataset"])
    m = build_model(plan["model"], device)
    wpath = plan["model"].get("weights")
    if wpath and os.path.exists(wpath):
        sd = torch.load(wpath, map_location=device)
        m.load_state_dict(sd, strict=False)
    m.eval()

    # fault knobs from env / override
    import os as _os
    p = float(p_override) if p_override is not None else float(_os.environ.get("INJ_P", "0.5"))
    bit_mode = _os.environ.get("INJ_BITPOS", "exp")
    k = int(_os.environ.get("INJ_K", "3"))
    rng = random.Random(_os.environ.get("INJ_SEED", None))

    total = 0
    wrong_single = 0
    wrong_vote = 0
    routed = 0
    corrected = 0

    t0 = time.time()
    for x, y in dl:
        x = _prep_for_model(m, x.to(device, non_blocking=True))
        y = y.to("cpu")
        B = y.numel()

        # S noisy passes
        preds = []
        for s in range(passes):
            logits = m(x).clone()
            _inject_logits_inplace(logits, p, k, bit_mode, rng)
            preds.append(logits.argmax(1).to("cpu"))
        preds = torch.stack(preds, dim=0)    # [S, B]

        # metrics: single pass (s=0) and majority vote
        single = preds[0]
        vote   = _mode_along_rows(preds)

        wrong_single += (single != y).sum().item()
        wrong_vote   += (vote != y).sum().item()

        # per-sample disagreement = fraction of passes not equal to the vote
        disagree = (preds != vote.unsqueeze(0)).float().mean(dim=0)  # [B]
        route_mask = disagree >= route_thresh
        n_route = route_mask.sum().item()
        routed += n_route

        if n_route:
            idx = torch.nonzero(route_mask).squeeze(1)
            x_fix = x[idx]
            clean = m(x_fix).argmax(1).to("cpu")
            # count how many vote predictions would be fixed by clean rerun
            corrected += ((vote[idx] != y[idx]) & (clean == y[idx])).sum().item()
            # replace vote with clean for final decision
            vote[idx] = clean

        total += B

    t1 = time.time()
    wrong_single_rate = wrong_single / total if total else 0.0
    wrong_vote_rate   = wrong_vote / total if total else 0.0  # before routing correction (for reporting)
    final_wrong_rate  = ( (wrong_vote - corrected) / total ) if total else 0.0
    routed_rate       = routed / total if total else 0.0
    # cost ≈ S noisy passes + clean reruns for routed samples
    cost_x = passes + routed_rate

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "samples","passes","route_thresh","p","bit_mode","k",
            "wrong_single_rate","wrong_vote_rate","routed_rate","corrected_rate","final_wrong_rate",
            "cost_x","device","dataset","runtime_s"
        ])
        w.writerow([
            total, passes, route_thresh, p, bit_mode, k,
            f"{wrong_single_rate:.4f}", f"{wrong_vote_rate:.4f}", f"{routed_rate:.4f}",
            f"{(corrected/total if total else 0.0):.4f}", f"{final_wrong_rate:.4f}",
            f"{cost_x:.2f}", _select_device(), dataset_desc, f"{t1-t0:.2f}"
        ])

    print(f"[hybrid] device={_select_device()} dataset={dataset_desc} passes={passes} p={p} bit_mode={bit_mode} k={k}")
    print(f"[hybrid] route_thresh={route_thresh:.2f}  -> routed_rate ~ {routed_rate:.4f}")
    print(f"[hybrid] 1-pass WRONG = {wrong_single_rate:.4f}")
    print(f"[hybrid] vote WRONG   = {wrong_vote_rate:.4f}")
    print(f"[hybrid] final WRONG  = {final_wrong_rate:.4f}")
    print(f"[hybrid] cost ≈ {cost_x:.2f}x")
    print(f"[hybrid] wrote {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--route_thresh", type=float, default=0.40, help="fraction of passes disagreeing with the vote to trigger clean rerun")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--p_override", type=float, default=None)
    args = ap.parse_args()
    run(args.plan, args.passes, args.route_thresh, args.out_csv, args.p_override)

if __name__ == "__main__":
    main()
