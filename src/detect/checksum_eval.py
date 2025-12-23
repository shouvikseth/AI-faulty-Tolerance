from __future__ import annotations

import argparse, csv, os, time, random, math
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
    # Resize to 224x224 for ResNet
    if x.dim() == 4 and (x.shape[-2] != 224 or x.shape[-1] != 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x

# ---------------------------
# post-logit injector
# ---------------------------
def _bit_mask(mode: str, rng: random.Random) -> int:
    # float32: 0..22 mantissa, 23..30 exponent, 31 sign
    mode = mode.lower()
    if mode == "sign":
        return 1 << 31
    if mode == "exp":
        return 1 << rng.randrange(23, 31)
    if mode == "signexp":
        return (1 << 31) | (1 << rng.randrange(23, 31))
    # "any"
    return 1 << rng.randrange(0, 32)

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

# ---------------------------
# signals
# ---------------------------
def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # per-sample cosine similarity over class dimension
    return F.cosine_similarity(a, b, dim=1)

def _margin(logits: torch.Tensor) -> torch.Tensor:
    # top1 - top2 margin per sample
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]

# ---------------------------
# main
# ---------------------------
@torch.no_grad()
def run(plan_path: str, out_csv: str, mode: str, tau: float, delta: float, digits: int,
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

    # fault knobs
    p = float(p_override) if p_override is not None else float(os.environ.get("INJ_P", "0.5"))
    bit_mode = os.environ.get("INJ_BITPOS", "exp")
    k = int(os.environ.get("INJ_K", "3"))
    rng = random.Random(os.environ.get("INJ_SEED", None))

    # stats
    total = 0
    wrong_1pass = 0
    detected = 0
    corrected = 0
    clean_forwards = 0

    t0 = time.time()
    for x, y in dl:
        x = _prep_for_model(m, x.to(device, non_blocking=True))
        y = y.to("cpu")
        B = y.numel()

        # pass A (noisy)
        logits_a = m(x).clone()
        _inject_logits_inplace(logits_a, p, k, bit_mode, rng)
        pred_a = logits_a.argmax(1).to("cpu")

        # pass B (noisy, independent)
        logits_b = m(x).clone()
        _inject_logits_inplace(logits_b, p, k, bit_mode, rng)

        # signals
        s_cos = _cosine(logits_a, logits_b).to("cpu")     # higher is "more similar"
        s_md  = (_margin(logits_a) - _margin(logits_b)).abs().to("cpu")  # larger means bigger change

        # decision: suspicious if cosine < tau OR margin_diff > delta
        suspicious = (s_cos < tau) | (s_md > delta)
        n_susp = suspicious.sum().item()
        detected += n_susp

        # baseline wrong on first pass
        wrong_1pass += (pred_a != y).sum().item()

        # for suspicious samples, run a CLEAN pass and take that prediction
        if n_susp:
            clean_forwards += n_susp
            # gather indices
            idx = torch.nonzero(suspicious).squeeze(1)
            x_fix = x[idx]
            logits_fix = m(x_fix)  # clean
            pred_fix = logits_fix.argmax(1).to("cpu")
            # how many corrections?
            # corrected if baseline pred was wrong but clean pred is right
            corrected += ((pred_a[idx] != y[idx]) & (pred_fix == y[idx])).sum().item()
            # for "final wrong", replace pred_a at suspicious spots with clean pred
            pred_a[idx] = pred_fix

        # (we could compute disagree(A,B) for logging; not strictly needed)
        total += B

    t1 = time.time()
    wrong_1pass_rate = wrong_1pass / total if total else 0.0
    final_wrong_rate = (wrong_1pass - corrected) / total if total else 0.0
    detected_rate    = detected / total if total else 0.0
    corrected_rate   = corrected / total if total else 0.0
    # cost = one base forward + one extra noisy forward always + clean reruns for detected
    # i.e., ~ 2 + detected_rate extra forwards; report as multiplier
    cost_x = 2.0 + detected_rate

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "samples","mode","tau","delta","digits","p","bit_mode","k",
            "wrong_1pass_rate","detected_rate","corrected_rate","final_wrong_rate",
            "cost_x","device","dataset","runtime_s"
        ])
        w.writerow([
            total, mode, tau, delta, digits, p, bit_mode, k,
            f"{wrong_1pass_rate:.4f}", f"{detected_rate:.4f}",
            f"{corrected_rate:.4f}", f"{final_wrong_rate:.4f}",
            f"{cost_x:.2f}", _select_device(), dataset_desc, f"{t1-t0:.2f}"
        ])

    print(f"[checksum] device={_select_device()} mode={mode} targets=['post-logits'] p={p} INJ_BITPOS={bit_mode} K={k}")
    print(f"[checksum] tau={tau:.4f} delta={delta:.3f} digits={digits} samples={total}")
    print(f"[checksum] 1-pass WRONG rate     = {wrong_1pass_rate:.4f}")
    print(f"[checksum] detected rate         = {detected_rate:.4f}")
    print(f"[checksum] corrected rate        = {corrected_rate:.4f}")
    print(f"[checksum] final WRONG rate      = {final_wrong_rate:.4f}")
    print(f"[checksum] est. forward cost     = {cost_x:.2f}× per sample")
    print(f"[checksum] wrote {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--mode", default="mix", help="kept for CLI symmetry; detection uses cos+margin")
    ap.add_argument("--tau", type=float, default=0.9990)
    ap.add_argument("--delta", type=float, default=0.080)
    ap.add_argument("--digits", type=int, default=2, help="placeholder; not used in this simple mix")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--p_override", type=float, default=None)
    args = ap.parse_args()
    run(args.plan, args.out_csv, args.mode, args.tau, args.delta, args.digits, args.p_override)

if __name__ == "__main__":
    main()
