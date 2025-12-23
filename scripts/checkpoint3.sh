#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1

# regenerate report to be safe
bash scripts/reproduce_cifar.sh >/dev/null 2>&1 || true

dst="out/checkpoint3"
rm -rf "$dst"
mkdir -p "$dst/csv" "$dst/plots" "$dst/campaigns" "$dst/scripts"

# copy artifacts
cp out/*.csv "$dst/csv/" 2>/dev/null || true
cp out/*frontier*.png "$dst/plots/" 2>/dev/null || true
cp out/resilience_cifar_logits.png "$dst/plots/" 2>/dev/null || true
cp out/cifar_report.html "$dst/" 2>/dev/null || true
cp campaigns/cifar_resnet18.yml "$dst/campaigns/" 2>/dev/null || true
cp scripts/reproduce_cifar.sh "$dst/scripts/"

# tiny README
cat > "$dst/README_CHECKPOINT3.md" << 'MD'
# Checkpoint 3

**What’s inside**
- `cifar_report.html` — open in a browser for the summary + frontier
- `plots/` — frontier + resilience curves
- `csv/` — vote / checksum / hybrid raw results
- `campaigns/cifar_resnet18.yml` — the plan used
- `scripts/reproduce_cifar.sh` — one-command rerun

**How to view**
1. Double-click `cifar_report.html` (or `open out/checkpoint3/cifar_report.html` from Terminal).
2. The scatter labels show method and cost× vs error rate.

**Environment**
- macOS + MPS; fault injectors: post-logit, bit_mode=exp, K=3; p=0.5.
- Voting: 3 & 5 passes. Checksum: τ∈{0.9980…0.9995}. Hybrid: 3-pass with route_thresh ∈ {0.25, 0.33, 0.40}.

**Headline**
- Best *accuracy per cost* here is **checksum** (~2.7×, ~9.3% error).
- **Hybrid-3** gets ~9.8–9.9% at ~3.26× if you want a vote-guarded variant.
- **Vote-5** alone is ~11.3% at ~5×.

MD

# zip it
( cd out && zip -qr checkpoint3.zip checkpoint3 )
echo "✅ Wrote out/checkpoint3.zip (and folder out/checkpoint3)"
