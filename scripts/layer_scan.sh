#!/usr/bin/env bash
set -euo pipefail

# knobs (you can override before calling)
P_LIST=${P_LIST:-"1e-2,1e-1,3e-1,1"}
LAYERS=("Conv2d[0]" "Conv2d[1]" "Conv2d[2]" "Linear[0]" "Linear[-1]")
TRIALS=${TRIALS:-471}

# injector env defaults (override if you like)
: "${INJ_BITPOS:=signexp}"
: "${INJ_K:=16}"

# env
source .venv/bin/activate
export PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p out/layers

for L in "${LAYERS[@]}"; do
  # sanitize for filenames (replace [, ], and spaces with _)
  SAN=$(echo "$L" | sed 's/[][ ]/_/g')
  IFS=',' read -r -a ARR <<< "$P_LIST"
  for p in "${ARR[@]}"; do
    PLAN_TMP=$(mktemp -t layerplan.XXXXXX.yml)
    cat > "$PLAN_TMP" <<YAML
name: layer_scan_${SAN}_p${p}
dataset: { kind: mnist, split: test, batch_size: 64 }
model:   { kind: SmallCNN, weights: out/mnist_smallcnn.pt }
inject:
  enabled: true
  p: ${p}
  bit_width: 32
  target_layers: ["${L}"]
trials:  { max: ${TRIALS}, seed_base: 52000 }
logging: { path: "out/layers/${SAN}_p${p}.jsonl", validate: true }
eval:    { mode: margin, delta: 0.15 }
resume:  false
YAML
    echo "[run] LAYER=${L}  p=${p}"
    INJ_BITPOS="$INJ_BITPOS" INJ_K="$INJ_K" \
      python -m src.campaign.orchestrator --plan "$PLAN_TMP" --resume 0
    rm -f "$PLAN_TMP"
  done
done

echo "[layer-scan] done → out/layers/*.jsonl"
