#!/usr/bin/env bash
set -euo pipefail

# knobs (override by: BIT_LIST="any,exp,sign,signexp" K_LIST="1,4,16,64" P=0.3 TRIALS=471)
BIT_LIST=${BIT_LIST:-"any,exp,sign,signexp"}
K_LIST=${K_LIST:-"1,4,16,64"}
P=${P:-0.3}
TRIALS=${TRIALS:-471}

source .venv/bin/activate
export PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p out/bitk

for BIT in ${BIT_LIST//,/ }; do
  for K in ${K_LIST//,/ }; do
    PLAN_TMP=$(mktemp -t bitkplan.XXXXXX.yml)
    OUT="out/bitk/bitk_linear_last_${BIT}_K${K}_p${P}.jsonl"
    cat > "$PLAN_TMP" <<YAML
name: bitk_linear_last_${BIT}_K${K}_p${P}
dataset: { kind: mnist, split: test, batch_size: 64 }
model:   { kind: SmallCNN, weights: out/mnist_smallcnn.pt }
inject:
  enabled: true
  p: ${P}
  bit_width: 32
  target_layers: ["Linear[-1]"]
trials:  { max: ${TRIALS}, seed_base: 61000 }
logging: { path: "${OUT}", validate: true }
eval:    { mode: margin, delta: 0.15 }
resume:  false
YAML
    echo "[run] BIT=${BIT}  K=${K}  p=${P} -> ${OUT}"
    INJ_BITPOS="${BIT}" INJ_K="${K}" \
      python -m src.campaign.orchestrator --plan "$PLAN_TMP" --resume 0
    rm -f "$PLAN_TMP"
  done
done

echo "[bitk] logs → out/bitk/*.jsonl"
