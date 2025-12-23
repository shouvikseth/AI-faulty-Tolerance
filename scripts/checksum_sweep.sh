#!/usr/bin/env bash
set -euo pipefail
# knobs
TAU_LIST=${TAU_LIST:-"0.97,0.98,0.99,0.995,0.999"}
PLAN=${PLAN:-campaigns/mnist_linear_last_vote.yml}  # p=0.3, Linear[-1]
OUTDIR=out/checksum_sweep
mkdir -p "$OUTDIR"

source .venv/bin/activate
export PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1

# keep faults “strong enough” to see signal (tweak if you want)
: "${INJ_BITPOS:=signexp}"
: "${INJ_K:=16}"

IFS=',' read -r -a ARR <<< "$TAU_LIST"
for tau in "${ARR[@]}"; do
  out="$OUTDIR/tau_${tau}.csv"
  echo "[run] tau=$tau -> $out"
  INJ_BITPOS="$INJ_BITPOS" INJ_K="$INJ_K" \
    python -m src.detect.checksum_eval --plan "$PLAN" --tau "$tau" --out_csv "$out"
done

echo "[sweep] results in $OUTDIR"
