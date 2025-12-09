#!/usr/bin/env bash
set -euo pipefail

# --- config ---
OUTDIR="artifacts/checkpoint2"
REPORT_DIR="$OUTDIR/reports"
LOG_DIR="$OUTDIR/logs"
PLAN_DIR="$OUTDIR/plans"
ENV_DIR="$OUTDIR/env"

# --- env prep (non-fatal if venv not present) ---
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- folders ---
mkdir -p "$REPORT_DIR" "$LOG_DIR" "$PLAN_DIR" "$ENV_DIR"

# --- collect logs (all sweeps) ---
LOGS=$(ls out/sweep_*_p*.jsonl 2>/dev/null || true)
if [ -z "${LOGS}" ]; then
  echo "[checkpoint2] No sweep logs found at out/sweep_*_p*.jsonl"
  echo "Run a sweep, e.g.:"
  echo "  INJ_BITPOS=signexp INJ_K=16 python -m src.campaign.sweep --plan campaigns/mnist_linear_last_strong.yml --p_list 1e-2,1e-1,3e-1,1 --out_prefix out/sweep_linear_last_signexpK16"
  exit 2
fi

# copy logs into bundle
for f in $LOGS; do
  cp -f "$f" "$LOG_DIR/"
done

# also copy any single-run logs you might want to retain
for f in out/runs_*.jsonl; do
  [ -e "$f" ] && cp -f "$f" "$LOG_DIR/" || true
done

# --- validate all logs ---
VALID_SUMMARY="$REPORT_DIR/validation_summary.txt"
: > "$VALID_SUMMARY"
for f in "$LOG_DIR"/*.jsonl; do
  echo "[validate] $f" | tee -a "$VALID_SUMMARY"
  python -m src.utils.validate --in "$f" --schema schemas/run.schema.json | tee -a "$VALID_SUMMARY"
done

# --- build HTML dashboard over all sweep logs in the bundle ---
python -m src.analysis.report_html \
  --glob "$LOG_DIR/sweep_*_p*.jsonl" \
  --out "$REPORT_DIR/checkpoint2.html"

# --- build CSV aggregates ---
python -m src.analysis.aggregate \
  --glob "$LOG_DIR/sweep_*_p*.jsonl" \
  --out_prefix "$REPORT_DIR/checkpoint2"

# --- build resilience curve (WRONG+DEGRADED) ---
python -m src.analysis.resilience_curve \
  --glob "$LOG_DIR/sweep_*_p*.jsonl" \
  --out_png "$REPORT_DIR/resilience.png" \
  --metric wrong+degraded

# --- snapshot plans (if present) ---
for p in campaigns/mnist_linear_last_strong.yml campaigns/mnist_conv_strong.yml campaigns/mnist_demo.yml campaigns/mnist_linear_all.yml; do
  [ -f "$p" ] && cp -f "$p" "$PLAN_DIR/"
done

# --- snapshot env files (if present) ---
for e in requirements.txt requirements-docker.txt Makefile Dockerfile .dockerignore; do
  [ -f "$e" ] && cp -f "$e" "$ENV_DIR/"
done

# --- metadata ---
METAFILE="$OUTDIR/metadata.json"
python - << 'PY' > "$METAFILE"
import json, os, subprocess, sys, torch, datetime, glob
def git_hash():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception:
        return None
def device_name():
    if torch.cuda.is_available(): return "cuda"
    try:
        if torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"
def list_logs():
    return sorted(glob.glob("artifacts/checkpoint2/logs/*.jsonl"))
meta = {
    "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_commit": git_hash(),
    "device": device_name(),
    "env": {
        "INJ_BITPOS": os.environ.get("INJ_BITPOS"),
        "INJ_K": os.environ.get("INJ_K"),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    },
    "logs": list_logs(),
    "reports": {
        "html": "artifacts/checkpoint2/reports/checkpoint2.html",
        "resilience_png": "artifacts/checkpoint2/reports/resilience.png",
        "csv_layer_p": "artifacts/checkpoint2/reports/checkpoint2_layer_p.csv",
        "csv_bitpos": "artifacts/checkpoint2/reports/checkpoint2_bitpos.csv",
        "validation_summary": "artifacts/checkpoint2/reports/validation_summary.txt"
    }
}
json.dump(meta, sys.stdout, indent=2)
PY

# --- human README ---
CHECK_MD="$OUTDIR/CHECKPOINT_2.md"
cat > "$CHECK_MD" <<'MD'
# Checkpoint 2

This bundle captures the second milestone of the fault-injection project.

## What’s inside
- `reports/checkpoint2.html` — interactive summary (WRONG/DEGRADED vs p, per-layer×p, bit histogram)
- `reports/resilience.png` — resilience curve (WRONG+DEGRADED rate vs p)
- `reports/checkpoint2_layer_p.csv` — per-layer × p summary
- `reports/checkpoint2_bitpos.csv` — bit-position histogram per layer
- `reports/validation_summary.txt` — JSONL schema validation output
- `logs/` — all JSONL logs included in this checkpoint
- `plans/` — YAML plans used (copied if present)
- `env/` — requirements, Makefile, Dockerfile (if present)
- `metadata.json` — creation time, git commit, device, env knobs

## View
open artifacts/checkpoint2/reports/checkpoint2.html
open artifacts/checkpoint2/reports/resilience.png
MD

echo "[checkpoint2] Done:"
echo "  - $CHECK_MD"
echo "  - $METAFILE"
echo "  - $REPORT_DIR/checkpoint2.html"
echo "  - $REPORT_DIR/resilience.png"
echo "  - $REPORT_DIR/checkpoint2_layer_p.csv"
echo "  - $REPORT_DIR/checkpoint2_bitpos.csv"
