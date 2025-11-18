# Fault-Tolerance Quickstart (from scratch)

## Create & activate a virtual env
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

## Install deps
pip install -r requirements.txt
# CPU-only PyTorch (works everywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (CUDA 12.x GPU) use: --index-url https://download.pytorch.org/whl/cu121

## Train a tiny MNIST baseline
python -m src.bench.mnist --train 1 --eval 1

## Run a fault-injection sweep (tensor flips)
python -m src.campaign.run --p 1e-6 --trials 200 --jsonl out/runs.jsonl

## Summarize
python -m src.analysis.summarize out/runs.jsonl
# computer_structures
