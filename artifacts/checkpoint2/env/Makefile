SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
EXPORTS := export PYTHONPATH=$(CURDIR) PYTORCH_ENABLE_MPS_FALLBACK=1

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  make setup                # create venv + install requirements"
	@echo "  make baseline             # train+eval MNIST baseline, save weights"
	@echo "  make drift-gate           # baseline accuracy gate (re-trains if missing)"
	@echo "  make sweep-linear-strong  # high-impact sweep on last Linear layer"
	@echo "  make report-linear-strong # HTML report for the sweep above"
	@echo "  make docker-build         # build CPU docker image"
	@echo "  make docker-drift         # run drift-gate inside docker"

.PHONY: setup
setup: .venv src/__init__.py src/utils/__init__.py
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@mkdir -p out data campaigns

.venv:
	python3 -m venv .venv

src/__init__.py:
	@mkdir -p src/utils
	@touch src/__init__.py src/utils/__init__.py

.PHONY: baseline
baseline:
	@$(EXPORTS); $(PYTHON) -m src.bench.mnist --train 1 --eval 1

.PHONY: drift-gate
drift-gate:
	@$(EXPORTS); $(PYTHON) -m src.ci.drift_gate --weights out/mnist_smallcnn.pt --min_acc 0.97 --train_if_missing 1

# ---------- Strong last-Linear sweep (visible impact) ----------
INJ_BITPOS ?= signexp   # sign | exp | signexp | any
INJ_K      ?= 16        # elements flipped per hooked layer per forward
P_LIST     ?= 1e-2,1e-1,3e-1,1
PLAN_FILE  ?= campaigns/mnist_linear_last_strong.yml
SWEEP_PREFIX ?= out/sweep_linear_last_signexpK16

campaigns/mnist_linear_last_strong.yml:
	@mkdir -p campaigns
	@cat > $@ <<'YAML'
name: mnist_linear_last_strong
dataset: { kind: mnist, split: test, batch_size: 64 }
model:   { kind: SmallCNN, weights: out/mnist_smallcnn.pt }
inject:  { enabled: true, p: 1.0, bit_width: 32, target_layers: ["Linear[-1]"] }
trials:  { max: 471, seed_base: 42000 }
logging: { path: out/runs_mnist_linear_last_strong.jsonl, validate: true }
eval:    { mode: margin, delta: 0.15 }   # label same but confidence drops => DEGRADED
resume:  false
YAML

.PHONY: sweep-linear-strong
sweep-linear-strong: campaigns/mnist_linear_last_strong.yml
	@$(EXPORTS); INJ_BITPOS=$(INJ_BITPOS) INJ_K=$(INJ_K) \
	$(PYTHON) -m src.campaign.sweep --plan $(PLAN_FILE) --p_list $(P_LIST) --out_prefix $(SWEEP_PREFIX)

.PHONY: report-linear-strong
report-linear-strong:
	@$(EXPORTS); $(PYTHON) -m src.analysis.report_html --glob '$(SWEEP_PREFIX)_p*.jsonl' --out out/report_linear_last_signexpK16.html
	@open out/report_linear_last_signexpK16.html || true

# ---------- Demo HTML from earlier sweeps (optional) ----------
.PHONY: demo-report
demo-report:
	@$(EXPORTS); $(PYTHON) -m src.analysis.report_html --glob 'out/sweep_mnist_*_p*.jsonl' --out out/report_demo.html
	@open out/report_demo.html || true

# ---------- Docker (CPU) ----------
.PHONY: docker-build
docker-build:
	docker build -t sdc-lab .

.PHONY: docker-drift
docker-drift:
	docker run --rm -it \
	  -v "$(CURDIR)/out:/app/out" \
	  -v "$(CURDIR)/data:/app/data" \
	  sdc-lab bash -lc 'python -m src.ci.drift_gate --weights out/mnist_smallcnn.pt --min_acc 0.97 --train_if_missing 1'
