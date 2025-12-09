import os, textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT_PDF = "out/checkpoint2_report.pdf"

def add_text_page(pdf, title, blocks):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    y = 0.95
    ax.text(0.5, y, title, ha="center", va="top", fontsize=20, fontweight="bold")
    y -= 0.05
    left = 0.08
    wrap = 98

    for kind, content in blocks:
        if kind == "p":
            for line in textwrap.wrap(content, width=wrap):
                ax.text(left, y, line, ha="left", va="top", fontsize=12)
                y -= 0.03
            y -= 0.01
        elif kind == "ul":
            for item in content:
                for i, line in enumerate(textwrap.wrap(item, width=wrap-4)):
                    prefix = "• " if i == 0 else "  "
                    ax.text(left, y, prefix + line, ha="left", va="top", fontsize=12)
                    y -= 0.03
                y -= 0.005
            y -= 0.01
        elif kind == "code":
            y -= 0.01
            for line in content.split("\n"):
                ax.text(left+0.01, y, line, ha="left", va="top", fontsize=9, family="monospace")
                y -= 0.028
            y -= 0.01

        if y < 0.08:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0,0,1,1]); ax.axis("off")
            y = 0.95
            ax.text(0.5, y, title + " (cont.)", ha="center", va="top", fontsize=20, fontweight="bold")
            y -= 0.05

    pdf.savefig(fig); plt.close(fig)

with PdfPages(OUT_PDF) as pdf:
    add_text_page(pdf, "Structures Project — Checkpoint 2 Report", [
        ("p", "Project: Fault Injection & Resilience Framework"),
        ("p", "As of: December 8, 2025"),
        ("p", "Executive summary"),
        ("p", "Since Checkpoint-1, we stabilized the end-to-end tensor-fault pipeline on MNIST (training, drift-gate, injection, logging, validation, analysis), introduced margin-based degradation labeling, and produced sweep reports (HTML/CSV/PNG) plus a checkpoint-bundle script. The prior import-path blocker is addressed in practice (validated by successful module runs, schema validation, and report generation). We now have a reproducible loop to quantify resilience versus injection probability, bit category, target layer, and K (elements flipped per forward). This positions us to scale trials and extend beyond MNIST, and to bring back the ISA path when ready."),
        ("p", "Why this matters: It operationalizes the core CP-1 objectives—inject faults, run large trial sets, and analyze SDC/robustness—on a fully reproducible basis."),
    ])

    add_text_page(pdf, "Scope & Objectives (unchanged from CP-1)", [
        ("ul", [
            "Robust fault-injection engine (tensor now; ISA path retained in scope).",
            "Large-scale sweeps (1k–100k trials as target envelope).",
            "Robustness metrics: SDC, CRASH, layer vulnerability.",
            "Clean baselines, CI gating, automated campaigns.",
        ]),
        ("p", "Key CP-1 deliverables we continue to build on: tensor/ISA hooks; YAML-driven runner; metrics & analysis; dataset/model baselines; structured logging; reproducible sweeps."),
    ])

    add_text_page(pdf, "What Changed Since Checkpoint-1", [
        ("p", "3.1 Blocker resolution (import path) → working pipeline"),
        ("p", "We addressed import/path failures by ensuring proper package layout (src/__init__.py), exporting PYTHONPATH=$PWD, and validating via repeated successful runs of training, orchestrator, and validators."),
        ("p", "3.2 Baseline & drift gate"),
        ("p", "Trained baseline SmallCNN on MNIST; saved weights; verified accuracy via drift-gate thresholding. This makes later WRONG/DEGRADED calls meaningful against a stable baseline."),
        ("p", "3.3 Injection & knobs (operational)"),
        ("ul", [
            "Hooks attach to selected layers (e.g., Linear[-1] or Conv2d*).",
            "Bit categories: sign, exp, signexp, any to control severity.",
            "K: elements flipped per forward (e.g., 1, 4, 16, 64).",
            "p: per-forward injection probability (swept across values).",
            "Guards: NaN/Inf checks and clipping retained (tunable).",
        ]),
        ("p", "3.4 Outcomes & labeling"),
        ("ul", [
            "WRONG (label changed), CLEAN (unchanged), CRASH (exception).",
            "DEGRADED (label same but margin dropped ≥ δ using margin mode).",
        ]),
        ("p", "3.5 Campaigns, validation, and reports"),
        ("ul", [
            "YAML plans drive runs and support sharding/resume.",
            "JSONL logs validated against run.schema.json.",
            "Analysis emits HTML dashboards, CSV aggregates, and resilience curves.",
        ]),
        ("p", "3.6 Packaging “Checkpoint 2”"),
        ("p", "Scripted artifact bundle: collects sweep logs, validates, builds HTML/CSV/PNG, snapshots plans/env, writes metadata."),
    ])

    add_text_page(pdf, "Results Highlights", [
        ("ul", [
            "Effect visibility: Targeting the last Linear layer with stronger flips (signexp) and larger K (e.g., 16) yields clear monotonic increases in WRONG/DEGRADED rates as p grows.",
            "Layer sensitivity: Early Conv layers often show smaller user-visible impact (activations + pooling damp noise). Linear-last is a reliable “canary.”",
            "Bit category: sign/exp flips are typically more disruptive than mantissa (any), as expected from FP32 layout.",
            "Validation: Logs across runs/sweeps validate cleanly against schema—key for trust and automation.",
        ]),
    ])

    add_text_page(pdf, "Risks & Mitigations", [
        ("ul", [
            "Damped effect in early layers → Mitigation: emphasize Linear[-1] for headline curves; run per-layer sweeps for mapping.",
            "Guards hide impact → Mitigation: establish sensitivity with strong settings, then restore guards for realistic ops.",
            "Throughput on macOS MPS vs CPU/Docker → Mitigation: Docker CPU for portability; MPS locally for speed.",
        ]),
    ])

    add_text_page(pdf, "What’s Ready Now", [
        ("ul", [
            "Reproducible baseline & drift-gate.",
            "Tensor-injection campaigns with configurable p, K, bitpos, targets.",
            "Deterministic site selection and resume/sharding.",
            "Validated JSONL + schema; dashboard/report generation; checkpoint bundling.",
        ]),
    ])

    add_text_page(pdf, "What’s Next (Near-Term)", [
        ("ul", [
            "Layer map: per-layer sweeps (Conv blocks, Linear[0], Linear[-1]) to produce a vulnerability heatmap.",
            "Bit-sensitivity map: sweep sign/exp/any at fixed K, per target layer; export comparative reports.",
            "Guard sensitivity: sweep clipping δ to quantify robustness/accuracy trade-offs.",
            "Scale trials: exercise 1k–10k trials on reliable plans to tighten error bars.",
            "Bring back ISA path: re-enable the RISC-V flip path and create parity plots versus tensor path.",
            "CI integration: nightly small sweeps + drift-gate; publish HTML/CSV artifacts.",
        ]),
    ])

    add_text_page(pdf, "How to Reproduce (Single-Screen)", [
        ("code", """# activate env
source .venv/bin/activate
export PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1

# strong settings for visible effect at last Linear
export INJ_BITPOS=signexp
export INJ_K=16

# sweep over p
python -m src.campaign.sweep --plan campaigns/mnist_linear_last_strong.yml \
  --p_list 1e-2,1e-1,3e-1,1 \
  --out_prefix out/sweep_linear_last_signexpK16

# reports
python -m src.analysis.report_html --glob 'out/sweep_linear_last_signexpK16_p*.jsonl' \
  --out out/report_linear_last_signexpK16.html
open out/report_linear_last_signexpK16.html

# optional: checkpoint bundle (HTML/CSV/PNG + metadata)
bash scripts/checkpoint2.sh"""),
    ])

    add_text_page(pdf, "Alignment with CP-1 Plan", [
        ("ul", [
            "Deliverables: hooks, CLI/YAML runner, metrics/analysis, baselines, logging, reproducible sweeps — all present and exercised in CP-2.",
            "Blocker from CP-1: import path issues — addressed via packaging and env setup; analysis now runs.",
            "Next-steps continuity: visualization and multi-model campaigns remain the immediate roadmap now enabled by this foundation.",
        ]),
    ])

print(f"[writeup] Wrote {OUT_PDF}")
