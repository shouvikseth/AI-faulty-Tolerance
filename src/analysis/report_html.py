
# src/analysis/report_html.py
import os, json, argparse, math
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_bar(data_pairs, path, title, xlabel, ylabel):
    labels = [k for k,_ in data_pairs]
    vals   = [v for _,v in data_pairs]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(len(vals)), vals)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def save_hist(vals, path, title, xlabel, bins=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=bins if bins else 'auto')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="jsonl log file")
    ap.add_argument("--out", dest="out_html", default="out/report.html")
    args = ap.parse_args()

    rows = load_jsonl(args.inp)
    if not rows:
        raise SystemExit(f"no rows found in {args.inp}")

    # tallies
    outcome_ct = Counter(r.get("outcome","UNK") for r in rows)
    total = sum(outcome_ct.values())
    wrong = outcome_ct.get("WRONG", 0)
    sdc_rate = wrong / total if total else 0.0

    # per-layer WRONG Pareto
    wrong_by_layer = Counter()
    for r in rows:
        if r.get("outcome") == "WRONG":
            lid = r.get("layer_id","?")
            wrong_by_layer[lid] += 1

    pareto = sorted(wrong_by_layer.items(), key=lambda kv: kv[1], reverse=True)
    assets = "out/report_assets"
    ensure_dir(assets)

    p_out = os.path.join(assets, "pareto_wrong_by_layer.png")
    save_bar(pareto or [("none", 0)], p_out, "WRONG by layer_id (Pareto)", "layer_id", "WRONG count")

    # bitpos histogram (when present)
    bitpos_vals = [r["bitpos"] for r in rows if "bitpos" in r and r["bitpos"] is not None]
    bp_out = os.path.join(assets, "bitpos_hist.png")
    if bitpos_vals:
        save_hist(bitpos_vals, bp_out, "Bit position histogram", "bit position (0..31)", bins=32)

    # simple outcome bar
    oc_out = os.path.join(assets, "outcomes.png")
    save_bar(sorted(outcome_ct.items()), oc_out, "Outcomes", "outcome", "count")

    # html
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>SDC Report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
h1,h2 {{ margin-bottom: 0.2rem; }}
.card {{ border: 1px solid #ccc; padding: 16px; border-radius: 8px; margin-bottom: 16px; }}
small {{ color: #666; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>SDC Report</h1>
<small>generated {ts}</small>

<div class="card">
  <h2>Summary</h2>
  <p><b>File:</b> {args.inp}</p>
  <p><b>Total trials:</b> {total} &nbsp; <b>WRONG:</b> {wrong} &nbsp; <b>SDC rate:</b> {sdc_rate:.4%}</p>
</div>

<div class="card">
  <h2>Outcomes</h2>
  <img src="{os.path.relpath(oc_out, start=os.path.dirname(args.out_html))}">
</div>

<div class="card">
  <h2>Pareto — WRONG by layer</h2>
  <img src="{os.path.relpath(p_out, start=os.path.dirname(args.out_html))}">
</div>

{"<div class='card'><h2>Bit position histogram</h2><img src='" + os.path.relpath(bp_out, start=os.path.dirname(args.out_html)) + "'></div>" if bitpos_vals else ""}

</body></html>
"""
    # ensure parent
    out_dir = os.path.dirname(args.out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_html, "w") as f:
        f.write(html)

    print(f"[report] wrote {args.out_html}")

if __name__ == "__main__":
    main()
