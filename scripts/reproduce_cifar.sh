#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1
export INJ_HOOK=post INJ_PER_SAMPLE=1 INJ_BITPOS=exp INJ_K=3

# Voting baselines
python -m src.detect.vote_eval --plan campaigns/cifar_resnet18.yml --passes 3 --p_override 0.5 --out_csv out/vote_cifar_p05_3pass.csv
python -m src.detect.vote_eval --plan campaigns/cifar_resnet18.yml --passes 5 --p_override 0.5 --out_csv out/vote_cifar_p05_5pass.csv

# Checksum sweep around good τ
python -m src.detect.checksum_eval --plan campaigns/cifar_resnet18.yml --mode mix --tau 0.9980 --delta 0.080 --digits 2 --p_override 0.5 --out_csv out/checksum_cifar_tau_0.9980.csv
python -m src.detect.checksum_eval --plan campaigns/cifar_resnet18.yml --mode mix --tau 0.9985 --delta 0.080 --digits 2 --p_override 0.5 --out_csv out/checksum_cifar_tau_0.9985.csv
python -m src.detect.checksum_eval --plan campaigns/cifar_resnet18.yml --mode mix --tau 0.9990 --delta 0.080 --digits 2 --p_override 0.5 --out_csv out/checksum_cifar_tau_0.9990.csv
python -m src.detect.checksum_eval --plan campaigns/cifar_resnet18.yml --mode mix --tau 0.9993 --delta 0.080 --digits 2 --p_override 0.5 --out_csv out/checksum_cifar_tau_0.9993.csv
python -m src.detect.checksum_eval --plan campaigns/cifar_resnet18.yml --mode mix --tau 0.9995 --delta 0.080 --digits 2 --p_override 0.5 --out_csv out/checksum_cifar_tau_0.9995.csv

# Hybrid (3-pass thresholds)
python -m src.detect.hybrid_eval --plan campaigns/cifar_resnet18.yml --passes 3 --route_thresh 0.40 --p_override 0.5 --out_csv out/hybrid_cifar_p05_3pass_t040.csv
python -m src.detect.hybrid_eval --plan campaigns/cifar_resnet18.yml --passes 3 --route_thresh 0.33 --p_override 0.5 --out_csv out/hybrid_cifar_p05_3pass_t033.csv
python -m src.detect.hybrid_eval --plan campaigns/cifar_resnet18.yml --passes 3 --route_thresh 0.25 --p_override 0.5 --out_csv out/hybrid_cifar_p05_3pass_t025.csv

# Build frontier figure + HTML report
python - << 'PY'
import csv, glob, re, html
import matplotlib.pyplot as plt

def last_row(p):
    with open(p, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}

def fget(d, *names):
    nmap = { re.sub(r'[^a-z0-9]', '', k.lower()): v for k,v in d.items() }
    for name in names:
        if name in d and d[name] != "": return d[name]
        key = re.sub(r'[^a-z0-9]', '', name.lower())
        if key in nmap and nmap[key] != "": return nmap[key]
    return None

def fnum(x, default=None):
    try: return float(x)
    except: return default

rows, frontier = [], []

# vote
for f in sorted(glob.glob("out/vote_*pass.csv")):
    r = last_row(f)
    passes = fnum(fget(r,"passes"))
    if passes is None:
        m = re.search(r'_(\d+)pass', f); passes = float(m.group(1)) if m else 0.0
    one  = fget(r, "1-pass WRONG rate", "onepass_wrong_rate")
    vote = fget(r, "vote WRONG rate", "vote_wrong_rate", "final_wrong_rate")
    cost = passes or fnum(fget(r,"cost_x"), 0.0)
    rows.append((f"vote-{int(passes)}", f, one, vote, vote, cost))
    if vote is not None: frontier.append((f"vote-{int(passes)}", float(cost), float(vote)))

# checksum
for f in sorted(glob.glob("out/checksum_cifar_tau_*.csv")):
    r = last_row(f)
    tau   = fget(r,"tau")
    one   = fget(r,"1-pass WRONG rate","onepass_wrong_rate")
    final = fget(r,"final_wrong_rate")
    cost  = fget(r,"cost_x")
    rows.append((f"checksum(τ={tau})", f, one, None, final, cost))
    if final and cost: frontier.append((f"checksum(τ={tau})", float(cost), float(final)))

# hybrid
for f in sorted(glob.glob("out/hybrid_*pass_t*.csv")):
    r = last_row(f)
    passes = fnum(fget(r,"passes"))
    if passes is None:
        m = re.search(r'_(\d+)pass', f); passes = float(m.group(1)) if m else 0.0
    rt    = fget(r,"route_thresh")
    one   = fget(r,"1-pass WRONG rate","onepass_wrong_rate")
    vote  = fget(r,"vote WRONG rate","vote_wrong_rate")
    final = fget(r,"final_wrong_rate")
    cost  = fget(r,"cost_x")
    name  = f"hybrid-{int(passes)}(t={rt})"
    rows.append((name, f, one, vote, final, cost))
    if final and cost: frontier.append((name, float(cost), float(final)))

frontier.sort(key=lambda x:x[1])
plt.figure(figsize=(9,6))
for name,cost,err in frontier:
    plt.scatter(cost,err)
    plt.annotate(name,(cost,err),textcoords="offset points",xytext=(6,4),fontsize=9)
plt.xlabel("Compute cost (× baseline forwards)")
plt.ylabel("Error rate")
plt.title("CIFAR-10 • post-logit flips (p=0.5, exp, K=3)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("out/cifar_method_frontier.png", dpi=140)

def fmt(x):
    try: return f"{float(x):.4f}"
    except: return (x or "")

head = "<tr><th>Method</th><th>CSV</th><th>1-pass</th><th>Vote</th><th>Final</th><th>Cost×</th></tr>"
trs=[]
for name,path,one,vote,final,cost in rows:
    trs.append(
        "<tr>"
        f"<td>{html.escape(str(name))}</td>"
        f"<td><code>{html.escape(path)}</code></td>"
        f"<td style='text-align:right'>{fmt(one)}</td>"
        f"<td style='text-align:right'>{fmt(vote)}</td>"
        f"<td style='text-align:right;font-weight:600'>{fmt(final)}</td>"
        f"<td style='text-align:right'>{fmt(cost)}</td>"
        "</tr>"
    )
table = "<table style='border-collapse:collapse;width:100%'>" + head + "\n" + "\n".join(trs) + "</table>"

html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>CIFAR-10 Fault Resilience Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}}
td,th{{border:1px solid #ddd;padding:6px 8px;font-size:14px}}
th{{background:#fafafa;text-align:left}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
code{{background:#f6f6f6;padding:1px 4px;border-radius:4px}}
.small{{color:#666;font-size:13px}}
</style>
<h1>CIFAR-10 Fault Resilience Report</h1>
<p class="small">Scenario: flips at post-logits, p=0.5, bit=exp, K=3. Device: MPS.</p>
<h2>Method Frontier</h2>
<p><img src="cifar_method_frontier.png" alt="frontier" style="max-width:100%;border:1px solid #eee"></p>
<h2>Key Results</h2>
{table}
"""
open("out/cifar_report.html","w").write(html_doc)
print("wrote out/cifar_report.html and out/cifar_method_frontier.png")
PY
