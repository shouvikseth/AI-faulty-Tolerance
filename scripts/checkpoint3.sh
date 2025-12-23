#!/usr/bin/env bash
set -euo pipefail
python - << 'PY'
import os, csv, base64, matplotlib.pyplot as plt
def exists(p): 
    try: return os.path.exists(p)
    except: return False
FILES = {
  "vote-3":  "out/vote_post_p05_3pass.csv",
  "vote-5":  "out/vote_post_p05.csv",
  "checksum-OR": "out/checksum_post_mix_p05.csv",
  "mixk3":   "out/checksum_post_mixk3_p05.csv",
  "mixk3_d1": "out/checksum_post_mixk3_digits1_p05.csv",
  "mixk2_loose": "out/checksum_post_mixk2_loose_p05.csv",
  "mixk2_d1": "out/checksum_post_mixk2_digits1_p05.csv",
}
baseline_clean = 0.0191
def read_vote(path):
    r = list(csv.DictReader(open(path)))[-1]
    wrong_1p  = float(r["wrong_single_rate"])
    wrong_fin = float(r["wrong_vote_rate"])
    cost = 5.0 if "5" in os.path.basename(path) and "3pass" not in path else 3.0
    return wrong_1p, wrong_fin, cost
def read_checksum(path):
    r = list(csv.DictReader(open(path)))[-1]
    fin   = float(r["final_wrong_rate"])
    det   = float(r.get("detected_rate", 0.0))
    cost  = float(r.get("overhead_forwards", 2 + det))
    return fin, cost, r
# build plot and html
pts = [("clean", 1.0, baseline_clean)]
table = []
os.makedirs("out", exist_ok=True)
if exists(FILES["vote-3"]):
    one, fin, cost = read_vote(FILES["vote-3"]); pts.append(("vote-3", cost, fin)); table.append(("vote-3", fin, cost, "", one))
if exists(FILES["vote-5"]):
    one, fin, cost = read_vote(FILES["vote-5"]); pts.append(("vote-5", cost, fin)); table.append(("vote-5", fin, cost, "", one))
for key in ["checksum-OR","mixk3","mixk3_d1","mixk2_loose","mixk2_d1"]:
    p = FILES[key]
    if exists(p):
        fin, cost, r = read_checksum(p)
        table.append((key, fin, cost, r.get("detected_rate",""), r.get("wrong_single_rate","")))
        pts.append((key, cost, fin))
table.append(("clean", baseline_clean, 1.0, "", ""))
plt.figure(figsize=(7.6,4.8))
for name, cost, err in pts:
    plt.scatter(cost, err, s=65); plt.text(cost+0.03, err+0.001, name, fontsize=8)
plt.xlabel("Compute cost (× forwards)"); plt.ylabel("Error rate")
plt.title("Cost–Error Trade-off @ p=0.5 (logits exp flips, K=3 per image)")
plt.grid(True, alpha=0.3); plt.tight_layout()
png = "out/tradeoff_p05.png"; plt.savefig(png, dpi=160)
enc = lambda p: "data:image/png;base64,"+base64.b64encode(open(p,"rb").read()).decode("ascii")
html = """<!doctype html><meta charset="utf-8"><title>Checkpoint 3 — Final</title>
<h1>Checkpoint 3 — Final Comparison</h1>
<p><b>Fault model:</b> post-hook on logits, exponent-bit flips, K=3 per image, p=0.5</p>
<table border="1" cellspacing="0" cellpadding="6"><tr>
<th>Method</th><th>Error</th><th>Cost</th><th>Detected</th><th>1-pass wrong</th></tr>"""
for nm, err, cost, det, one in sorted(table, key=lambda x:(x[2],x[1])):
    html += f"<tr><td>{nm}</td><td>{float(err):.4f}</td><td>{float(cost):.2f}</td><td>{det}</td><td>{one}</td></tr>"
html += "</table><img src=\"%s\" style=\"max-width:900px;display:block;margin-top:12px;\">" % enc(png)
open("out/checkpoint3_final.html","w").write(html)
print("[wrote]", png); print("[wrote] out/checkpoint3_final.html")
PY
zip -q -r out/checkpoint3_bundle.zip out/tradeoff_p05.png out/checkpoint3_final.html out/vote_post_p05*.csv out/checksum_post_*.csv || true
echo "[bundle] out/checkpoint3_bundle.zip"
