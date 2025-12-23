from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib.pyplot as plt

ORDER_BITS = ["any","exp","sign","signexp"]

def load_matrix(pattern: str):
    files = sorted(glob.glob(pattern))
    table = {}  # (bit, K) -> (wrong_rate, impact_rate)
    bits, ks = set(), set()
    for f in files:
        rows = [json.loads(x) for x in open(f) if x.strip()]
        if not rows: continue
        bit = None
        K = None
        # infer from filename like bitk_linear_last_<bit>_K<k>_p*.jsonl
        base = os.path.basename(f)
        parts = base.split("_")
        for i,p in enumerate(parts):
            if p in ORDER_BITS: bit = p
            if p.startswith("K"):
                try: K = int(p[1:])
                except: pass
        if bit is None or K is None: continue
        tot = len(rows)
        wrong = sum(r.get("outcome")=="WRONG" for r in rows)
        deg   = sum(r.get("outcome")=="DEGRADED" for r in rows)
        wr = wrong/tot if tot else 0.0
        ir = (wrong+deg)/tot if tot else 0.0
        bits.add(bit); ks.add(K)
        table[(bit,K)] = (wr,ir)
    bits = [b for b in ORDER_BITS if b in bits]
    ks = sorted(ks)
    W = np.zeros((len(bits), len(ks)))
    I = np.zeros_like(W)
    for i,b in enumerate(bits):
        for j,k in enumerate(ks):
            W[i,j], I[i,j] = table.get((b,k),(0.0,0.0))
    return bits, ks, W, I

def plot_heat(ax, data, title, xticks, yticks):
    im = ax.imshow(data, aspect="auto")  # default colormap
    ax.set_xticks(range(len(xticks)), [str(x) for x in xticks])
    ax.set_yticks(range(len(yticks)), yticks)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]*100:.1f}%", ha="center", va="center", fontsize=9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob like 'out/bitk/bitk_linear_last_*_p0.3.jsonl'")
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    bits, ks, W, I = load_matrix(args.glob)
    if not bits or not ks: raise SystemExit("no files matched / empty logs")

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(211)
    plot_heat(ax1, W, "WRONG rate (%)", ks, bits)
    ax2 = fig.add_subplot(212)
    plot_heat(ax2, I, "WRONG+DEGRADED rate (%)", ks, bits)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200)
    print("[bit×K heatmap] wrote", args.out_png)

if __name__ == "__main__":
    main()
