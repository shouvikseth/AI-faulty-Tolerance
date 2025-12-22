from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib.pyplot as plt

def load_rates(pattern: str):
    # returns (layers, pvals, wrong_rate matrix, impact_rate matrix)
    files = sorted(glob.glob(pattern))
    table = {}  # (layer, p) -> (wrong_rate, impact_rate)
    layers, pvals = set(), set()
    for f in files:
        rows = [json.loads(x) for x in open(f) if x.strip()]
        if not rows: continue
        layer = rows[0].get("layer_id","?")
        p = rows[0].get("p")
        tot = len(rows)
        wrong = sum(r.get("outcome")=="WRONG" for r in rows)
        deg   = sum(r.get("outcome")=="DEGRADED" for r in rows)
        wr = wrong/tot if tot else 0.0
        ir = (wrong+deg)/tot if tot else 0.0
        layers.add(layer); pvals.add(p)
        table[(layer,p)] = (wr, ir)
    layers = sorted(layers)
    pvals = sorted(pvals)
    W = np.zeros((len(layers), len(pvals)))
    I = np.zeros_like(W)
    for i,L in enumerate(layers):
        for j,p in enumerate(pvals):
            W[i,j], I[i,j] = table.get((L,p),(0.0,0.0))
    return layers, pvals, W, I

def fmt_p(p):
    if p is None: return "?"
    s = f"{p:.0e}" if p>=1e-2 else f"{p:.0e}"
    return s.replace("e-0","e-").replace("e+0","e+")

def plot_heat(ax, data, title, xticks, yticks):
    im = ax.imshow(data, aspect="auto")  # default colormap
    ax.set_xticks(range(len(xticks)), [fmt_p(p) for p in xticks], rotation=0)
    ax.set_yticks(range(len(yticks)), yticks)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]*100:.1f}%", ha="center", va="center", fontsize=9)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob like 'out/layers/*_p*.jsonl'")
    ap.add_argument("--out_png", required=True, help="output PNG")
    args = ap.parse_args()

    layers, pvals, W, I = load_rates(args.glob)
    if not layers or not pvals:
        raise SystemExit("no files matched / empty logs")

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
    plot_heat(ax1, W, "WRONG rate (%)", pvals, layers)
    ax2 = fig.add_subplot(212)
    plot_heat(ax2, I, "WRONG+DEGRADED rate (%)", pvals, layers)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200)
    print("[heatmap] wrote", args.out_png)

if __name__ == "__main__":
    main()
