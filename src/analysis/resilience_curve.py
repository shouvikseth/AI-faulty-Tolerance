# src/analysis/resilience_curve.py
import argparse, json, glob
from collections import defaultdict
import matplotlib.pyplot as plt

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_rows(paths):
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rows.append(json.loads(ln))
                    except:
                        pass
        except FileNotFoundError:
            pass
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. out/sweep_mnist_tensor_p*.jsonl")
    ap.add_argument("--out_png", default="out/resilience.png")
    args = ap.parse_args()

    rows = load_rows(glob.glob(args.glob))
    if not rows:
        print("no rows matched")
        return

    # group by (p, layer_id)
    group = defaultdict(list)
    for r in rows:
        p_raw = r.get("p")
        p_val = to_float(p_raw)
        if p_val is None:
            continue
        lid = r.get("layer_id", "?")
        outc = r.get("outcome", "?")
        group[(p_val, lid)].append(outc)

    if not group:
        print("no usable rows with 'p' found")
        return

    # compute SDC rate = WRONG / total
    series = defaultdict(list)  # layer_id -> list of (p, rate)
    for (p, lid), outs in group.items():
        tot = len(outs)
        wrong = sum(1 for o in outs if o == "WRONG")
        rate = (wrong / tot) if tot else 0.0
        series[lid].append((p, rate))

    # plot (each layer as one line), sorted by p
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lid, pts in series.items():
        pts = sorted(pts, key=lambda x: x[0])
        ax.plot([p for p, _ in pts], [r for _, r in pts], marker="o", label=lid)
    ax.set_xscale("log")
    ax.set_xlabel("injection probability p (log scale)")
    ax.set_ylabel("SDC rate (WRONG/total)")
    ax.set_title("Resilience curve by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    print("[wrote]", args.out_png)

if __name__ == "__main__":
    main()
