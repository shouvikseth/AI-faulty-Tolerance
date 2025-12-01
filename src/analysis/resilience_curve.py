from __future__ import annotations
import argparse, json, glob
from collections import defaultdict
import matplotlib.pyplot as plt

def to_float(x):
    try: return float(x)
    except: return None

def load_rows(paths):
    rows=[]
    for p in paths:
        try:
            with open(p) as f:
                for ln in f:
                    ln=ln.strip()
                    if ln:
                        try: rows.append(json.loads(ln))
                        except: pass
        except FileNotFoundError:
            pass
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. out/sweep_*_p*.jsonl")
    ap.add_argument("--out_png", default="out/resilience.png")
    ap.add_argument("--metric", choices=["wrong","wrong+degraded"], default="wrong")
    args = ap.parse_args()

    rows = load_rows(glob.glob(args.glob))
    if not rows:
        print("no rows matched"); return

    group = defaultdict(list)  # (p, layer_id) -> [outcomes]
    for r in rows:
        p_val = to_float(r.get("p"))
        if p_val is None: continue
        lid = r.get("layer_id","?")
        group[(p_val, lid)].append(r.get("outcome","?"))

    series = defaultdict(list)
    for (p,lid), outs in group.items():
        tot=len(outs)
        wrong = sum(o=="WRONG" for o in outs)
        deg   = sum(o=="DEGRADED" for o in outs)
        if args.metric == "wrong": rate = wrong/tot if tot else 0.0
        else:                      rate = (wrong+deg)/tot if tot else 0.0
        series[lid].append((p, rate))

    fig=plt.figure(); ax=fig.add_subplot(111)
    for lid, pts in series.items():
        pts=sorted(pts, key=lambda x:x[0])
        ax.plot([p for p,_ in pts], [r for _,r in pts], marker="o", label=lid)
    ax.set_xscale("log")
    ax.set_xlabel("injection probability p (log scale)")
    ax.set_ylabel(f"{args.metric} rate")
    ax.set_title(f"Resilience curve by layer ({args.metric})")
    ax.legend()
    fig.tight_layout(); fig.savefig(args.out_png, dpi=140)
    print("[wrote]", args.out_png)

if __name__=="__main__":
    main()
