from __future__ import annotations
import argparse, glob, csv
import matplotlib.pyplot as plt

def load_rows(pattern):
    rows=[]
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rd=csv.DictReader(fh)
            for r in rd: rows.append((float(r["tau"]), float(r["final_wrong_rate"]), float(r["overhead_forwards"])))
    rows.sort(key=lambda x:x[0])
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. 'out/checksum_sweep/tau_*.csv'")
    ap.add_argument("--out_png", required=True)
    args=ap.parse_args()

    rows=load_rows(args.glob)
    if not rows: raise SystemExit("no rows matched")

    taus=[r[0] for r in rows]
    err =[r[1] for r in rows]
    cost=[r[2] for r in rows]

    fig=plt.figure(figsize=(8,5))
    ax=fig.add_subplot(111)
    ax.plot(taus, err, marker='o', label='final wrong rate')
    ax.plot(taus, cost, marker='o', label='forward cost (×)')
    ax.set_xlabel('tau (cosine threshold)')
    ax.set_title('Checksum trade-off')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200)
    print("[tradeoff] wrote", args.out_png)

if __name__=="__main__":
    main()
