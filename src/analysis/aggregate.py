from __future__ import annotations
import argparse, json, glob, csv
from collections import defaultdict, Counter

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
    ap.add_argument("--glob", required=True, help="e.g. out/sweep_*_p*.jsonl or a single file")
    ap.add_argument("--out_prefix", default="out/agg")
    args = ap.parse_args()

    rows = load_rows(glob.glob(args.glob))
    if not rows:
        print("no rows matched")
        return

    # per layer × p
    layer_p = defaultdict(list)
    for r in rows:
        p = to_float(r.get("p"))
        lid = r.get("layer_id","?")
        if p is None: continue
        layer_p[(lid,p)].append(r)

    with open(f"{args.out_prefix}_layer_p.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_id","p","trials","wrong","degraded","wrong_rate","impact_rate","avg_margin_drop_impacted"])
        for (lid,p), rs in sorted(layer_p.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            tot=len(rs)
            wcnt = sum(r.get("outcome")=="WRONG" for r in rs)
            dcnt = sum(r.get("outcome")=="DEGRADED" for r in rs)
            drops=[r.get("margin_drop_avg") for r in rs if r.get("outcome") in ("WRONG","DEGRADED") and isinstance(r.get("margin_drop_avg"),(int,float))]
            avgd = sum(drops)/len(drops) if drops else 0.0
            w.writerow([lid, p, tot, wcnt, dcnt, (wcnt/tot if tot else 0.0), ((wcnt+dcnt)/tot if tot else 0.0), avgd])

    # per layer bitpos histogram
    bit_hist = defaultdict(Counter)
    for r in rows:
        lid = r.get("layer_id","?")
        bp = r.get("bitpos")
        if isinstance(bp,int):
            bit_hist[lid][bp]+=1

    with open(f"{args.out_prefix}_bitpos.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_id","bitpos","count"])
        for lid, ctr in sorted(bit_hist.items()):
            for bp,count in sorted(ctr.items()):
                w.writerow([lid, bp, count])

    print("[aggregate] wrote", f"{args.out_prefix}_layer_p.csv", "and", f"{args.out_prefix}_bitpos.csv")

if __name__ == "__main__":
    main()
