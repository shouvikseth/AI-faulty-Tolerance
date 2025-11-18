import argparse, json, csv
from collections import Counter, defaultdict

def load_rows(path):
    rows=[]
    with open(path) as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: rows.append(json.loads(ln))
            except: pass
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out_prefix", default="out/summary")
    args=ap.parse_args()

    rows=load_rows(args.inp)
    if not rows:
        print("no rows")
        return

    # tallies
    by_layer = defaultdict(lambda: Counter())
    by_bit   = defaultdict(lambda: Counter())  # per layer -> bitpos counts

    for r in rows:
        layer = r.get("layer_id","?")
        outc  = r.get("outcome","?")
        by_layer[layer][outc]+=1
        if "bitpos" in r and r["bitpos"] is not None:
            by_bit[layer][("bit", r["bitpos"])]+=1
            by_bit["ALL"][("bit", r["bitpos"])]+=1

    # print per-layer summary
    print("\nPer-layer outcomes:")
    print("layer_id, total, CLEAN, WRONG, DETECTED, CRASH, SDC_rate")
    layer_rows=[]
    for layer,cnts in by_layer.items():
        tot=sum(cnts.values())
        wrong=cnts.get("WRONG",0)
        sdc= wrong/tot if tot else 0.0
        line=[layer, tot, cnts.get("CLEAN",0), wrong, cnts.get("DETECTED",0), cnts.get("CRASH",0), f"{sdc:.4f}"]
        layer_rows.append(line)
        print(",".join(map(str,line)))

    # write CSVs
    with open(args.out_prefix+"_layer.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["layer_id","total","CLEAN","WRONG","DETECTED","CRASH","SDC_rate"])
        w.writerows(layer_rows)

    if by_bit:
        bit_rows=[]
        print("\nPer-bit (aggregated across trials) for ALL layers (if available):")
        print("bitpos, count")
        agg = defaultdict(int)
        for k,v in by_bit["ALL"].items():
            _,bp=k
            agg[bp]+=v
        for bp in sorted(agg):
            print(f"{bp},{agg[bp]}")
            bit_rows.append([bp, agg[bp]])
        with open(args.out_prefix+"_bitpos.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["bitpos","count"]); w.writerows(bit_rows)

    print(f"\n[wrote] {args.out_prefix}_layer.csv")
    if by_bit:
        print(f"[wrote] {args.out_prefix}_bitpos.csv")

if __name__=="__main__":
    main()
