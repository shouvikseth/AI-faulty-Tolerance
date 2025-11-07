import sys, json
from collections import Counter

def main(path):
    c = Counter(); total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line); c[rec.get("outcome","UNK")] += 1; total += 1
    sdc_rate = (c["SDC"]/total) if total else 0.0
    print({"total": total, "counts": dict(c), "sdc_rate": sdc_rate})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.summarize out/runs.jsonl"); raise SystemExit(1)
    main(sys.argv[1])
