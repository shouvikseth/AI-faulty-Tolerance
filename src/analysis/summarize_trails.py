
# analysis/summarize_trials.py
import json, argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="reports/bert_tiny_trials.jsonl")
    args = ap.parse_args()

    total = wrong = 0
    latencies = []
    with open(args.path) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            wrong += int(r["outcome"] == "WRONG")
            latencies.append(r["avg_latency_ms_per_sample"])
    wr = wrong / total if total else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"Trials: {total}, WRONG: {wrong} ({wr:.3%}), avg latency: {avg_lat:.3f} ms/sample")
if __name__ == "__main__":
    main()
