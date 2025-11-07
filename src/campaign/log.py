import json, time, os

def write_run(jsonl_path, **fields):
    fields.setdefault("ts", time.time())
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fields) + "\n")
