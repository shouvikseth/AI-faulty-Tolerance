import os, json, argparse, sys
from jsonschema import Draft7Validator

def load_schema() -> dict:
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", ".."))
    schema_path = os.path.join(root, "schemas", "run.schema.json")
    with open(schema_path) as f:
        return json.load(f)

def validate_file(path: str, validator: Draft7Validator) -> tuple[int, int]:
    ok = bad = 0
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                errs = sorted(validator.iter_errors(obj), key=lambda e: e.path)
                if errs:
                    bad += 1
                    e = errs[0]
                    loc = "/".join(str(p) for p in e.path)
                    print(f"[INVALID] {path}:{i} at '{loc}': {e.message}")
                else:
                    ok += 1
            except json.JSONDecodeError as je:
                bad += 1
                print(f"[INVALID-JSON] {path}:{i}: {je}")
    return ok, bad

def main():
    ap = argparse.ArgumentParser(description="Validate JSONL logs against run.schema.json")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="One or more JSONL files")
    args = ap.parse_args()

    schema = load_schema()
    validator = Draft7Validator(schema)

    total_ok = total_bad = 0
    for p in args.inputs:
        ok, bad = validate_file(p, validator)
        total_ok += ok; total_bad += bad
        print(f"[SUMMARY] {p}: valid={ok}, invalid={bad}")
    print(f"[TOTAL] valid={total_ok}, invalid={total_bad}")
    if total_bad > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
