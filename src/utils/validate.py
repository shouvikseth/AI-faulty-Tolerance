from __future__ import annotations
import argparse, json
from typing import Tuple, Optional

try:
    import jsonschema
    from jsonschema import Draft202012Validator as _Validator
except Exception:
    # Fall back to the widely available Draft7 if 2020-12 is unavailable
    import jsonschema
    from jsonschema import Draft7Validator as _Validator


def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)


def make_validator(schema_path: str) -> _Validator:
    schema = load_schema(schema_path)
    return _Validator(schema)


def validate_file(path: str,
                  schema_path: Optional[str] = None,
                  validator: Optional[_Validator] = None) -> Tuple[int, int]:
    """
    Validate a JSONL file against a schema.

    Returns: (valid_count, invalid_count)

    Either pass a compiled `validator`, or pass `schema_path` (defaults to
    'schemas/run.schema.json' if neither is given).
    """
    if validator is None:
        if schema_path is None:
            schema_path = "schemas/run.schema.json"
        validator = make_validator(schema_path)

    valid = invalid = 0
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                invalid += 1
                continue
            errs = sorted(validator.iter_errors(obj), key=lambda e: e.path)
            if errs:
                invalid += 1
            else:
                valid += 1
    print(f"[SUMMARY] {path}: valid={valid}, invalid={invalid}")
    return valid, invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="path to JSONL file")
    ap.add_argument("--schema", dest="schema", default="schemas/run.schema.json",
                    help="path to JSON schema (default: schemas/run.schema.json)")
    args = ap.parse_args()

    validate_file(args.inp, schema_path=args.schema)


if __name__ == "__main__":
    main()
