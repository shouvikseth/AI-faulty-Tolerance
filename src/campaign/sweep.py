from __future__ import annotations
import argparse, os, subprocess, tempfile, yaml, sys, re

def sanitize_for_filename(s: str) -> str:
    s = s.replace("+", "").replace("-", "m").replace(".", "p")
    return re.sub(r"[^0-9a-zA-Z_]", "", s)

def run_once(plan_path, p_override, out_path, shard, shards, resume):
    with open(plan_path) as f:
        cfg = yaml.safe_load(f)

    try:
        p_val = float(p_override)
    except ValueError:
        raise SystemExit(f"Invalid p value: {p_override}")

    # override probability and output path
    cfg.setdefault("inject", {})
    cfg["inject"]["p"] = p_val
    cfg.setdefault("logging", {})
    cfg["logging"]["path"] = out_path
    # friendly plan name
    base_name = cfg.get("name", "plan")
    cfg["name"] = f"{base_name}_p{p_val:.0e}"

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_plan = tmp.name

    cmd = [
        sys.executable, "-m", "src.campaign.orchestrator",
        "--plan", tmp_plan, "--shard", str(shard), "--shards", str(shards),
        "--resume", str(resume),
    ]
    print("[sweep]", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    finally:
        os.unlink(tmp_plan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--p_list", required=True, help="comma-separated, e.g. 1e-2,5e-2,1e-1,3e-1,1")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0)
    args = ap.parse_args()

    plist = [x.strip() for x in args.p_list.split(",") if x.strip()]
    for p in plist:
        try:
            p_val = float(p); label = f"{p_val:.0e}"
        except ValueError:
            label = p
        fname = sanitize_for_filename(label)
        out_path = f"{args.out_prefix}_p{fname}.jsonl"
        run_once(args.plan, p, out_path, args.shard, args.shards, args.resume)

if __name__ == "__main__":
    main()
