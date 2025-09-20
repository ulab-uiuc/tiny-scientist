#!/usr/bin/env python3
from pathlib import Path
import json
from statistics import mean

ROOT = Path("/Users/yuhaofei/Downloads/noBio_withtool")  # same ROOT
KEY = "Overall"  # score key inside payload["result"]
breakpoint()

def main():
    files = sorted(ROOT.glob("*/latex/review.json"))
    if not files:
        print(f"[WARN] no review.json under {ROOT}/*/latex/")
        return

    vals, missing = [], []
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            # support either {"result": {"Overall": x}} or top-level {"Overall": x}
            val = (payload.get("result", payload))[KEY]
            vals.append(float(val))
        except Exception as e:
            missing.append((fp, str(e)))

    if vals:
        print(f"[INFO] Parsed {len(vals)} {KEY} scores from {len(files)} files.")
        print(f"[RESULT] Average {KEY} = {mean(vals):.4f}")
    else:
        print(f"[RESULT] No {KEY} scores parsed.")

    if missing:
        print("\n[WARN] Failed files:")
        for fp, err in missing:
            print(f" - {fp}: {err}")

if __name__ == "__main__":
    main()
