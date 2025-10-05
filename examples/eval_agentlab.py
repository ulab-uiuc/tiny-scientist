#!/usr/bin/env python3
from pathlib import Path
import json
from statistics import mean

ROOT = Path("/Users/yuhaofei/Downloads/bio_20_agentlab")

def main():
    files = sorted(ROOT.glob("*/review.json"))
    if not files:
        print(f"[WARN] no review.json under immediate subfolders of {ROOT}")
        return

    vals = []
    missing = []
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            # 典型结构：{"model":..., "source_txt":..., "timestamp":..., "result": {"Overall": 4.2, ...}}
            val = payload["result"]["Overall"]  # 这里就直接用 "Overall" 作为 key
            vals.append(float(val))
        except Exception as e:
            missing.append((fp, str(e)))

    if vals:
        print(f"[INFO] Parsed {len(vals)} Overall scores from {len(files)} files.")
        print(f"[RESULT] Average Overall = {mean(vals):.4f}")
    else:
        print("[RESULT] No Overall scores parsed.")

    if missing:
        print("\n[WARN] Failed files:")
        for fp, err in missing:
            print(f" - {fp}: {err}")

if __name__ == "__main__":
    main()
