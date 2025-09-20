#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import json
from datetime import datetime

ROOT = Path("/Users/yuhaofei/Downloads/notBio_20_agentlab")
MODEL = "gpt-4o"
WORKERS = int(os.getenv("WORKERS", "5"))  # 并行度，改环境变量即可

def _to_jsonable(res):
    try:
        if hasattr(res, "to_dict") and callable(getattr(res, "to_dict")):
            return res.to_dict()
        if isinstance(res, (dict, list)):
            return res
        if isinstance(res, str):
            return {"review": res}
        return {"review": str(res)}
    except Exception:
        return {"review": str(res)}

def review_one(report_path: Path) -> str:
    from tiny_scientist import TinyScientist  # 在线程内导入 & 实例化
    scientist = TinyScientist(model=MODEL)
    raw_text = report_path.read_text(encoding="utf-8", errors="ignore")

    for attempt in range(3):
        try:
            res = scientist.review(paper_text=raw_text)
            payload = {
                "model": MODEL,
                "source_txt": str(report_path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "result": _to_jsonable(res),
            }
            out_path = report_path.parent / "review.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"[OK] {out_path}"
        except Exception as e:
            if attempt == 2:
                return f"[ERROR] {report_path}: {e}"
            time.sleep(1.5 * (attempt + 1))

def main():
    # 只遍历一层子目录：ROOT/*/report.txt
    report_files = sorted(ROOT.glob("*/report.txt"))
    if not report_files:
        print(f"[WARN] no report.txt under immediate subfolders of {ROOT}")
        return

    print(f"[INFO] {len(report_files)} reports, workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut2file = {ex.submit(review_one, p): p for p in report_files}
        for fut in as_completed(fut2file):
            print(fut.result())

if __name__ == "__main__":
    main()
