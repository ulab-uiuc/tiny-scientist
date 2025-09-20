#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import json
from datetime import datetime

ROOT = Path("/Users/yuhaofei/Downloads/noBio_withtool")
MODEL = "gpt-4o"
WORKERS = int(os.getenv("WORKERS", "5"))  # 并行度，改环境变量即可

def _to_jsonable(res):
    # res 可能是 str / dict / 有 .to_dict()
    try:
        if hasattr(res, "to_dict") and callable(getattr(res, "to_dict")):
            return res.to_dict()
        if isinstance(res, (dict, list)):
            return res
        if isinstance(res, str):
            return {"review": res}
        # 兜底：尝试转成字符串
        return {"review": str(res)}
    except Exception:
        return {"review": str(res)}

def review_one(tex_path: Path) -> str:
    from tiny_scientist import TinyScientist  # 在线程内导入 & 实例化
    scientist = TinyScientist(model=MODEL)
    raw_tex = tex_path.read_text(encoding="utf-8", errors="ignore")

    for attempt in range(3):
        try:
            res = scientist.review(paper_text=raw_tex)
            payload = {
                "model": MODEL,
                "source_tex": str(tex_path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "result": _to_jsonable(res),
            }
            out_path = tex_path.parent / "review.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"[OK] {out_path}"
        except Exception as e:
            if attempt == 2:
                return f"[ERROR] {tex_path}: {e}"
            time.sleep(1.5 * (attempt + 1))

def main():
    tex_files = sorted(ROOT.glob("*/latex/acl_latex.tex"))
    if not tex_files:
        print(f"[WARN] no acl_latex.tex under {ROOT}")
        return

    print(f"[INFO] {len(tex_files)} papers, workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut2file = {ex.submit(review_one, p): p for p in tex_files}
        for fut in as_completed(fut2file):
            print(fut.result())

if __name__ == "__main__":
    main()
