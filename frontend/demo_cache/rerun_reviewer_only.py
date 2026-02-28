import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: int = 3600,
) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {body}") from e


def _write_cache_snapshot(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _extract_pdf_path(cache: Dict[str, Any]) -> str:
    queues = cache.get("queues") or {}
    write_items = queues.get("write") or []
    if not isinstance(write_items, list) or not write_items:
        raise RuntimeError("cache.json is missing queues.write[0]")
    first = write_items[0]
    if not isinstance(first, dict):
        raise RuntimeError("queues.write[0] is not an object")
    pdf_path = first.get("pdf_path")
    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise RuntimeError("queues.write[0].pdf_path is missing")
    return pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call real backend /api/review and save result into frontend/demo_cache/cache.json"
    )
    parser.add_argument("--backend-base", default="http://localhost:5000")
    parser.add_argument(
        "--cache",
        default=str(Path(__file__).resolve().parent / "cache.json"),
    )
    parser.add_argument(
        "--pdf-path",
        default=None,
        help="Override pdf_path (otherwise taken from queues.write[0].pdf_path)",
    )
    parser.add_argument(
        "--s2-api-key",
        default=os.environ.get("S2_API_KEY"),
        help="Optional Semantic Scholar API key",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_CACHE_MODEL"),
        help="Optional model name (falls back to backend default if omitted)",
    )
    args = parser.parse_args()

    backend_base = args.backend_base.rstrip("/")
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise SystemExit(f"Cache file not found: {cache_path}")

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    pdf_path = args.pdf_path or _extract_pdf_path(cache)

    payload: Dict[str, Any] = {"pdf_path": pdf_path}
    if args.s2_api_key:
        payload["s2_api_key"] = args.s2_api_key
    if args.model:
        payload["model"] = args.model

    result = _http_json("POST", f"{backend_base}/api/review", payload=payload)
    if not isinstance(result, dict):
        raise RuntimeError("/api/review returned non-object JSON")
    if result.get("error"):
        raise RuntimeError(str(result.get("error")))
    if "review" not in result:
        raise RuntimeError("/api/review response missing 'review' field")

    cache.setdefault("queues", {})["review"] = [result]
    meta = cache.setdefault("meta", {})
    meta["status"] = meta.get("status") or "ok"
    meta["error"] = None
    _write_cache_snapshot(cache_path, cache)

    print("Saved review into cache.json")
    print(f"cache: {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
