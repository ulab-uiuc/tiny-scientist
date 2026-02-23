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
    timeout_s: int = 1800,
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


def _load_cache(cache_path: Path) -> Dict[str, Any]:
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _extract_workflow_inputs(
    cache: Dict[str, Any]
) -> tuple[Dict[str, Any], Optional[str]]:
    queues = cache.get("queues") or {}
    merge_items = queues.get("merge") or []
    code_items = queues.get("code") or []

    if not merge_items:
        raise RuntimeError("cache.json has no queues.merge payload")

    merge0 = merge_items[0]
    workflow_idea = merge0.get("originalData") or merge0
    if not isinstance(workflow_idea, dict):
        raise RuntimeError("merge payload does not contain a valid idea object")

    experiment_dir = None
    if code_items and isinstance(code_items[0], dict):
        experiment_dir = code_items[0].get("experiment_dir")

    return workflow_idea, experiment_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run /api/write using existing cache and coder output."
    )
    parser.add_argument("--backend-base", default="http://localhost:5000")
    parser.add_argument(
        "--cache",
        default=str(Path(__file__).resolve().parent / "cache.json"),
    )
    parser.add_argument(
        "--model", default=os.environ.get("DEMO_CACHE_MODEL", "gpt-5-mini")
    )
    parser.add_argument("--s2-api-key", default=os.environ.get("S2_API_KEY"))
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Override experiment_dir from cache queues.code[0].experiment_dir",
    )
    args = parser.parse_args()

    backend_base = args.backend_base.rstrip("/")
    cache_path = Path(args.cache).resolve()

    if not cache_path.exists():
        raise SystemExit(f"Cache file not found: {cache_path}")

    cache = _load_cache(cache_path)
    workflow_idea, cached_experiment_dir = _extract_workflow_inputs(cache)
    experiment_dir = (
        args.experiment_dir
        if args.experiment_dir is not None
        else cached_experiment_dir
    )

    if "is_experimental" not in workflow_idea:
        workflow_idea = {**workflow_idea, "is_experimental": bool(experiment_dir)}

    payload: Dict[str, Any] = {
        "idea": {"originalData": workflow_idea},
        "model": args.model,
    }
    if experiment_dir:
        payload["experiment_dir"] = experiment_dir
    if args.s2_api_key:
        payload["s2_api_key"] = args.s2_api_key

    response = _http_json("POST", f"{backend_base}/api/write", payload=payload)

    if not response.get("success"):
        print(json.dumps(response, indent=2))
        return 1

    print("Writer rerun succeeded")
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
