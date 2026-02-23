import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: int = 600,
) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {body}") from e


def _http_get_bytes(url: str, timeout_s: int = 600) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _dim_key(pair: Dict[str, Any]) -> str:
    return f"{pair.get('dimensionA','')}-{pair.get('dimensionB','')}"


def _dedupe_pairs(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for p in pairs:
        k = _dim_key(p)
        if not k.strip() or k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _collect_dimension_pairs(
    backend_base: str, intent: str, want: int = 5
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for _ in range(6):
        data = _http_json(
            "POST",
            f"{backend_base}/api/suggest-dimensions",
            payload={"intent": intent},
            timeout_s=300,
        )
        pairs = data.get("dimension_pairs") or []
        if isinstance(pairs, list):
            collected.extend([p for p in pairs if isinstance(p, dict)])
        collected = _dedupe_pairs(collected)
        if len(collected) >= want:
            return collected[:want]
        time.sleep(0.5)
    return collected[:want]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_dynamic_modifications(
    dimension_pairs: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    keys: List[str] = []
    for pair in dimension_pairs:
        if not isinstance(pair, dict):
            continue
        key = _dim_key(pair).strip("-")
        if not key:
            continue
        if key in keys:
            continue
        keys.append(key)
    keys = keys[:3]

    score_mods: List[Dict[str, Any]] = []
    legacy_mods: List[Dict[str, Any]] = []

    if keys:
        for i, key in enumerate(keys):
            new_score = 15 + i * 10
            score_mods.append(
                {
                    "metric": key,
                    "previousScore": 0,
                    "newScore": new_score,
                    "change": new_score,
                }
            )
            legacy_mods.append({"metric": key, "direction": "increase"})
        return score_mods, legacy_mods

    return (
        [{"metric": "noveltyScore", "previousScore": 0, "newScore": 20, "change": 20}],
        [{"metric": "noveltyScore", "direction": "increase"}],
    )


def _local_fallback_modified_idea(
    root_idea: Dict[str, Any],
    root_id: str,
) -> Dict[str, Any]:
    base = dict(root_idea)
    title = str(base.get("Title") or base.get("Name") or "Modified Idea")
    base["id"] = f"{root_id}-X1"
    base["Name"] = str(base.get("Name") or "modified_idea") + "_modified"
    base["Title"] = f"{title} (Modified)"
    if "Description" in base and isinstance(base["Description"], str):
        base["Description"] = (
            base["Description"]
            + " This version is a modified variant for demo cache generation."
        )
    return {
        "id": base["id"],
        "title": base["Title"],
        "content": "",
        "originalData": base,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate frontend/demo_cache/cache.json and associated files from a real running backend."
    )
    parser.add_argument("--backend-base", default="http://localhost:5000")
    parser.add_argument("--intent", default="Self Improving Agents")
    parser.add_argument(
        "--model", default=os.environ.get("DEMO_CACHE_MODEL", "gpt-5-mini")
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEMO_CACHE_API_KEY"),
        help="LLM API key passed to /api/configure. You can also set DEMO_CACHE_API_KEY.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "cache.json"),
        help="Output cache JSON path.",
    )
    parser.add_argument(
        "--files-dir",
        default=str(Path(__file__).resolve().parent / "files"),
        help="Directory to store downloaded /api/files artifacts.",
    )
    args = parser.parse_args()

    backend_base = args.backend_base.rstrip("/")
    out_path = Path(args.out).resolve()
    files_root = Path(args.files_dir).resolve()

    if not args.api_key:
        raise SystemExit(
            "Missing --api-key (or DEMO_CACHE_API_KEY). Needed to call /api/configure on the real backend."
        )

    _http_json("POST", f"{backend_base}/api/clear-session", payload={}, timeout_s=60)
    _http_json(
        "POST",
        f"{backend_base}/api/configure",
        payload={"model": args.model, "api_key": args.api_key},
        timeout_s=60,
    )

    prompts = _http_json(
        "GET", f"{backend_base}/api/get-prompts", payload=None, timeout_s=60
    )

    dimension_pairs = _collect_dimension_pairs(
        backend_base=backend_base, intent=args.intent, want=5
    )

    gen_initial = _http_json(
        "POST",
        f"{backend_base}/api/generate-initial",
        payload={
            "intent": args.intent,
            "num_ideas": 1,
            "dimension_pairs": dimension_pairs,
        },
        timeout_s=600,
    )
    ideas0 = gen_initial.get("ideas") or []
    if not ideas0:
        raise RuntimeError("/api/generate-initial returned no ideas")

    root_idea = ideas0[0]
    root_original = root_idea.get("originalData") or {}
    root_id = root_idea.get("id") or root_original.get("id")
    if not root_id:
        raise RuntimeError("Missing root idea id")

    gen_child = _http_json(
        "POST",
        f"{backend_base}/api/generate-children",
        payload={
            "parent_content": root_idea.get("content", ""),
            "parent_id": root_id,
            "context": "",
        },
        timeout_s=600,
    )

    score_mods, legacy_mods = _build_dynamic_modifications(dimension_pairs)
    modify_payload_base = {
        "original_idea": root_original,
        "original_id": root_id,
        "behind_idea": (gen_child.get("ideas") or [{}])[0].get("originalData"),
        "dimension_pairs": dimension_pairs,
    }

    modify: Dict[str, Any]
    try:
        modify = _http_json(
            "POST",
            f"{backend_base}/api/modify",
            payload={**modify_payload_base, "modifications": score_mods},
            timeout_s=900,
        )
    except Exception:
        try:
            modify = _http_json(
                "POST",
                f"{backend_base}/api/modify",
                payload={**modify_payload_base, "modifications": legacy_mods},
                timeout_s=900,
            )
        except Exception:
            modify = _local_fallback_modified_idea(root_original, root_id)

    merge = _http_json(
        "POST",
        f"{backend_base}/api/merge",
        payload={
            "idea_a": root_original,
            "idea_b": (modify.get("originalData") or modify),
            "idea_a_id": root_id,
            "idea_b_id": modify.get("id"),
            "dimension_pairs": dimension_pairs,
        },
        timeout_s=600,
    )

    request_ideas: List[Dict[str, Any]] = []
    for item in [
        root_original,
        (gen_child.get("ideas") or [{}])[0].get("originalData"),
        modify.get("originalData"),
        merge.get("originalData"),
    ]:
        if isinstance(item, dict) and item:
            request_ideas.append(item)
    request_ideas = [i for i in request_ideas if isinstance(i, dict)]

    evaluate = _http_json(
        "POST",
        f"{backend_base}/api/evaluate",
        payload={
            "ideas": request_ideas,
            "intent": args.intent,
            "dimension_pairs": dimension_pairs,
            "mode": "full",
        },
        timeout_s=600,
    )

    workflow_idea = merge.get("originalData") or merge
    code = _http_json(
        "POST",
        f"{backend_base}/api/code",
        payload={"idea": {"originalData": workflow_idea}},
        timeout_s=3600,
    )

    downloaded_text: Dict[str, str] = {}
    downloaded_bin: Dict[str, str] = {}

    exp_dir = code.get("experiment_dir")
    if exp_dir:
        candidates = [
            "experiment.py",
            "experiment_results.txt",
            "notes.txt",
        ]
        for i in range(1, 6):
            candidates.append(f"run_{i}.py")
            candidates.append(f"run_{i}/final_info.json")

        for name in candidates:
            api_rel = f"{exp_dir}/{name}"
            url = f"{backend_base}/api/files/{api_rel}"
            try:
                data = _http_json("GET", url, payload=None, timeout_s=60)
                content = data.get("content")
                if isinstance(content, str):
                    local_rel = Path("experiments") / exp_dir / name
                    local_path = files_root / local_rel
                    _write_text(local_path, content)
                    downloaded_text[f"experiments/{api_rel}"] = str(
                        Path("files") / local_rel
                    ).replace(os.path.sep, "/")
            except Exception:
                continue

    write = _http_json(
        "POST",
        f"{backend_base}/api/write",
        payload={"idea": {"originalData": workflow_idea}, "experiment_dir": exp_dir},
        timeout_s=3600,
    )

    pdf_path = write.get("pdf_path")
    if isinstance(pdf_path, str) and pdf_path.startswith("/api/files/"):
        pdf_rel = pdf_path[len("/api/files/") :]
        pdf_url = f"{backend_base}{pdf_path}"
        pdf_bytes = _http_get_bytes(pdf_url, timeout_s=3600)
        local_rel = Path(pdf_rel)
        local_path = files_root / local_rel
        _write_bytes(local_path, pdf_bytes)
        downloaded_bin[pdf_rel] = str(Path("files") / local_rel).replace(
            os.path.sep, "/"
        )

    review: Optional[Dict[str, Any]] = None
    if isinstance(pdf_path, str):
        try:
            review = _http_json(
                "POST",
                f"{backend_base}/api/review",
                payload={"pdf_path": pdf_path},
                timeout_s=3600,
            )
        except Exception:
            review = None

    cache = {
        "meta": {
            "intent": args.intent,
            "created_at": _now_iso(),
            "schema_version": 1,
            "source_backend_base": backend_base,
        },
        "dimension_pairs": dimension_pairs,
        "prompts": {
            "system_prompt": prompts.get("system_prompt"),
            "criteria": (prompts.get("criteria") or {}),
            "defaults": (prompts.get("defaults") or {}),
        },
        "queues": {
            "generate_initial": [gen_initial],
            "generate_children": [gen_child],
            "evaluate": [evaluate],
            "modify": [modify],
            "merge": [merge],
            "code": [code],
            "write": [write],
            "review": [review] if review else [],
        },
        "files": {
            "text": downloaded_text,
            "binary": downloaded_bin,
        },
        "logs": [
            {
                "message": "(cached) demo_cache generated",
                "level": "info",
                "timestamp": 0,
            },
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"Wrote cache JSON: {out_path}")
    print(f"Wrote files under: {files_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
