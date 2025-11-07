#!/usr/bin/env python3
"""
Generate static review payloads for the demo cache.

This script iterates over each cached paper inside frontend/demo_cache/generated/papers,
calls the backend /api/review endpoint to obtain a review, stores the response
under frontend/demo_cache/reviews/<idea-id>/review.json, and updates
frontend/demo_cache/session.json so DemoCacheService can replay the review results.

Usage:
    python scripts/review_demo_papers.py \
        --backend-url http://localhost:5000 \
        --demo-cache ./frontend/demo_cache
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate demo cache with paper reviews"
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:5000",
        help="Base URL of the backend providing /api/review (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--demo-cache",
        default="frontend/demo_cache",
        help="Path to the demo cache directory (default: ./frontend/demo_cache)",
    )
    parser.add_argument(
        "--generated-root",
        default="generated",
        help="Path to the backend generated directory (default: ./generated)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds for review API calls (default: 600)",
    )
    parser.add_argument(
        "--s2-api-key",
        default=None,
        help="Optional Semantic Scholar API key to forward with the review request",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name used for /api/configure (e.g. gpt-4o)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key used for /api/configure",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Optional budget to send with /api/configure",
    )
    parser.add_argument(
        "--budget-preference",
        default=None,
        help="Optional budget preference to send with /api/configure",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_session(session_path: Path) -> Dict[str, Any]:
    with session_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_session(session_path: Path, payload: Dict[str, Any]) -> None:
    with session_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def discover_papers(papers_root: Path) -> List[Tuple[str, Path]]:
    if not papers_root.exists():
        raise FileNotFoundError(f"Demo papers directory not found: {papers_root}")
    pairs: List[Tuple[str, Path]] = []
    for idea_dir in sorted(papers_root.iterdir()):
        if not idea_dir.is_dir():
            continue
        pdf_files = sorted(
            [p for p in idea_dir.iterdir() if p.suffix.lower() == ".pdf"]
        )
        if not pdf_files:
            continue
        pairs.append((idea_dir.name, pdf_files[0]))
    return pairs


def configure_backend(
    session: requests.Session,
    base_url: str,
    *,
    model: Optional[str],
    api_key: Optional[str],
    budget: Optional[float],
    budget_preference: Optional[str],
    timeout: int,
) -> None:
    if not model or not api_key:
        return

    payload: Dict[str, Any] = {"model": model, "api_key": api_key}
    if budget is not None:
        payload["budget"] = budget
    if budget_preference:
        payload["budget_preference"] = budget_preference

    response = session.post(
        f"{base_url.rstrip('/')}/api/configure",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()


def call_review(
    base_url: str,
    pdf_path: str,
    *,
    timeout: int,
    s2_api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"pdf_path": pdf_path}
    if s2_api_key:
        payload["s2_api_key"] = s2_api_key
    http = session or requests.Session()
    response = http.post(
        f"{base_url.rstrip('/')}/api/review",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.demo_cache).resolve()
    session_path = cache_root / "session.json"
    papers_root = cache_root / "generated" / "papers"
    reviews_root = cache_root / "reviews"
    generated_root = Path(args.generated_root).resolve()
    generated_papers_root = generated_root / "papers"

    ensure_dir(reviews_root)
    session_payload = load_session(session_path)

    write_entries: List[Dict[str, Any]] = session_payload.get("write") or []
    idea_meta: Dict[str, Dict[str, Any]] = {}
    for entry in write_entries:
        idea_data = entry.get("idea") or {}
        idea_id = idea_data.get("id")
        if not idea_id:
            continue
        idea_meta[idea_id] = {
            "pdf_path": entry.get("pdf_path"),
            "_cached_idea_index": entry.get("_cached_idea_index"),
        }

    paper_pairs = discover_papers(papers_root)
    if not paper_pairs:
        raise RuntimeError(
            f"No cached papers found under {papers_root}. "
            "Run generate_demo_cache.py first."
        )

    http_session = requests.Session()
    configure_backend(
        http_session,
        args.backend_url,
        model=args.model,
        api_key=args.api_key,
        budget=args.budget,
        budget_preference=args.budget_preference,
        timeout=args.timeout,
    )

    review_entries: List[Dict[str, Any]] = []
    for idea_id, pdf_file in paper_pairs:
        relative_pdf_path = f"/api/files/papers/{idea_id}/{pdf_file.name}"
        idea_info = idea_meta.get(idea_id, {})
        idea_index = idea_info.get("_cached_idea_index")

        print(f"[review] Reviewing {idea_id}: {relative_pdf_path}")

        # Ensure backend can serve the PDF via /api/files by copying it into generated folder
        backend_paper_dir = generated_papers_root / idea_id
        ensure_dir(backend_paper_dir)
        backend_pdf_path = backend_paper_dir / pdf_file.name
        if not backend_pdf_path.exists():
            shutil.copy2(pdf_file, backend_pdf_path)

        review_response = call_review(
            args.backend_url,
            relative_pdf_path,
            timeout=args.timeout,
            s2_api_key=args.s2_api_key,
            session=http_session,
        )

        review_payload = {
            "pdf_path": relative_pdf_path,
            "review": review_response.get("review"),
            "success": review_response.get("success", True),
            "message": review_response.get("message"),
        }
        if idea_index is not None:
            review_payload["_cached_idea_index"] = idea_index

        review_entries.append(review_payload)

        idea_review_dir = reviews_root / idea_id
        ensure_dir(idea_review_dir)
        with (idea_review_dir / "review.json").open("w", encoding="utf-8") as fh:
            json.dump(review_response, fh, indent=2)
            fh.write("\n")

    session_payload["review"] = review_entries
    save_session(session_path, session_payload)
    print(f"[done] Stored {len(review_entries)} review entries into {session_path}")


if __name__ == "__main__":
    main()
