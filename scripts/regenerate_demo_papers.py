#!/usr/bin/env python3
"""
Refresh cached demo papers (writer output) and reviews using existing experiments.

This script assumes code artefacts in the demo cache have already been updated
manually (e.g., experiment.py, run_*.py, experiment_results.txt). It copies those
artefacts into the backend `generated` directory, replays the writer for the
selected ideas, and then regenerates cached reviews so the frontend demo stays
in sync.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

from scripts import generate_demo_cache as demo_cache
from scripts import review_demo_papers as review_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run cached writer + review steps for demo ideas"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:5000",
        help="Backend base URL (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model name passed to /api/configure for writer (default: gpt-5)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key passed to /api/configure for writer (default: empty string)",
    )
    parser.add_argument("--budget", type=float, default=None, help="Optional budget")
    parser.add_argument(
        "--budget-preference",
        default=None,
        help="Optional budget preference value",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Request timeout (seconds) for writer and review calls (default: 1200)",
    )
    parser.add_argument(
        "--output-dir",
        default="frontend/demo_cache",
        help="Demo cache directory containing ideas/ and generated/ (default: frontend/demo_cache)",
    )
    parser.add_argument(
        "--generated-root",
        default="generated",
        help="Backend generated directory (default: generated)",
    )
    parser.add_argument(
        "--ideas",
        nargs="*",
        default=None,
        help="Optional list of idea IDs to refresh (defaults to all ideas with experiments)",
    )
    parser.add_argument(
        "--s2-api-key",
        default=None,
        help="Optional Semantic Scholar API key forwarded to writer/reviewer",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Re-run writer only and skip review regeneration",
    )
    parser.add_argument(
        "--review-model",
        default=None,
        help="Override model for review /api/configure (defaults to --model)",
    )
    parser.add_argument(
        "--review-api-key",
        default=None,
        help="Override API key for review /api/configure (defaults to --api-key)",
    )
    parser.add_argument(
        "--review-budget",
        type=float,
        default=None,
        help="Override budget for review requests (defaults to --budget)",
    )
    parser.add_argument(
        "--review-budget-preference",
        default=None,
        help="Override budget preference for reviews (defaults to --budget-preference)",
    )
    parser.add_argument(
        "--review-timeout",
        type=int,
        default=None,
        help="Override timeout for review requests (defaults to --timeout)",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def discover_idea_payloads(
    output_dir: Path, selected: Optional[Set[str]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return (idea_id, payload) tuples for all ideas that have experiment artefacts."""
    idea_root = output_dir / "ideas"
    experiments_root = output_dir / "generated" / "experiments"
    if not experiments_root.exists():
        raise FileNotFoundError(
            f"Demo experiments directory missing: {experiments_root}"
        )

    idea_payloads: List[Tuple[str, Dict[str, Any]]] = []
    for exp_dir in sorted(p for p in experiments_root.iterdir() if p.is_dir()):
        idea_id = exp_dir.name
        if selected and idea_id not in selected:
            continue
        idea_json = idea_root / idea_id / "idea.json"
        if not idea_json.exists():
            raise FileNotFoundError(
                f"Idea payload missing for {idea_id}: {idea_json} (expected to exist)"
            )
        payload = load_json(idea_json)
        payload.setdefault("id", idea_id)
        idea_payloads.append((idea_id, payload))
    if selected:
        missing = selected - {idea_id for idea_id, _ in idea_payloads}
        if missing:
            raise ValueError(
                f"Requested idea IDs not found with experiments: {', '.join(sorted(missing))}"
            )
    return idea_payloads


def copy_experiments_for_backend(
    demo_experiments: Path, backend_root: Path, idea_ids: Iterable[str]
) -> None:
    backend_experiments = backend_root / "experiments"
    backend_experiments.mkdir(parents=True, exist_ok=True)
    for idea_id in idea_ids:
        src = demo_experiments / idea_id
        if not src.exists():
            raise FileNotFoundError(f"Expected experiment folder missing: {src}")
        dst = backend_experiments / idea_id
        shutil.copytree(src, dst, dirs_exist_ok=True)


def rel_experiment_path(idea_id: str) -> str:
    return str(Path("experiments") / idea_id)


def merge_logs(
    existing: Optional[Dict[str, List[str]]], updates: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    if existing:
        for channel, messages in existing.items():
            merged[channel] = list(messages)
    for channel, messages in updates.items():
        merged.setdefault(channel, []).extend(messages)
    return merged


def extract_idea_id(entry: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit idea metadata if present
    idea = entry.get("idea")
    if isinstance(idea, dict):
        idea_id = idea.get("id")
        if isinstance(idea_id, str):
            return idea_id
    pdf_path = entry.get("pdf_path") or entry.get("local_pdf_path")
    if isinstance(pdf_path, str) and "papers/" in pdf_path:
        parts = [p for p in pdf_path.split("/") if p]
        try:
            idx = parts.index("papers")
            return parts[idx + 1]
        except (ValueError, IndexError):
            return None
    return None


def merge_entries(
    previous: Sequence[Dict[str, Any]],
    replacements: Dict[str, Dict[str, Any]],
    processed_ids: Set[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in previous:
        idea_id = extract_idea_id(entry)
        if not idea_id or idea_id in processed_ids:
            continue
        merged[idea_id] = entry
    for idea_id, entry in replacements.items():
        merged[idea_id] = entry
    return [merged[key] for key in sorted(merged.keys())]


def regenerate_reviews(
    args: argparse.Namespace,
    session_payload: Dict[str, Any],
    idea_ids: Set[str],
    output_dir: Path,
    generated_root: Path,
) -> Dict[str, Dict[str, Any]]:
    if not idea_ids:
        return {}

    papers_root = output_dir / "generated" / "papers"
    reviews_root = output_dir / "reviews"
    review_cache.ensure_dir(reviews_root)

    generated_papers_root = generated_root / "papers"
    generated_papers_root.mkdir(parents=True, exist_ok=True)

    write_entries: List[Dict[str, Any]] = session_payload.get("write") or []
    idea_meta: Dict[str, Dict[str, Any]] = {}
    for entry in write_entries:
        idea_id = extract_idea_id(entry)
        if idea_id:
            idea_meta[idea_id] = {
                "pdf_path": entry.get("pdf_path"),
                "_cached_idea_index": entry.get("_cached_idea_index"),
            }

    paper_pairs = review_cache.discover_papers(papers_root)
    filtered_pairs = [
        (idea_id, pdf_path) for idea_id, pdf_path in paper_pairs if idea_id in idea_ids
    ]
    if not filtered_pairs:
        raise RuntimeError(
            "No cached papers found for the requested ideas; run the writer step first."
        )

    http_session = requests.Session()
    review_cache.configure_backend(
        http_session,
        args.base_url,
        model=args.review_model or args.model,
        api_key=args.review_api_key or args.api_key,
        budget=args.review_budget if args.review_budget is not None else args.budget,
        budget_preference=args.review_budget_preference or args.budget_preference,
        timeout=args.review_timeout or args.timeout,
    )

    updated_entries: Dict[str, Dict[str, Any]] = {}
    for idea_id, pdf_file in filtered_pairs:
        relative_pdf_path = f"/api/files/papers/{idea_id}/{pdf_file.name}"
        idea_info = idea_meta.get(idea_id, {})
        idea_index = idea_info.get("_cached_idea_index")

        backend_paper_dir = generated_papers_root / idea_id
        review_cache.ensure_dir(backend_paper_dir)
        backend_pdf_path = backend_paper_dir / pdf_file.name
        if not backend_pdf_path.exists():
            shutil.copy2(pdf_file, backend_pdf_path)

        review_response = review_cache.call_review(
            args.base_url,
            relative_pdf_path,
            timeout=args.review_timeout or args.timeout,
            s2_api_key=args.s2_api_key,
            session=http_session,
        )

        payload: Dict[str, Any] = {
            "pdf_path": relative_pdf_path,
            "review": review_response.get("review"),
            "success": review_response.get("success", True),
            "message": review_response.get("message"),
        }
        if idea_index is not None:
            payload["_cached_idea_index"] = idea_index

        updated_entries[idea_id] = payload

        idea_review_dir = reviews_root / idea_id
        review_cache.ensure_dir(idea_review_dir)
        with (idea_review_dir / "review.json").open("w", encoding="utf-8") as fh:
            json.dump(review_response, fh, indent=2)
            fh.write("\n")

    return updated_entries


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    generated_root = Path(args.generated_root).resolve()
    session_path = output_dir / "session.json"

    session_payload = load_json(session_path)
    intent = session_payload.get("intent") or "Demo cache regeneration"

    idea_filter = set(args.ideas) if args.ideas else None
    idea_payloads = discover_idea_payloads(output_dir, idea_filter)
    if not idea_payloads:
        raise RuntimeError("No ideas found to process - nothing to do.")

    idea_ids = [idea_id for idea_id, _ in idea_payloads]
    copy_experiments_for_backend(
        output_dir / "generated" / "experiments", generated_root, idea_ids
    )

    ctx = demo_cache.CacheContext(
        base_url=args.base_url,
        intent=intent,
        num_ideas=len(idea_payloads),
        generated_root=generated_root,
        output_dir=output_dir,
        timeout=args.timeout,
    )
    demo_cache.configure_backend(ctx, args)

    # Register ideas in the same order we discovered them
    for idea_id, payload in idea_payloads:
        demo_cache.ensure_dir(ctx.ideas_root / idea_id)
        ctx.register_ideas([payload], preassigned_ids=[idea_id])

    new_write_entries: Dict[str, Dict[str, Any]] = {}
    for idx, record in enumerate(ctx.ideas):
        idea_id = record.id
        # Ensure backend can locate the experiment artefacts
        experiment_rel_path = rel_experiment_path(idea_id)
        ctx.last_real_experiment_dir = experiment_rel_path
        ctx.last_demo_experiment_dir = None
        ctx.last_real_pdf_path = None
        ctx.last_demo_pdf_path = None

        print(f"[writer] Regenerating paper for {idea_id} using {experiment_rel_path}")
        before = len(ctx.write_payloads)
        demo_cache.generate_paper(ctx, idea_index=idx, s2_api_key=args.s2_api_key)
        if len(ctx.write_payloads) != before + 1:
            raise RuntimeError(f"Writer did not return an entry for {idea_id}")
        new_write_entries[idea_id] = ctx.write_payloads[-1]

    processed_ids = set(new_write_entries.keys())
    session_payload["write"] = merge_entries(
        session_payload.get("write") or [], new_write_entries, processed_ids
    )
    session_payload["logs"] = merge_logs(
        session_payload.get("logs"), ctx.operations_log
    )
    save_json(session_path, session_payload)
    print(f"[writer] Updated session write entries for: {', '.join(sorted(processed_ids))}")

    if not args.skip_review:
        refreshed_reviews = regenerate_reviews(
            args, session_payload, processed_ids, output_dir, generated_root
        )
        session_payload["review"] = merge_entries(
            session_payload.get("review") or [], refreshed_reviews, processed_ids
        )
        save_json(session_path, session_payload)
        print(
            f"[review] Refreshed reviews for: {', '.join(sorted(refreshed_reviews.keys()))}"
        )
    else:
        print("[review] Skipped as requested.")


if __name__ == "__main__":
    main()
