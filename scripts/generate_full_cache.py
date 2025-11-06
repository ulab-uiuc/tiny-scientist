#!/usr/bin/env python3
"""
Run the demo cache capture flow and immediately populate cached reviews.

This script wraps the behaviour of `generate_demo_cache.py` followed by
`review_demo_papers.py` so that a single command produces a fully populated
`frontend/demo_cache` directory (ideas, code artefacts, papers, and reviews).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts import generate_demo_cache as demo_cache
from scripts import review_demo_papers as review_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demo cache artefacts and populate cached reviews"
    )

    # Generation phase parameters (mirrors generate_demo_cache.py)
    parser.add_argument(
        "--base-url", default="http://localhost:5000", help="Backend base URL"
    )
    parser.add_argument(
        "--intent",
        default="Adaptive Prompt Decomposition for Coherent Long-Range Code Generation",
        help="Research intent to feed into the Thinker",
    )
    parser.add_argument(
        "--model", default="gpt-5", help="Model name passed to /api/configure"
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key passed to /api/configure",
    )
    parser.add_argument("--budget", type=float, default=None, help="Optional budget")
    parser.add_argument(
        "--budget-preference", default=None, help="Optional budget preference"
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=3,
        help="Number of initial ideas to request from the backend",
    )
    parser.add_argument(
        "--output-dir",
        default="frontend/demo_cache",
        help="Directory to dump cache artefacts (default: frontend/demo_cache)",
    )
    parser.add_argument(
        "--generated-root",
        default="generated",
        help="Root directory where backend writes artefacts (default: generated)",
    )
    parser.add_argument(
        "--operations-file",
        type=str,
        default=None,
        help="JSON file describing additional operations to replay",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional replacement system prompt to push before generation",
    )
    parser.add_argument(
        "--criteria-file",
        type=str,
        default=None,
        help="Optional JSON file describing evaluation criteria updates",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Request timeout (seconds) for generation calls (default: 1200)",
    )

    # Review phase parameters (mirrors review_demo_papers.py)
    parser.add_argument(
        "--review-model",
        default=None,
        help="Optional override for review /api/configure model (defaults to --model)",
    )
    parser.add_argument(
        "--review-api-key",
        default=None,
        help="Optional override for review /api/configure API key (defaults to --api-key)",
    )
    parser.add_argument(
        "--review-budget",
        type=float,
        default=None,
        help="Optional override for review budget (defaults to --budget)",
    )
    parser.add_argument(
        "--review-budget-preference",
        default=None,
        help="Optional override for review budget preference (defaults to --budget-preference)",
    )
    parser.add_argument(
        "--review-timeout",
        type=int,
        default=None,
        help="Optional timeout (seconds) for review API calls (defaults to --timeout)",
    )
    parser.add_argument(
        "--s2-api-key",
        default=None,
        help="Optional Semantic Scholar API key forwarded to review requests",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Generate the demo cache but skip the review population step",
    )

    return parser.parse_args()


def run_generation(args: argparse.Namespace) -> demo_cache.CacheContext:
    output_dir = Path(args.output_dir).resolve()
    generated_root = Path(args.generated_root).resolve()
    demo_cache.ensure_dir(output_dir)

    ctx = demo_cache.CacheContext(
        base_url=args.base_url,
        intent=args.intent,
        num_ideas=args.num_ideas,
        generated_root=generated_root,
        output_dir=output_dir,
        timeout=args.timeout,
    )

    demo_cache.configure_backend(ctx, args)
    demo_cache.capture_prompts(ctx)
    if args.system_prompt:
        demo_cache.set_system_prompt(ctx, args.system_prompt)
    if args.criteria_file:
        with open(args.criteria_file, "r", encoding="utf-8") as fh:
            criteria_map = json.load(fh)
        demo_cache.set_criteria(ctx, criteria_map)

    demo_cache.generate_initial(ctx)
    demo_cache.evaluate_now(ctx)

    plan = demo_cache.load_operations(args.operations_file)
    for step in plan.get("operations", []):
        action = step.get("action")
        if action == "generate_children":
            demo_cache.generate_children(
                ctx,
                parent_index=step.get("parent_index", 0),
                context=step.get("context", ""),
            )
        elif action == "modify":
            demo_cache.modify_idea(
                ctx,
                original_index=step.get("idea_index", 0),
                modifications=step.get("modifications", []),
                behind_index=step.get("behind_index"),
            )
        elif action == "merge":
            demo_cache.merge_ideas(
                ctx,
                index_a=step.get("idea_a_index", 0),
                index_b=step.get("idea_b_index", 1),
            )
        elif action == "evaluate":
            demo_cache.evaluate_now(ctx)
        elif action == "code":
            demo_cache.generate_code(
                ctx,
                idea_index=step.get("idea_index", 0),
                baseline_results=step.get("baseline_results", {}),
            )
        elif action == "write":
            demo_cache.generate_paper(
                ctx,
                idea_index=step.get("idea_index", 0),
                s2_api_key=step.get("s2_api_key") or args.s2_api_key,
            )
        elif action == "review":
            demo_cache.review_paper(
                ctx,
                s2_api_key=step.get("s2_api_key") or args.s2_api_key,
            )
        else:
            raise ValueError(f"Unsupported action: {action}")

    session_payload = demo_cache.build_session_payload(ctx)
    output_path = output_dir / "session.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(session_payload, fh, indent=2)
        fh.write("\n")
    ctx.log("summary", f"[record] wrote session to {output_path}")

    print(f"[generation] Demo cache captured at {output_path}")
    for channel, messages in ctx.operations_log.items():
        for message in messages:
            print(f"[generation] {channel}: {message}")

    return ctx


def run_reviews(args: argparse.Namespace) -> None:
    cache_root = Path(args.output_dir).resolve()
    session_path = cache_root / "session.json"
    papers_root = cache_root / "generated" / "papers"
    reviews_root = cache_root / "reviews"
    generated_root = Path(args.generated_root).resolve()
    generated_papers_root = generated_root / "papers"

    review_cache.ensure_dir(reviews_root)
    session_payload = review_cache.load_session(session_path)

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

    paper_pairs = review_cache.discover_papers(papers_root)
    if not paper_pairs:
        raise RuntimeError(
            f"No cached papers found under {papers_root}. "
            "Ensure the generation phase produced papers."
        )

    http_session = review_cache.requests.Session()
    review_cache.configure_backend(
        http_session,
        args.base_url,
        model=args.review_model or args.model,
        api_key=args.review_api_key or args.api_key,
        budget=(args.review_budget if args.review_budget is not None else args.budget),
        budget_preference=(args.review_budget_preference or args.budget_preference),
        timeout=args.review_timeout or args.timeout,
    )

    review_entries: List[Dict[str, Any]] = []
    for idea_id, pdf_file in paper_pairs:
        relative_pdf_path = f"/api/files/papers/{idea_id}/{pdf_file.name}"
        idea_info = idea_meta.get(idea_id, {})
        idea_index = idea_info.get("_cached_idea_index")

        print(f"[review] Reviewing {idea_id}: {relative_pdf_path}")

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

        review_payload: Dict[str, Any] = {
            "pdf_path": relative_pdf_path,
            "review": review_response.get("review"),
            "success": review_response.get("success", True),
            "message": review_response.get("message"),
        }
        if idea_index is not None:
            review_payload["_cached_idea_index"] = idea_index

        review_entries.append(review_payload)

        idea_review_dir = reviews_root / idea_id
        review_cache.ensure_dir(idea_review_dir)
        with (idea_review_dir / "review.json").open("w", encoding="utf-8") as fh:
            json.dump(review_response, fh, indent=2)
            fh.write("\n")

    session_payload["review"] = review_entries
    review_cache.save_session(session_path, session_payload)
    print(f"[review] Stored {len(review_entries)} review entries into {session_path}")


def main() -> None:
    args = parse_args()
    ctx = run_generation(args)
    if args.skip_review:
        print("[full-cache] Skipping review population as requested.")
        return
    run_reviews(args)
    print(
        f"[full-cache] Completed generation and review for {len(ctx.ideas)} ideas under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
