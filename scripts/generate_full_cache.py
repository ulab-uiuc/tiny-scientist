#!/usr/bin/env python3
"""
Run the demo cache capture flow.

This script wraps the behaviour of `generate_demo_cache.py` so that a single
command produces a fully populated `frontend/demo_cache` directory (ideas,
code artefacts, and papers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts import generate_demo_cache as demo_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demo cache artefacts"
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
        "--model",
        default="claude-sonnet-4-5",
        help="Model name passed to /api/configure",
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
    parser.add_argument(
        "--s2-api-key",
        default=None,
        help="Optional Semantic Scholar API key for paper generation",
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


def main() -> None:
    args = parse_args()
    ctx = run_generation(args)
    print(
        f"[full-cache] Completed generation for {len(ctx.ideas)} ideas under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
