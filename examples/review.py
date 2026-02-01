#!/usr/bin/env python3
import argparse
import json
import os.path as osp
from typing import Any, List

from tiny_scientist.reviewer import Reviewer
from tiny_scientist.utils.llm import AVAILABLE_LLMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform a paper review using the specified model."
    )
    parser.add_argument(
        "--paper",
        type=str,
        default="../example/attention.pdf",
        help="Path to the paper text/PDF to be reviewed, or raw text directly.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=AVAILABLE_LLMS,
        help="Model to use for reviewing.",
    )
    parser.add_argument(
        "--reviews-num",
        type=int,
        default=3,
        help="Number of independent reviews to generate (default: 3).",
    )
    parser.add_argument(
        "--reflection-num",
        type=int,
        default=2,
        help="Number of re_review (reflection) iterations per review (default: 2).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.75, help="Temperature for the LLM."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="review.json",
        help="Path to save the final review JSON.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="../configs",
        help="Path to directory containing model configurations.",
    )
    parser.add_argument(
        "--prompt-template-dir",
        type=str,
        default=None,
        help="Path to directory containing prompt templates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dummy_tools: List[Any] = []
    reviewer = Reviewer(
        tools=dummy_tools,
        num_reviews=args.reviews_num,
        num_reflections=args.reflection_num,
        model=args.model,
        temperature=args.temperature,
        prompt_template_dir=args.prompt_template_dir,
    )

    final_review = reviewer.run(args.paper)

    # Print and save the final meta-review.
    print("\nFinal Review JSON:")
    print(json.dumps(final_review, indent=4))

    output_path = osp.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_review, f, indent=4)
    print(f"\nReview saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
