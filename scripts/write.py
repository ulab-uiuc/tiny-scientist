#!/usr/bin/env python3
import argparse
import json
import os

from tiny_scientist.utils.llm import create_client
from tiny_scientist.writer import Writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write paper.")

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to the experiment directory containing experiment details",
    )
    parser.add_argument(
        "--prompt_template_dir",
        type=str,
        default="../configs",
        help="Path to directory containing model configurations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for writing and refinement",
    )
    parser.add_argument(
        "--num-cite-rounds",
        type=int,
        default=2,
        help="Number of citation addition rounds",
    )
    parser.add_argument("--template", type=str, help="Template of the output paper")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["semanticscholar", "openalex"],
        default="semanticscholar",
        help="Search engine for citation retrieval",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save final paper PDF (defaults to experiment directory)",
    )

    return parser.parse_args()


def main() -> int:
    args: argparse.Namespace = parse_args()

    try:
        # Create LLM client and model
        client, model = create_client(args.model)

        # Initialize Writer
        writer = Writer(
            model=model,
            client=client,
            output_dir=args.experiment,
            prompt_template_dir=args.prompt_template_dir,
            template=args.template,
        )

        # idea should be import from args.experiemnt and idea.json
        with open(os.path.join(args.experiment, "idea.json"), "r") as f:
            idea = json.load(f)

        # Perform paper writing
        print("\nStarting paper write-up...")
        writer.run(
            idea=idea,
            folder_name=args.experiment,
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
