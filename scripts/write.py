#!/usr/bin/env python3
import argparse
import json
import os

from tiny_scientist.coder import Coder  # Ensure Coder is properly implemented
from tiny_scientist.llm import AVAILABLE_LLMS, create_client
from tiny_scientist.writer import Writer

os.environ['OPENAI_API_KEY'] = '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'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write paper.")

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to the experiment directory containing experiment details"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        choices=AVAILABLE_LLMS,
        help="Model to use for writing and refinement"
    )
    parser.add_argument(
        "--num-cite-rounds",
        type=int,
        default=2,
        help="Number of citation addition rounds"
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Template of the output paper"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["semanticscholar", "openalex"],
        default="semanticscholar",
        help="Search engine for citation retrieval"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save final paper PDF (defaults to experiment directory)"
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
            base_dir=args.experiment,
            template=args.template,
            config_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
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

        # Save output PDF
        output_path = args.output or os.path.join(args.experiment, "Generated_Paper.pdf")
        print(f"\nFinal paper saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
