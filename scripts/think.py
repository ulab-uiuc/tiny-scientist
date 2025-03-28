#!/usr/bin/env python3
import argparse
import json
import os

from tiny_scientist.llm import AVAILABLE_LLMS, create_client
from tiny_scientist.thinker import Thinker
from tiny_scientist.utils.loader import load_paper


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate research ideas")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="../ideas",
        help="Path to base directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=AVAILABLE_LLMS,
        help="Model to use for generating ideas"
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing ideas from file"
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=1,
        help="Number of new ideas to generate"
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection iterations per idea"
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of generated ideas"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["semanticscholar", "openalex"],
        default="semanticscholar",
        help="Search engine for checking novelty"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Temperature for idea generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save ideas JSON (defaults to ideas.json in experiment directory)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="../configs",
        help="Path to directory containing model configurations"
    )
    parser.add_argument(
        "--initial-idea",
        type=str,
        help="Path to JSON file containing initial idea(s)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to the PDF paper for idea generation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="think",
        choices=["think", "rethink", "run"],
        help="Which mode to run: 'think' to generate an idea, 'rethink' to refine an existing idea, or 'run' to create an experimental plan"
    )

    return parser.parse_args()


def load_initial_ideas(filepath: str) -> list:
    """Load initial ideas from a JSON file."""
    try:
        with open(filepath, "r") as f:
            ideas = json.load(f)
        if not isinstance(ideas, list):
            ideas = [ideas]  # Convert single idea to list
        print(f"Loaded {len(ideas)} initial ideas from {filepath}")
        return ideas
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading initial ideas: {e}")
        raise ValueError("Valid initial ideas must be provided")


def create_default_idea() -> list:
    """Create a default initial idea."""
    default_idea = [{
        "Name": "baseline",
        "Title": "Baseline Implementation",
        "Experiment": "Implement baseline model with standard parameters",
        "Interestingness": 5,
        "Feasibility": 9,
        "Novelty": 3,
        "Score": 6
    }]
    return default_idea


def main():
    args = parse_args()

    pdf_content = ""

    if args.pdf:
        try:
            pdf_content = load_paper(args.pdf)
            print("Loaded PDF content for idea generation.")
        except Exception as e:
            print(f"Error loading PDF: {e}")

    try:
        # Create client and model
        client, model = create_client(args.model)

        # Initialize thinker
        dummy_tools = []
        thinker = Thinker(
            tools=dummy_tools,
            iter_num=3,
            model=model,
            client=client,
            base_dir=args.base_dir,
            config_dir=args.config_dir,
            temperature=args.temperature,
            s2_api_key=os.getenv("S2_API_KEY")
        )

        # Get initial ideas
        if args.load_existing:
            try:
                ideas_path = os.path.join(args.base_dir, "ideas.json")
                with open(ideas_path, "r") as f:
                    ideas = json.load(f)
                initial_ideas = ideas[0] if ideas else create_default_idea()
                print(f"Loaded {len(initial_ideas)} existing ideas from {ideas_path}")
            except (FileNotFoundError, json.JSONDecodeError):
                print("No valid existing ideas found. Please provide initial ideas.")
                return 1
        elif args.initial_idea:
            initial_ideas = load_initial_ideas(args.initial_idea)
        else:
            # Use default idea if no initial ideas provided
            print("No initial ideas provided. Using default idea.")
            initial_ideas = create_default_idea()

        intent = {"idea": initial_ideas}

        result = {}
        if args.mode == "think":
            result = thinker.think(intent, check_novelty=args.check_novelty, pdf_content=pdf_content)
        elif args.mode == "rethink":
            result = thinker.rethink(intent)
        elif args.mode == "run":
            result = thinker.run(intent)

        print("\nFinal Result:")
        print(json.dumps(result, indent=4))

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
