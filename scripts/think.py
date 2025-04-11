#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, cast

from tiny_scientist.thinker import Thinker
from tiny_scientist.utils.input_formatter import InputFormatter
from tiny_scientist.utils.llm import AVAILABLE_LLMS, create_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate research ideas")
    parser.add_argument(
        "--base-dir", type=str, default="../ideas", help="Path to base directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=AVAILABLE_LLMS,
        help="Model to use for generating ideas",
    )
    parser.add_argument(
        "--load-existing", action="store_true", help="Load existing ideas from file"
    )
    parser.add_argument(
        "--num-ideas", type=int, default=1, help="Number of new ideas to generate"
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection iterations per idea",
    )
    parser.add_argument(
        "--check-novelty", action="store_true", help="Check novelty of generated ideas"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["semanticscholar", "openalex"],
        default="semanticscholar",
        help="Search engine for checking novelty",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Temperature for idea generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save ideas JSON (defaults to ideas.json in experiment directory)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="../configs",
        help="Path to directory containing model configurations",
    )
    parser.add_argument(
        "--initial-idea", type=str, help="Path to JSON file containing initial idea(s)"
    )
    parser.add_argument(
        "--pdf", type=str, help="Path to the PDF paper for idea generation"
    )
    return parser.parse_args()


def load_initial_idea(filepath: str) -> Dict[str, Any]:
    """Load initial idea from a JSON file."""
    try:
        with open(filepath, "r") as f:
            idea = json.load(f)
        print(f"Loaded initial idea from {filepath}")
        return cast(Dict[str, Any], idea)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading initial idea: {e}")
        raise ValueError("Valid initial idea must be provided")


def create_default_idea() -> Dict[str, Any]:
    """Create a default initial idea."""
    default_idea = {
        "Name": "baseline",
        "Title": "Baseline Implementation",
        "Experiment": "Implement baseline model with standard parameters",
        "Interestingness": 5,
        "Feasibility": 9,
        "Novelty": 3,
        "Score": 6,
    }
    return default_idea


def main() -> int:
    args = parse_args()
    formatter = InputFormatter()

    pdf_content: str = ""
    if args.pdf:
        try:
            pdf_dict = formatter.parse_paper_pdf_to_json(args.pdf)
            pdf_content = json.dumps(pdf_dict)  # Convert to string immediately
            print("Loaded PDF content for idea generation.")
        except Exception as e:
            print(f"Error loading PDF: {e}")

    try:
        # Create client and model
        client, model = create_client(args.model)

        thinker = Thinker(
            model=model,
            client=client,
            base_dir=args.base_dir,
            config_dir=args.config_dir,
            temperature=args.temperature,
            iter_num=args.num_reflections,
            tools=[],
        )

        # Get initial idea
        if args.load_existing:
            try:
                ideas_path = os.path.join(args.base_dir, "ideas.json")
                with open(ideas_path, "r") as f:
                    loaded_ideas = json.load(f)
                if loaded_ideas:
                    initial_idea = loaded_ideas[0]  # Take the first idea
                    print(f"Loaded existing idea from {ideas_path}")
                else:
                    print("No valid existing ideas found. Using default idea.")
                    initial_idea = create_default_idea()
            except (FileNotFoundError, json.JSONDecodeError):
                print("No valid existing ideas found. Using default idea.")
                initial_idea = create_default_idea()
        elif args.initial_idea:
            initial_idea = load_initial_idea(args.initial_idea)
        else:
            print("No initial idea provided. Using default idea.")
            initial_idea = create_default_idea()

        # Prepare initial idea dictionary
        initial_idea_dict = {"idea": initial_idea}

        # Generate ideas and refine them by calling run()
        final_result = thinker.run(
            initial_idea_dict,
            num_ideas=args.num_ideas,
            check_novelty=args.check_novelty,
            pdf_content=pdf_content,  # Already a string
        )

        print("\nGenerated and Refined Ideas:")
        for i, idea in enumerate(final_result["ideas"]):
            print(f"\nIdea {i+1}:")
            print(json.dumps(idea, indent=4))

        output_path = args.output or os.path.join(args.base_dir, "refined_ideas.json")
        with open(output_path, "w") as f:
            json.dump(final_result, f, indent=4)
        print(f"\nRefined ideas saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
