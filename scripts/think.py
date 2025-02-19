#!/usr/bin/env python3
import argparse
import os
import json
from typing import Optional

from tiny_scientist.thinker import Thinker
from tiny_scientist.llm import create_client, AVAILABLE_LLMS


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate research ideas")
    parser.add_argument(
        "--experiment",
        type=str,
        # required=True,
        default="",
        help="Path to experiment directory containing experiment.py and prompt.json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=AVAILABLE_LLMS,
        help="Model to use for generating ideas"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas"
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=20,
        help="Maximum number of ideas to generate"
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
    return parser.parse_args()


# def validate_experiment_dir(experiment_dir: str) -> None:
#     """Validate that the experiment directory has required files."""
#     if not os.path.isdir(experiment_dir):
#         raise ValueError(f"Experiment directory {experiment_dir} does not exist")
#
#     required_files = ["experiment.py", "prompt.json"]
#     for file in required_files:
#         if not os.path.isfile(os.path.join(experiment_dir, file)):
#             raise ValueError(f"Required file {file} not found in {experiment_dir}")
#
#     # Validate prompt.json format
#     with open(os.path.join(experiment_dir, "prompt.json"), "r") as f:
#         prompt_data = json.load(f)
#         if "task_description" not in prompt_data:
#             raise ValueError("prompt.json must contain 'task_description' field")


def load_or_create_seed_ideas(experiment_dir: str) -> None:
    """Create seed_ideas.json if it doesn't exist."""
    seed_path = os.path.join(experiment_dir, "seed_ideas.json")
    if not os.path.isfile(seed_path):
        default_seed = [{
            "Name": "baseline",
            "Title": "Baseline Implementation",
            "Experiment": "Implement baseline model with standard parameters",
            "Interestingness": 5,
            "Feasibility": 9,
            "Novelty": 3
        }]
        with open(seed_path, "w") as f:
            json.dump(default_seed, f, indent=4)
        print(f"Created default seed_ideas.json in {experiment_dir}")


def load_or_create_prompt_json(experiment_dir: str) -> None:
    prompt_path = os.path.join(experiment_dir, "prompt.json")
    if not os.path.isfile(prompt_path):
        default_prompt = {
            "task_description": "This is a default task description."
        }
        with open(prompt_path, "w") as f:
            json.dump(default_prompt, f, indent=4)
        print(f"Created default prompt.json in {experiment_dir}")


def main():
    args = parse_args()

    try:
        # Validate experiment directory
        load_or_create_prompt_json(args.experiment)

        # Ensure seed ideas exist
        load_or_create_seed_ideas(args.experiment)

        # Create client and model
        client, model = create_client(args.model)

        # Initialize thinker
        thinker = Thinker(
            model=model,
            client=client,
            base_dir=args.experiment,
            temperature=args.temperature,
            s2_api_key=os.getenv("S2_API_KEY")
        )

        # Generate ideas
        ideas = thinker.generate_ideas(
            skip_generation=args.skip_generation,
            max_num_generations=args.max_generations,
            num_reflections=args.num_reflections
        )

        print(f"\nGenerated {len(ideas)} ideas")

        # Check novelty if requested
        if args.check_novelty:
            print("\nChecking novelty of ideas...")
            ideas = thinker.check_idea_novelty(
                ideas=ideas,
                engine=args.engine
            )

        # Save ideas
        output_path = args.output or os.path.join(args.experiment, "ideas.json")
        with open(output_path, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"\nSaved ideas to {output_path}")

        # Print summary
        print("\nIdea Summary:")
        for i, idea in enumerate(ideas):
            novelty_status = "✓" if idea.get("novel", False) else "✗"
            if "novel" not in idea:
                novelty_status = "?"
            print(f"{i + 1}. {idea['Title']} [{novelty_status}]")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
