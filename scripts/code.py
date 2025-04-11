#!/usr/bin/env python
import argparse
import os
from typing import Any, Dict

# Import the Coder class - assuming it's in a module called "coder"
# You may need to adjust this import based on your actual project structure
from tiny_scientist.coder import Coder


def create_sample_idea() -> Dict[str, Any]:
    """Create a sample experiment idea for testing."""
    return {
        "Title": "Learning Rate Impact on Model Convergence",
        "Experiment": "Investigate how different learning rates affect the convergence speed and final performance of a simple neural network on MNIST dataset.",
    }


def create_baseline_results() -> Dict[str, Any]:
    """Create sample baseline results for comparison."""
    return {
        "accuracy": {"means": 0.92, "std": 0.015},
        "training_time": {"means": 125.3, "std": 12.7},
        "convergence_epoch": {"means": 8.5, "std": 1.2},
    }


def setup_experiment_directory(base_dir: str) -> None:
    """Set up the experiment directory with necessary files."""
    os.makedirs(base_dir, exist_ok=True)

    # Create an empty experiment.py file
    with open(os.path.join(base_dir, "experiment.py"), "w") as f:
        f.write(
            """
# This is a placeholder experiment file that will be modified by the Coder
import argparse
import json
import os

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.join(os.path.dirname(__file__), args.out_dir), exist_ok=True)

    # Just return dummy results for testing
    results = {
        "accuracy": {"means": 0.94, "std": 0.01},
        "training_time": {"means": 120.5, "std": 10.2},
        "convergence_epoch": {"means": 7.8, "std": 0.9}
    }

    # Save results
    with open(os.path.join(os.path.dirname(__file__), args.out_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=2)

    return 0

if __name__ == "__main__":
    exit(main())
"""
        )

    # Create an empty notes.txt file
    with open(os.path.join(base_dir, "notes.txt"), "w") as f:
        f.write(
            "# Experiment Notes\n\nThis file will contain notes about the experiment.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trial of the Coder class")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./experiment_trial",
        help="Base directory for the experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use (e.g., llama3.1-405b, deepseek-coder-v2-0724)",
    )
    parser.add_argument(
        "--max_iters", type=int, default=2, help="Maximum iterations per experiment"
    )
    parser.add_argument(
        "--max_runs", type=int, default=2, help="Maximum experiment runs"
    )
    parser.add_argument(
        "--config_dir", type=str, default="./configs", help="Config directory"
    )
    args = parser.parse_args()

    # Set up the experiment directory
    setup_experiment_directory(args.base_dir)

    print(f"Setting up Coder with model: {args.model}")
    print(f"Base directory: {args.base_dir}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Max runs: {args.max_runs}")

    # Create the Coder instance
    coder = Coder(
        base_dir=args.base_dir,
        model=args.model,
        max_iters=args.max_iters,
        max_runs=args.max_runs,
        config_dir=args.config_dir,
    )

    # Create a sample idea and baseline results
    idea = create_sample_idea()
    baseline_results = create_baseline_results()

    print("\nStarting experiment...")
    print(f"Idea: {idea['Title']}")

    # Run the experiment
    success = coder.run(idea, baseline_results)

    if success:
        print("\nExperiment completed successfully!")
        print(f"Results and plots can be found in: {args.base_dir}")
    else:
        print("\nExperiment did not complete successfully.")
        print("Check the logs for more information.")


if __name__ == "__main__":
    main()
