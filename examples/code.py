#!/usr/bin/env python
import argparse
import os
from typing import Any, Dict

from tiny_scientist.budget_checker import BudgetChecker

# Import the Coder class - assuming it's in a module called "coder"
# You may need to adjust this import based on your actual project structure
from tiny_scientist.coder import Coder


def test_docker_availability() -> bool:
    """Test if Docker is available."""
    try:
        from tiny_scientist.tool import DockerExperimentRunner
        runner = DockerExperimentRunner()
        if runner.use_docker:
            print("✅ Docker is available and will be used")
            return True
        else:
            print("⚠️  Docker is not available, will use local execution")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False


def create_sample_idea() -> Dict[str, Any]:
    """Create a sample experiment idea for testing."""
    return {
        "Title": "Learning Rate Impact on Model Convergence",
        "Problem": "Investigate how different learning rates affect neural network training",
        "NoveltyComparison": "Comprehensive analysis of learning rate effects on convergence",
        "Approach": "Systematic evaluation of learning rates with controlled experiments",
        "Experiment": {
            "Model": "Neural Network",
            "Dataset": "MNIST",
            "Metric": "Accuracy and Convergence Speed"
        }
    }


def create_baseline_results() -> Dict[str, Any]:
    """Create sample baseline results for comparison."""
    return {
        "accuracy": {"means": 0.92, "std": 0.015},
        "training_time": {"means": 125.3, "std": 12.7},
        "convergence_epoch": {"means": 8.5, "std": 1.2},
    }


def setup_experiment_directory(output_dir: str) -> None:
    """Set up the experiment directory with necessary files."""
    os.makedirs(output_dir, exist_ok=True)

    # Create an empty experiment.py file
    with open(os.path.join(output_dir, "experiment.py"), "w") as f:
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
    with open(os.path.join(output_dir, "notes.txt"), "w") as f:
        f.write(
            "# Experiment Notes\n\nThis file will contain notes about the experiment.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trial of the Coder class")
    parser.add_argument(
        "--output_dir",
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
        "--prompt_template_dir", type=str, default="./configs", help="Config directory"
    )
    parser.add_argument(
        "--use_docker", 
        action="store_true", 
        default=True,
        help="Use Docker for experiment execution (default: True)"
    )
    parser.add_argument(
        "--auto_install", 
        action="store_true", 
        default=True,
        help="Auto-install missing packages in local mode (default: True)"
    )
    args = parser.parse_args()

    # Test Docker availability
    docker_available = test_docker_availability()
    
    # Set up the experiment directory
    setup_experiment_directory(args.output_dir)

    print(f"Setting up Coder with model: {args.model}")
    print(f"Base directory: {args.output_dir}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Max runs: {args.max_runs}")
    print(f"Docker enabled: {args.use_docker}")
    print(f"Auto-install enabled: {args.auto_install}")

    # Create the Coder instance
    coder = Coder(
        output_dir=args.output_dir,
        model=args.model,
        max_iters=args.max_iters,
        max_runs=args.max_runs,
        prompt_template_dir=args.prompt_template_dir,
        use_docker=args.use_docker,
        auto_install=args.auto_install,
        cost_tracker=BudgetChecker()
    )

    # Create a sample idea and baseline results
    idea = create_sample_idea()
    baseline_results = create_baseline_results()

    print("\nStarting experiment...")
    print(f"Idea: {idea['Title']}")

    try:
        # Run the experiment
        success, output_path, error = coder.run(idea, baseline_results)

        if success:
            print("\n✅ Experiment completed successfully!")
            print(f"📁 Results saved to: {output_path}")
            
            # Check the results
            import json
            results_file = os.path.join(output_path, "experiment_results.txt")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"📊 Experiment results: {results}")
        else:
            print(f"\n❌ Experiment failed: {error}")
            print("Check the logs for more information.")
            
    except Exception as e:
        print(f"\n💥 Error during experiment: {e}")
        
    finally:
        # Clean up Docker images if Docker was used
        if args.use_docker and docker_available:
            print("\n🧹 Cleaning up Docker images...")
            coder.cleanup_docker_images()
            print("✅ Cleanup completed")


if __name__ == "__main__":
    main()
