import argparse
import json
import os

from tiny_scientist.scientist import TinyScientist


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyScientist pipeline.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/demo",
        help="Base output directory",
    )
    parser.add_argument(
        "--prompt_template_dir",
        type=str,
        default=None,
        help="Configuration directory with prompt YAML files",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument(
        "--template",
        type=str,
        default="acl",
        help="Paper format template (e.g. acl, iclr)",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        import shutil

        shutil.rmtree(args.output_dir)
        print(f"🧹 Cleared existing directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct experiment intent and baseline result
    initial_idea = {
        "Name": "demo_project",
        "Title": "Evaluating Adaptive Step Sizes in Numerical Optimization",
        "Experiment": "Implement and compare different adaptive step size strategies (e.g., diminishing step size, line search) for optimizing a simple convex function like a 2D quadratic.",
        "Interestingness": 6,
        "Feasibility": 9,
        "Novelty": 5,
        "Score": 6,
    }
    baseline_result = {
        "experiment_name": "baseline_quadratic_optimization",
        "function": "f(x, y) = x^2 + y^2",
        "optimizer": "Gradient Descent",
        "step_size": 0.1,
        "iterations": 100,
        "metrics": {"final_function_value": 0.001, "steps_to_convergence": 85},
        "notes": "This baseline uses fixed step-size gradient descent on a quadratic bowl. Adaptive step-size methods aim to converge faster.",
    }

    with open(os.path.join(args.output_dir, "baseline_results.txt"), "w") as f:
        json.dump(baseline_result, f, indent=2)

    # Instantiate TinyScientist and run pipeline
    scientist = TinyScientist(
        model=args.model,
        output_dir=args.output_dir,
        prompt_template_dir=args.prompt_template_dir,
        template=args.template,
    )

    intent = {"idea": initial_idea}
    scientist.think(intent)
    scientist.code(baseline_result)
    scientist.write()
    scientist.review()

    print(f"\n📄 Final paper and review saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
