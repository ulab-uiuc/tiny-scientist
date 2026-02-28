import argparse

import _bootstrap
from tiny_scientist import TinyScientist


def test_docker_availability() -> bool:
    """Test if Docker is available."""
    try:
        from tiny_scientist.tool_impls import DockerExperimentRunner

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


def create_formatted_idea(model: str, dataset: str, metric: str) -> dict:
    """Create a formatted idea dictionary that matches TinyScientist's expected structure."""
    return {
        "Name": f"evaluate_{model.replace('/', '_').replace('-', '_')}_{dataset.replace('/', '_').replace('-', '_')}",
        "Title": f"Evaluating {model} on {dataset} using {metric} Metric",
        "Description": f"Reproduce and evaluate the performance of the Hugging Face model {model} on the {dataset} dataset, specifically measuring the {metric} metric to establish baseline performance.",
        "Problem": f"Need to reproduce and validate the evaluation of {model} on {dataset} with focus on {metric} metric for performance verification and comparison.",
        "Importance": f"Reproducing model evaluations is crucial for scientific reproducibility and establishing reliable baselines. The {metric} metric provides key insights into model performance on {dataset}.",
        "Difficulty": "Moderate - requires proper model loading, dataset preprocessing, and evaluation setup, but uses standard HuggingFace libraries.",
        "NoveltyComparison": f"While model evaluation is standard practice, this specific reproduction of {model} on {dataset} focusing on {metric} provides valuable validation and baseline establishment.",
        "Approach": f"Load the pre-trained {model} from HuggingFace, prepare the {dataset} dataset, implement evaluation pipeline, and compute {metric} along with other relevant metrics.",
        "is_experimental": True,
        "Interestingness": 6,
        "Feasibility": 9,
        "Novelty": 4,
        "IntentAlignment": 10,
        "Score": 7,
        "Experiment": {"Model": model, "Dataset": dataset, "Metric": metric},
    }


def main():
    """
    This script uses TinyScientist to automate the process of reproducing
    a model evaluation on a given dataset for a specific task.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce a model evaluation using TinyScientist."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The Hugging Face model name (e.g., 'dslim/bert-base-NER').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The Hugging Face dataset name (e.g., 'eriktks/conll2003').",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="The specific metric to evaluate (e.g., 'F1', 'accuracy', 'BLEU', 'ROUGE', 'precision', 'recall').",
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="The model to use for TinyScientist.",
    )
    parser.add_argument(
        "--use_docker",
        action="store_true",
        default=True,
        help="Use Docker for experiment execution (default: True)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1.0,
        help="Maximum USD budget for the automated run",
    )

    args = parser.parse_args()

    # Test Docker availability
    docker_available = test_docker_availability()

    if args.use_docker and not docker_available:
        print("⚠️  Docker requested but not available, falling back to local execution")
        args.use_docker = False

    # Before running, ensure you have tiny_scientist installed:
    # pip install tiny-scientist

    # Initialize TinyScientist with the specified model and Docker configuration
    print(f"Initializing TinyScientist with model: {args.gpt_model}")
    print(f"Docker enabled: {args.use_docker}")
    scientist = TinyScientist(
        model=args.gpt_model,
        use_docker=args.use_docker,
        budget=args.budget,
        agent_sdk="claude",
    )

    # 1. Define the research intent based on user input.
    # This string is the core instruction for TinyScientist.
    intent = (
        f"I want to write a script to reproduce the evaluation of the Hugging Face model '{args.model}' "
        f"on the dataset '{args.dataset}'. I want to specifically measure the {args.metric} metric. "
        f"The script should load the model and dataset, run the evaluation, "
        f"and report the {args.metric} metric along with other relevant evaluation metrics."
    )

    print(f"🔬 Intent: {intent}")

    # Step 1: Create a formatted idea directly (skipping scientist.think)
    print("\nStep 1: Creating formatted research idea...")
    idea = create_formatted_idea(args.model, args.dataset, args.metric)
    print("✅ Research idea created.")
    print(f"📋 Idea Title: {idea['Title']}")
    print(f"📊 Target Metric: {idea['Experiment']['Metric']}")

    # Step 2: Generate and run the experiment code
    print("\nStep 2: Generating and running experiment code...")
    status, experiment_dir = scientist.code(idea=idea)

    # If the experiments run successfully, proceed to writing the paper
    if status is True:
        print(f"✅ Experiments completed successfully. Results are in: {experiment_dir}")

        # Step 3: Write a research paper based on the findings
        print("\nStep 3: Writing a research paper...")
        pdf_path = scientist.write(idea=idea, experiment_dir=experiment_dir)
        if not pdf_path:
            print("❌ Failed to write the paper.")
            return
        print(f"✅ Paper written and saved to: {pdf_path}")

        # Step 4: Review the generated paper
        print("\nStep 4: Reviewing the paper...")
        review = scientist.review(pdf_path=pdf_path)
        print("✅ Review complete.")
        print("\n--- Paper Review ---")
        print(review)
        print("--------------------")
    else:
        print(
            f"❌ Experiments failed. Check the logs in the experiment directory: {experiment_dir}"
        )


if __name__ == "__main__":
    main()
