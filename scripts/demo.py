import argparse
import json
import os

from tiny_scientist.scientist import TinyScientist


def categorize_domain(domain: str) -> str:
    """
    Categorize a research domain into either 'computational' or 'physical'.
    
    Args:
        domain: The research domain (e.g., Biology, Physics, Information Science)
        
    Returns:
        str: 'computational' or 'physical'
    """
    # Computational domains
    computational_domains = ["Information Science"]
    
    # Physical domains
    physical_domains = [
        "Biology", "Physics", "Chemistry", "Material Science", "Medical Science"
    ]
    
    if domain in computational_domains:
        return "computational"
    elif domain in physical_domains:
        return "physical"
    else:
        # Default to computational if domain is not specified or not recognized
        return "computational"


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
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        choices=["", "Biology", "Physics", "Information Science", "Chemistry", "Material Science", "Medical Science"],
        help="Research domain (e.g., Biology, Physics, Information Science)",
    )
    parser.add_argument(
        "--test_all_domains",
        action="store_true",
        help="Test idea generation for all domains",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=3,
        help="Number of groups for multi-group discussion",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        import shutil

        shutil.rmtree(args.output_dir)
        print(f"🧹 Cleared existing directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct experiment intent and baseline result
    baseline_results = {
        "experiment_name": "baseline_quadratic_optimization",
        "function": "f(x, y) = x^2 + y^2",
        "optimizer": "Gradient Descent",
        "step_size": 0.1,
        "iterations": 100,
        "metrics": {"final_function_value": 0.001, "steps_to_convergence": 85},
        "notes": "This baseline uses fixed step-size gradient descent on a quadratic bowl. Adaptive step-size methods aim to converge faster.",
    }

    with open(os.path.join(args.output_dir, "baseline_results.txt"), "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Instantiate TinyScientist
    scientist = TinyScientist(
        model=args.model,
        prompt_template_dir=args.prompt_template_dir,
        template=args.template,
    )

    # Test all domains if requested
    if args.test_all_domains:
        domains = ["Biology", "Physics", "Information Science", "Chemistry", "Material Science", "Medical Science"]
        intents = {
            "Biology": "Understanding cellular response to environmental stress",
            "Physics": "Investigating quantum entanglement in novel materials",
            "Information Science": "Developing efficient algorithms for large-scale data processing",
            "Chemistry": "Synthesizing novel organic compounds with specific properties",
            "Material Science": "Creating self-healing materials for industrial applications",
            "Medical Science": "Studying the effectiveness of a new drug delivery system"
        }
        
        for domain in domains:
            print(f"\n\n{'='*50}")
            print(f"TESTING DOMAIN: {domain}")
            print(f"{'='*50}\n")
            
            intent = intents[domain]
            print(f"Intent: {intent}")
            
            # Categorize the domain
            experiment_type = categorize_domain(domain)
            print(f"Experiment type: {experiment_type}")
            
            # Create domain-specific output directory
            domain_dir = os.path.join(args.output_dir, domain.lower().replace(" ", "_"))
            os.makedirs(domain_dir, exist_ok=True)
            
            # Generate idea for this domain
            idea = scientist.think(
                intent=intent, 
                domain=domain, 
                experiment_type=experiment_type
            )
            
            # Save the idea to a file
            with open(os.path.join(domain_dir, "idea.json"), "w") as f:
                json.dump(idea, f, indent=2)
            
            print(f"\nIdea saved to {os.path.join(domain_dir, 'idea.json')}")
    else:
        # Single domain test
        intent = "Evaluating Adaptive Step Sizes in Numerical Optimization"
        if args.domain:
            print(f"\nTesting domain: {args.domain}")
            intent = {
                "Biology": "Understanding cellular response to environmental stress",
                "Physics": "Investigating quantum entanglement in novel materials",
                "Information Science": "Developing efficient algorithms for large-scale data processing",
                "Chemistry": "Synthesizing novel organic compounds with specific properties",
                "Material Science": "Creating self-healing materials for industrial applications",
                "Medical Science": "Studying the effectiveness of a new drug delivery system"
            }.get(args.domain, intent)
        
        # Categorize the domain
        experiment_type = categorize_domain(args.domain)
        print(f"Experiment type: {experiment_type}")
        
        idea = scientist.think(
            intent=intent, 
            domain=args.domain, 
            experiment_type=experiment_type
        )
        
        # Save the idea to a file
        with open(os.path.join(args.output_dir, "idea.json"), "w") as f:
            json.dump(idea, f, indent=2)
        
        print(f"\nIdea saved to {os.path.join(args.output_dir, 'idea.json')}")

    # Commented out code/write/review sections
    """

    status, experiment_dir = scientist.code(
        idea=idea, baseline_results=baseline_results
    )
    if status is False:
        return
    pdf_path = scientist.write(idea=idea, experiment_dir=experiment_dir)
    scientist.review(pdf_path=pdf_path)
    """


if __name__ == "__main__":
    main()
