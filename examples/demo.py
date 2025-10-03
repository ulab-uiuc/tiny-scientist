import argparse
import os

from tiny_scientist.scientist import TinyScientist
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyScientist pipeline.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/demo_3",
        help="Base output directory",
    )
    parser.add_argument(
        "--prompt_template_dir",
        type=str,
        default=None,
        help="Configuration directory with prompt YAML files",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="LLM model to use"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="acl",
        help="Paper format template (e.g. acl, iclr)",
    )
    parser.add_argument(
        "--enable_safety_check",
        type=bool,
        default=True,
        help="Enable safety check for input prompts (True/False)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1.0,
        help="Maximum USD budget for the entire run",
    )
    args = parser.parse_args()

    ideas = [
        "Adaptive Confidence-Guided Prompting for Improved Factuality in Large Language Models",
        "Adaptive Contextual Pruning: Improving Relevance and Conciseness in Long-Form Generation",
        "Adaptive Prompt Decomposition for Coherent Long-Range Code Generation",
        "Adversarial Confidence Stress Testing: Improving Uncertainty Quantification in Large Language Models",
        "Conceptual Pivot Prompting: Reducing Social Biases in Large Language Models through Analogical Reframing",
        "Confidence-Calibrated Semantic Branching: Improving Uncertainty Quantification in Large Language Models",
        "Culturally-Grounded Chain-of-Thought (CG-CoT): Enhancing LLMs' Performance on Culturally-Specific Tasks in Low-Resource Languages",
        "Differential Confidence Mapping: Enhancing Uncertainty Quantification in Large Language Models",
        "Emergent Axiom Distillation: Improving Code Generation through Paradigm-Specific Principles",
        "Linguistic Pivot Constellation: Enhancing Cross-Lingual Transfer for Low-Resource Languages and Dialects",
        "Metaphorical Concept Transposition: Enhancing Mathematical Problem Solving in Large Language Models",
        "Neuro-Symbolic Vernacular Parsing: Enhancing Language Models' Performance on Low-Resource Languages and Vernaculars",
        "Neurosymbolic API Synthesis: Improving Code Generation through Hybrid Prompting",
        "Probabilistic Proof Outline Generation: Improving Mathematical Problem Solving in Large Language Models",
        "Recursive Dialectal Expansion: Improving Large Language Models' Performance on Low-Resource and Vernacular Languages",
        "Semantic Constellation Diffraction: A Novel Prompting Technique for Privacy-Preserving Language Model Outputs",
        "Semantic Fog Injection: Enhancing Large Language Model Robustness Against Adversarial Attacks",
        "Sociolinguistic Role-Play Prompting: Enhancing Language Models' Performance in Multilingual and Low-Resource Contexts",
        "Temporal Bias Decay Simulation: Reducing Social Biases in Large Language Models through Evolutionary Prompting",
        "Temporal Dependency Unfolding: Improving Code Generation for Complex Stateful Systems",
    ]
    for idx, idea in enumerate(ideas):
        args.output_dir = f"{args.output_dir}_{idx}"

        os.makedirs(args.output_dir, exist_ok=True)

        # Instantiate TinyScientist and run pipeline
        scientist = TinyScientist(
            model=args.model,
            output_dir=args.output_dir,
            prompt_template_dir=args.prompt_template_dir,
            template=args.template,
            enable_safety_check=args.enable_safety_check,
            budget=args.budget,
        )

        idea = scientist.think(
            intent="Adaptive Confidence-Guided Prompting for Improved Factuality in Large Language Models"
        )

        with open(args.output_dir + '/idea.json', 'w') as f:
            json.dump(idea, f, indent=2)

        status, experiment_dir = scientist.code(idea=idea)

        if status is True:
            pdf_path = scientist.write(idea=idea, experiment_dir=experiment_dir)

        # Display total cost summary
        scientist.get_total_cost()

        tex_path = args.output_dir + '/latex/acl_latex.tex'
        review = scientist.review(
            tex_path=tex_path,
        )
        print(review)
        with open(args.output_dir + '/review.json', 'w') as f:
            json.dump(review, f, indent=2)

if __name__ == "__main__":
    main()
