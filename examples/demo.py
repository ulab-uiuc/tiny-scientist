import argparse
import json
import os

from torch import cross

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
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
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
    args = parser.parse_args()

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

    # Instantiate TinyScientist and run pipeline
    scientist = TinyScientist(
        model=args.model,
        output_dir=args.output_dir,
        prompt_template_dir=args.prompt_template_dir,
        template=args.template,
        enable_safety_check=args.enable_safety_check,
    )

    '''
    idea = scientist.think(
        intent="Evaluating Adaptive Step Sizes in Numerical Optimization"
    )

    if isinstance(idea, list):
        idea = idea[0]
    '''

    idea = {
    'Name': 'linguistic_pivot_constellation',
    'Title': 'Enhancing Cross-Lingual Transfer for Low-Resource Languages and Dialects',
    'Problem': 'Current cross-lingual transfer methods often fail to adequately support low-resource languages due to a lack of training data and the over-reliance on high-resource language structures, which can lead to ineffective model performance and neglect of dialectal nuances.',
    'Importance': 'Addressing the challenges faced by low-resource languages is crucial for global linguistic diversity, and enhancing cross-lingual transfer mechanisms can promote inclusivity and accessibility in NLP applications, as cited in recent discussions on language preservation.',
    'Difficulty': 'Challenges include sourcing adequate linguistic data for low-resource languages, developing robust models that can generalize across languages and dialects, and evaluating model performance in a context-sensitive manner. Existing studies highlight the complexities of this task.',
    'NoveltyComparison': "This approach distinguishes itself by proposing the 'Linguistic Pivot Constellation' framework, which leverages a constellation of linguistic features across multiple languages to enhance transfer learning, contrasting with traditional methods that typically focus on direct bilingual mappings.",
    'Approach': 'The methodology involves creating a linguistic feature database from both high-resource and low-resource languages, employing techniques such as transfer learning, multilingual embeddings, and unsupervised learning to facilitate cross-lingual understanding and generation.',
    'Interestingness': 9,
    "is_experimental": False,
    'Feasibility': 7,
    'Novelty': 8,
    'IntentAlignment': 10,
    'Score': 9,
    'Experiment': {
        'Model': {
            'Architecture': 'A multi-layer transformer architecture that incorporates attention mechanisms to weigh linguistic features across a constellation of languages, integrating multilingual embeddings from sources such as mBERT or XLM-R.',
            'Features': 'Utilizes linguistic features such as syntax, semantics, and morphology derived from a linguistic feature database.'
        },
        'Dataset': {
            'Description': 'A curated dataset comprising low-resource languages (e.g., Quechua, Xhosa) and their dialects, supplemented with high-resource language data (e.g., English, Spanish) to create cross-lingual mappings.',
            'Source': 'Data sourced from resources like OPUS, Common Crawl, and existing linguistic corpora.'
        },
        'Metric': {
            'Evaluation Metric': 'Utilizes BLEU and F1 scores for translation tasks, along with a contextual understanding score to evaluate performance on dialect-specific tasks.',
            'Contextual Score': 'Developed from human evaluations to gauge the understanding of dialectal nuances.'
        }
    },
    'ExperimentTable': '| Component           | Specification                                                                                                                                                                         | Justification / Rationale                                                                                                                                                                                                                             | Status |\n|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|\n| Model               | A multi-layer transformer model with attention mechanisms tailored to learn from multiple languages simultaneously. It will integrate multilingual embeddings from mBERT or XLM-R.     | Transformer models have proven effective in NLP tasks (Vaswani et al., 2017). Utilizing attention mechanisms allows the model to focus on relevant linguistic features, which is crucial for effectively transferring knowledge across languages (Devlin et al., 2019). |        |\n| Dataset             | A collection of low-resource languages (Quechua, Xhosa), enriched with high-resource language data for comparative analysis. Data will be sourced from OPUS and Common Crawl.         | The choice of low-resource languages is crucial as they are often neglected in NLP research (Joshi et al., 2020). High-resource data serves to provide a comparative framework against which the low-resource performance can be evaluated (Ruder, 2019).            |        |\n| Baselines           | Compare against existing cross-lingual transfer models such as multilingual BERT (mBERT) and XLM-R to establish benchmarks for performance.                                          | Prior studies have established baseline models that can be used for comparative analysis (Pires et al., 2019). This helps in understanding the performance improvements made by the Linguistic Pivot Constellation framework.                              |        |\n| Evaluation Metric   | Leverage BLEU and F1 scores for translation tasks, alongside a contextual understanding score developed from human evaluations to assess dialectal comprehension.                      | BLEU and F1 scores are standard for translation quality assessment, while human evaluations on contextual understanding capture nuanced differences in dialect use, aligning with the goal of improving model sensitivity to linguistic diversity (Liu et al., 2020). |        |'
}
    pdf_path = scientist.write(idea=idea, experiment_dir=args.output_dir)
    scientist.review(pdf_path=pdf_path)


if __name__ == "__main__":
    main()
