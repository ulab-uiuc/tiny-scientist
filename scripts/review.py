#!/usr/bin/env python3
import argparse
import json
import os

from tiny_scientist.llm import AVAILABLE_LLMS, create_client
from tiny_scientist.reviewer import Reviewer


def parse_args():
    parser = argparse.ArgumentParser(description="Perform a paper review using the specified model.")
    parser.add_argument(
        "--paper",
        type=str,
        default="../example/attention.pdf",
        help="Path to the paper text/PDF to be reviewed, or raw text directly."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=AVAILABLE_LLMS,
        help="Model to use for reviewing."
    )
    parser.add_argument(
        "--num-fs-examples",
        type=int,
        default=0,
        help="Number of few-shot review examples to include in the prompt."
    )
    parser.add_argument(
        "--num-ensemble",
        type=int,
        default=1,
        help="Number of ensemble reviews to generate and then combine into a meta-review."
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=1,
        help="Number of reflection steps on the final review."
    )
    parser.add_argument(
        "--reviewer-type",
        type=str,
        default="neg",
        choices=["neg", "pos", "base"],
        help=(
            "Which system prompt style to use for the reviewer: "
            "'neg' = critical/negative, 'pos' = more positive, "
            "'base' = the generic system prompt."
        )
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Temperature for the LLM."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="review.json",
        help="Path to save the final review JSON."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="../configs",
        help="Path to directory containing model configurations"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Create the client and select the model
    client, model = create_client(args.model)

    # Instantiate our reviewer
    reviewer = Reviewer(
        model=model,
        client=client,
        temperature=args.temperature,
        config_dir=args.config_dir
    )

    # Load the paper text from PDF, plaintext file, or use raw text
    if os.path.isfile(args.paper):
        _, ext = os.path.splitext(args.paper)
        if ext.lower() == ".pdf":
            text = reviewer.load_paper(args.paper)
        else:
            # Assume plaintext file
            with open(args.paper, "r", encoding="utf-8") as f:
                text = f.read()
    else:
        # If the file doesn't exist, assume the user passed raw text
        text = args.paper

    # Pick which system prompt to use
    if args.reviewer_type == "neg":
        system_prompt = reviewer.reviewer_system_prompt_neg
    elif args.reviewer_type == "pos":
        system_prompt = reviewer.reviewer_system_prompt_pos
    else:
        system_prompt = reviewer.reviewer_system_prompt_base

    # Perform the review
    review = reviewer.perform_review(
        text=text,
        num_reflections=args.num_reflections,
        num_fs_examples=args.num_fs_examples,
        num_reviews_ensemble=args.num_ensemble,
        reviewer_system_prompt=system_prompt,
        return_msg_history=False
    )

    # If return_msg_history=True were used, review might be (review_obj, msg_history). Here we only expect the review.
    # Print and save
    print("\nFinal Review JSON:")
    print(json.dumps(review, indent=4))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=4)
    print(f"\nReview saved to {args.output}")

    # Print a summary of key fields in the review (similar style to "Idea Summary" in think.py)
    print("\nReview Summary:")
    summary_field = review.get("Summary", "")
    if len(summary_field) > 150:
        short_summary = summary_field[:150] + "..."
    else:
        short_summary = summary_field
    print(f"Paper Summary: {short_summary if short_summary else 'No summary provided.'}")

    overall_score = review.get("Overall", "N/A")
    print(f"Overall Score: {overall_score}")

    decision = review.get("Decision", "No decision")
    print(f"Decision: {decision}\n")


if __name__ == "__main__":
    main()
