import json
import os

import _bootstrap
from tiny_scientist import TinyScientist

scientist = TinyScientist(
    model="gpt-5.2",
    budget=5.0,
    agent_sdk="openai",
    thinker_iter_num=1,
)
output_dir = scientist.output_dir
os.makedirs(output_dir, exist_ok=True)

# Step 1: Generate a json-format research idea
idea = scientist.think(
    intent="Benchmarking adaptive step size strategies using a convex quadratic optimization function",
    save_json=True,
)
if not idea:
    raise SystemExit("[demo] Idea generation failed; aborting demo.")

# Step 2: Run experiments (you can provide baseline_results if available)
status, experiment_dir = scientist.code(idea=idea)

# if the experiments run successfully
if status is True:
    # Step 3: Write a paper
    pdf_path = scientist.write(
        idea=idea,
        experiment_dir=experiment_dir,
        save_json=True,
    )

    # Step 4: Review the paper
    review = scientist.review(pdf_path=pdf_path)
    review_path = os.path.join(output_dir, "review.json")
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)
    print(f"[demo] Saved review to: {review_path}")
