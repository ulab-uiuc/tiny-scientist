#!/usr/bin/env python3
"""
Batch‑evaluate all model/dataset/metric triples from perfect_model_dataset_metrics.json

Usage:
    python hf_eval_all.py --llm-model gpt-4o --runs 1 --max-fixes 5 --limit 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing tiny_scientist
sys.path.insert(0, str(Path(__file__).parent.parent))
from tiny_scientist.coder_docker import DockerCoder


def load_combinations(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("results", [])


def iter_triples(combos):
    """
    Yield (model_id, dataset_id, metric_name) for every simple metric.
    Skips metrics whose value is a dict (nested/complex).
    """
    for combo in combos:
        model = combo.get("model_id")
        dataset = combo.get("dataset_id")
        metrics = combo.get("metrics", {}) or {}
        for metric_name, value in metrics.items():
            if value is None or isinstance(value, dict):
                continue
            yield model, dataset, metric_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", default=str(Path(__file__).parent / "perfect_model_dataset_metrics.json"))
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-fixes", type=int, default=15)
    parser.add_argument("--limit", type=int, default=30, help="Evaluate only first N triples (0 = all)")
    args = parser.parse_args()

    combos = load_combinations(Path(args.json_file))
    triples = list(iter_triples(combos))
    if args.limit and args.limit > 0:
        triples = triples[: args.limit]

    if not triples:
        print("No evaluable triples found.")
        return

    os.makedirs("simple_results", exist_ok=True)

    summary = []
    success_count = 0

    print(f"Starting evaluation of {len(triples)} triples...\n")
    for i, (model, dataset, metric) in enumerate(triples, 1):
        print("="*50)
        print(f"[{i}/{len(triples)}] {model} | {dataset} | {metric}")
        print("="*50)
        safe_dir = f"{model}_{dataset}_{metric}".replace("/", "_").replace("-", "_")
        out_dir = f"simple_results/{safe_dir}"

        try:
            # Create a new DockerCoder instance for each evaluation with its own output directory
            coder = DockerCoder(model=args.llm_model, output_dir=out_dir)
            success, message = coder.evaluate_model(
                model_name=model,
                dataset_name=dataset,
                metric=metric,
                max_runs=args.runs,
                max_fixes=args.max_fixes,
            )
        except Exception as e:
            success, message = False, f"Exception: {e}"

        summary.append(
            {
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "success": success,
                "message": message,
                "output_dir": out_dir if success else None,
            }
        )
        if success:
            success_count += 1

    # Save summary
    summary_path = "simple_results/batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total": len(triples),
                "success": success_count,
                "failed": len(triples) - success_count,
                "success_rate": success_count / len(triples),
                "results": summary,
            },
            f,
            indent=2,
        )

    print("\nDone.")
    print(f"Success: {success_count} / {len(triples)} "
          f"({success_count / len(triples) * 100:.1f}%)")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
