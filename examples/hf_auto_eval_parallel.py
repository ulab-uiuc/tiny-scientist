#!/usr/bin/env python3
"""
Parallel batch evaluation of model/dataset/metric triples

Usage:
    python hf_auto_eval_parallel.py --workers 4 --limit 20 --llm-model gpt-4o
"""

import argparse
import json
import os
import sys
from pathlib import Path
from multiprocessing import Pool, Manager
import time

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


def evaluate_single_triple(args_tuple):
    """
    Worker function to evaluate a single (model, dataset, metric) triple
    """
    model, dataset, metric, llm_model, runs, max_fixes, worker_id = args_tuple
    
    print(f"🔄 [Worker {worker_id}] Starting: {model} | {dataset} | {metric}")
    
    safe_dir = f"{model}_{dataset}_{metric}".replace("/", "_").replace("-", "_")
    out_dir = f"simple_results/{safe_dir}"
    
    try:
        # Create DockerCoder instance for this evaluation
        coder = DockerCoder(model=llm_model, output_dir=out_dir)
        success, message = coder.evaluate_model(
            model_name=model,
            dataset_name=dataset,
            metric=metric,
            max_runs=runs,
            max_fixes=max_fixes,
        )
        
        status = "✅" if success else "❌"
        print(f"{status} [Worker {worker_id}] Completed: {model} | {dataset} | {metric}")
        
        return {
            "model": model,
            "dataset": dataset,
            "metric": metric,
            "success": success,
            "message": message,
            "output_dir": out_dir if success else None,
            "worker_id": worker_id
        }
        
    except Exception as e:
        print(f"❌ [Worker {worker_id}] Exception: {model} | {dataset} | {metric} - {e}")
        return {
            "model": model,
            "dataset": dataset,
            "metric": metric,
            "success": False,
            "message": f"Exception: {e}",
            "output_dir": None,
            "worker_id": worker_id
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", default=str(Path(__file__).parent / "perfect_model_dataset_metrics.json"))
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument("--limit", type=int, default=1000, help="Evaluate only first N triples (0 = all)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    args = parser.parse_args()

    combos = load_combinations(Path(args.json_file))
    triples = list(iter_triples(combos))
    if args.limit and args.limit > 0:
        triples = triples[: args.limit]

    if not triples:
        print("No evaluable triples found.")
        return

    os.makedirs("simple_results", exist_ok=True)

    print(f"🚀 Starting parallel evaluation of {len(triples)} triples using {args.workers} workers...\n")
    
    # Prepare arguments for worker processes
    worker_args = []
    for i, (model, dataset, metric) in enumerate(triples):
        worker_id = i % args.workers + 1
        worker_args.append((model, dataset, metric, args.llm_model, args.runs, args.max_fixes, worker_id))
    
    start_time = time.time()
    
    # Run evaluations in parallel
    with Pool(processes=args.workers) as pool:
        results = pool.map(evaluate_single_triple, worker_args)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Process results
    success_count = sum(1 for r in results if r["success"])
    
    # Save summary
    summary_path = "simple_results/batch_summary_parallel.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total": len(triples),
                "success": success_count,
                "failed": len(triples) - success_count,
                "success_rate": success_count / len(triples),
                "workers": args.workers,
                "duration_seconds": duration,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n🎉 Parallel evaluation completed!")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"📊 Success: {success_count} / {len(triples)} ({success_count / len(triples) * 100:.1f}%)")
    print(f"🔄 Workers: {args.workers}")
    print(f"📁 Summary: {summary_path}")


if __name__ == "__main__":
    main() 