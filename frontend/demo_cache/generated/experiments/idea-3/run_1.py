# -*- coding: utf-8 -*-
"""
HumanEval Torch Evaluation Script

Usage (example):
    python experiment.updated.py \
        --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
        --humaneval_jsonl /path/to/HumanEval.jsonl \
        --k 1 5 10 \
        --temperature 0.2 0.6 \
        --top_p 0.9 \
        --max_new_tokens 384 \
        --timeout 20 \
        --device auto

Outputs:
    predictions.jsonl  # one JSON per sampled completion with pass/fail
    scores.json        # aggregate pass@k and summary stats

Notes:
- This script expects a HumanEval-style JSONL with fields: "task_id", "prompt", "test".
- We run each (task, sample) in a sandboxed Python subprocess with timeouts.
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def load_model(model_name_or_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    return tokenizer, model


def stop_at_stop_tokens(text: str, stop_tokens: List[str]) -> str:
    end = len(text)
    for s in stop_tokens:
        idx = text.find(s)
        if idx != -1:
            end = min(end, idx)
    return text[:end]


def generate_one(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_tokens: List[str],
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            do_sample=True if temperature > 0 else False,
            temperature=max(1e-6, temperature),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)
    gen = stop_at_stop_tokens(gen, stop_tokens)
    return gen


def run_tests_in_subprocess(program_text: str, timeout_sec: int) -> Dict[str, Any]:
    """
    Write program_text to a temp file and run it with Python.
    program_text = prompt + completion + "\n" + tests
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program_text)

        try:
            start = time.time()
            # Use a clean subprocess; do not inherit unneeded env
            proc = subprocess.run(
                [sys.executable, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
                text=True,
            )
            elapsed = time.time() - start
            passed = proc.returncode == 0
            return {
                "passed": passed,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
                "elapsed_sec": elapsed,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "passed": False,
                "returncode": None,
                "stdout": (e.stdout or "")[-2000:] if hasattr(e, "stdout") else "",
                "stderr": (e.stderr or "Timeout")[-2000:],
                "elapsed_sec": timeout_sec,
                "timeout": True,
            }


def estimate_pass_at_k(num_total: int, num_correct: int, k: int) -> float:
    """
    Unbiased estimator from Chen et al. (HumanEval).
    If we sampled n completions and c are correct:
        pass@k = 1 - comb(n - c, k) / comb(n, k)  (for n >= k)
    """
    import math

    n = num_total
    c = num_correct
    if n < k:
        return float("nan")
    if c == 0:
        return 0.0
    if n == k:
        return 1.0 if c > 0 else 0.0

    # compute 1 - C(n-c, k)/C(n, k)
    def comb(a, b):
        if b < 0 or b > a:
            return 0.0
        return math.comb(a, b)

    return 1.0 - (comb(n - c, k) / comb(n, k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--humaneval_jsonl",
        type=str,
        required=True,
        help="Path to HumanEval problems (JSONL with fields: task_id, prompt, test).",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Total samples per task for estimating pass@k. Should be >= max(k).",
    )
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.2, 0.6])
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument(
        "--stop_tokens", type=str, nargs="*", default=["\n\n", "\nclass", "\ndef"]
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    assert args.num_samples >= max(args.k), "num_samples must be >= max(k)."
    device = pick_device(args.device)
    set_seed(args.seed)

    print(f"[Info] Loading model: {args.model_name_or_path} on {device}", flush=True)
    tokenizer, model = load_model(args.model_name_or_path, device=device)

    # Load HumanEval JSONL
    problems = []
    with open(args.humaneval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    scores_path = os.path.join(args.output_dir, "scores.json")

    # Write predictions incrementally
    pred_f = open(pred_path, "w", encoding="utf-8")

    aggregate = {"k_list": args.k, "temperatures": args.temperature, "summary": {}}

    for temp in args.temperature:
        print(f"[Info] Evaluating temperature={temp}", flush=True)
        total_correct = {
            k: 0 for k in args.k
        }  # for simple pass@k counting when num_samples==k; we still compute unbiased later
        per_task_correct_counts = []  # store c for each task

        for task in problems:
            task_id = task.get("task_id")
            prompt = task.get("prompt")
            tests = task.get("test")
            # Collect num_samples completions
            completions = []
            passed_flags = []

            for i in range(args.num_samples):
                comp = generate_one(
                    tokenizer,
                    model,
                    prompt,
                    args.max_new_tokens,
                    temp,
                    args.top_p,
                    args.stop_tokens,
                )
                program = prompt + comp + "\n" + tests + "\n"
                result = run_tests_in_subprocess(program, timeout_sec=args.timeout)
                passed = bool(result.get("passed", False))
                completions.append(comp)
                passed_flags.append(passed)

                rec = {
                    "task_id": task_id,
                    "sample_index": i,
                    "temperature": temp,
                    "completion": comp,
                    "passed": passed,
                    "elapsed_sec": result.get("elapsed_sec"),
                    "stderr_tail": result.get("stderr", ""),
                }
                pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pred_f.flush()

            c = sum(passed_flags)
            per_task_correct_counts.append(
                {"task_id": task_id, "correct": c, "n": args.num_samples}
            )

        # Compute unbiased pass@k over tasks
        temp_summary = {}
        for k in args.k:
            # average of per-task pass@k estimators
            vals = []
            for item in per_task_correct_counts:
                vals.append(estimate_pass_at_k(item["n"], item["correct"], k))
            # filter NaNs
            vals = [v for v in vals if isinstance(v, float) and not math.isnan(v)]
            mean_pass_at_k = (
                float(sum(vals) / max(1, len(vals))) if vals else float("nan")
            )
            temp_summary[f"pass@{k}"] = mean_pass_at_k

        temp_summary["num_tasks"] = len(per_task_correct_counts)
        temp_summary["samples_per_task"] = args.num_samples
        aggregate["summary"][str(temp)] = temp_summary
        print(f"[Info] Temp={temp} summary: {temp_summary}", flush=True)

    pred_f.close()
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"[Done] Wrote {pred_path} and {scores_path}")


if __name__ == "__main__":
    main()
