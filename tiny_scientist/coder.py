import json
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
from typing import Any, Dict, List, Optional, Tuple

from aider.coders import Coder as AiderCoder
from aider.io import InputOutput
from aider.models import Model
from rich import print

from .configs import Config
from .utils.llm import create_client, get_response_from_llm


class Coder:
    def __init__(
        self,
        model: str,
        output_dir: str,
        max_iters: int = 4,
        max_runs: int = 5,
        max_stderr_output: int = 1500,
        prompt_template_dir: Optional[str] = None,
        chat_history: Optional[str] = None,
    ):
        """Initialize the ExperimentCoder with configuration and Aider setup."""
        self.client, self.model = create_client(model)
        self.output_dir = osp.abspath(output_dir)
        self.max_iters = max_iters
        self.max_runs = max_runs
        self.max_stderr_output = max_stderr_output
        self.config = Config()

        # Load prompts
        self.prompts = self.config.prompt_template.coder_prompt

    def setup_aider(
        self, model: str, fnames: List[str], chat_history: Optional[str] = None
    ) -> None:
        """Setup Aider coder with the specified model."""
        io = InputOutput(
            yes=True, chat_history_file=chat_history or f"{self.output_dir}/aider.txt"
        )

        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "deepseek-chat":
            main_model = Model("deepseek/deepseek-chat")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)

        self.coder = AiderCoder.create(
            main_model=main_model,
            fnames=fnames,  # Will be set per operation
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

    def run(
        self, idea: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = {}
    ) -> Tuple[bool, str]:
        fnames = [
            osp.join(self.output_dir, "experiment.py"),
            osp.join(self.output_dir, "notes.txt"),
        ]

        self.setup_aider(self.model, fnames)

        # Run experiments
        success = self._run_experiment_loop(idea, baseline_results)

        if not success:
            # Even if failed, save an empty result file to avoid breaking writer
            save_path = osp.join(self.output_dir, "experiment_results.txt")
            with open(save_path, "w") as f:
                json.dump({}, f, indent=2)
            print(
                f"[System] No experiments succeeded, but wrote empty result to {save_path}"
            )
            return False, self.output_dir

        self._update_notes()

        result_summary = {}
        for run_num in range(1, self.max_runs + 1):
            run_dir = osp.join(self.output_dir, f"run_{run_num}")
            result_path = osp.join(run_dir, "final_info.json")
            if osp.exists(result_path):
                with open(result_path, "r") as f:
                    result_summary[f"run_{run_num}"] = json.load(f)

        # Save combined results
        save_path = osp.join(self.output_dir, "experiment_results.txt")
        with open(save_path, "w") as f:
            json.dump(result_summary, f, indent=2)

        print(f"[System] All experiment results saved to {save_path}")

        return True, self.output_dir

    def _format_experiment_for_prompt(
        self, exp: Dict[str, str]
    ) -> Tuple[str, str, str]:
        llm_prompt = self.prompts.experiment_keyword_prompt.format(
            model=exp["Model"], dataset=exp["Dataset"], metric=exp["Metric"]
        )

        llm_output, _ = get_response_from_llm(
            msg=llm_prompt,
            client=self.client,
            model=self.model,
            system_message="You are helping an AI agent extract implementation-relevant key information from an experiment description.",
        )

        try:
            # Clean and parse JSON block
            llm_output_clean = llm_output.strip().strip("`").strip("json").strip()
            keyword_info = json.loads(llm_output_clean)
        except json.JSONDecodeError:
            print("[System] Failed to parse LLM keyword JSON.")
            keyword_info = {
                "model": [],
                "dataset": [],
                "metric": [],
            }

        model_kw = ", ".join(keyword_info.get("model", []))
        dataset_kw = ", ".join(keyword_info.get("dataset", []))
        metric_kw = ", ".join(keyword_info.get("metric", []))

        return model_kw, dataset_kw, metric_kw

    def _summarize_to_bullets(self, paragraph: str) -> str:
        # Simple sentence-splitting bullet conversion
        lines = paragraph.strip().split(". ")
        return "\n".join(f"- {line.strip().rstrip('.')}" for line in lines if line)

    def _run_experiment_loop(
        self, idea: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = {}
    ) -> bool:
        """Run the experiment loop with multiple iterations if needed."""
        current_iter = 0
        run_time = 1

        # Initial prompt
        model, dataset, metric = self._format_experiment_for_prompt(idea["Experiment"])

        next_prompt = self.prompts.experiment_prompt.format(
            title=idea["Title"],
            problem=idea["Problem"],
            novelty=idea["NoveltyComparison"],
            approach=idea["Approach"],
            model=model,
            dataset=dataset,
            metric=metric,
            max_runs=self.max_runs,
            baseline_results=baseline_results,
        )

        while run_time < self.max_runs + 1:
            if current_iter >= self.max_iters:
                print("Max iterations reached")
                return False

            coder_out = self.coder.run(next_prompt)
            exp_path = osp.join(self.output_dir, "experiment.py")

            if "ALL_COMPLETED" in coder_out:
                return True

            if osp.exists(exp_path):
                with open(exp_path) as f:
                    content = f.read()
                    if "..." in content:
                        print("[System] Placeholder '...' detected. Attempting fix.")
                        self.coder.run(
                            "Please replace all placeholders (`...`) in experiment.py with complete runnable code."
                        )

            return_code, message = self._run_single_experiment(run_time)

            if return_code == 0:
                run_time += 1
                current_iter = 0
                next_prompt = message
            else:
                print("[System] Experiment run failed. Attempting fix with Aider...")
                next_prompt = self.prompts.experiment_error_prompt.format(
                    message=message,
                    Title=idea["Title"],
                    Experiment=idea["Experiment"],
                    run_time=run_time,
                    max_runs=self.max_runs,
                )

                current_iter += 1

        return current_iter < self.max_iters

    def _run_single_experiment(
        self, run_num: int, timeout: int = 7200
    ) -> Tuple[int, str]:
        """Run a single experiment iteration."""

        shutil.copy(
            osp.join(self.output_dir, "experiment.py"),
            osp.join(self.output_dir, f"run_{run_num}.py"),
        )

        # Run experiment
        command = ["python", "experiment.py", f"--out_dir=run_{run_num}"]

        try:
            result = subprocess.run(
                command,
                cwd=self.output_dir,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )

            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if result.returncode != 0:
                print(f"Run {run_num} failed with return code {result.returncode}")
                self._cleanup_failed_run(run_num)

                stderr_output = result.stderr
                if len(stderr_output) > self.max_stderr_output:
                    stderr_output = "..." + stderr_output[-self.max_stderr_output :]

                return 1, stderr_output

            # Load and format results
            with open(
                osp.join(self.output_dir, f"run_{run_num}", "final_info.json"), "r"
            ) as f:
                results = json.load(f)

            if isinstance(results, dict):
                results = {
                    k: v["means"] if isinstance(v, dict) and "means" in v else v
                    for k, v in results.items()
                }
            elif isinstance(results, list):
                results = {f"entry_{i+1}": entry for i, entry in enumerate(results)}

            results = {
                k: v["means"] if isinstance(v, dict) and "means" in v else v
                for k, v in results.items()
            }

            return 0, self.prompts.experiment_success_prompt.format(
                run_num=run_num, results=results, next_run=run_num + 1
            )

        except TimeoutExpired:
            print(f"Run {run_num} timed out after {timeout} seconds")
            self._cleanup_failed_run(run_num)
            return 1, self.prompts.experiment_timeout_prompt.format(timeout=timeout)

    def _update_notes(self) -> None:
        """Update notes.txt with plot descriptions."""
        # Set files for this operation
        self.coder.fnames = [osp.join(self.output_dir, "notes.txt")]
        self.coder.run(self.prompts.notes_prompt)

    def _cleanup_failed_run(self, run_num: int) -> None:
        """Clean up files from a failed run."""
        run_dir = osp.join(self.output_dir, f"run_{run_num}")
        if osp.exists(run_dir):
            shutil.rmtree(run_dir)
