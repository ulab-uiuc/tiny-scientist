import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from subprocess import TimeoutExpired
from typing import Any, Dict, List, Optional, Tuple

from aider.coders import Coder as AiderCoder
from aider.io import InputOutput
from aider.models import Model
from rich import print

from .budget_checker import BudgetChecker
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
        auto_install: bool = True,
        cost_tracker: Optional[BudgetChecker] = None,
    ):
        """Initialize the ExperimentCoder with configuration and Aider setup."""
        self.client, self.model = create_client(model)
        self.output_dir = osp.abspath(output_dir)
        self.max_iters = max_iters
        self.max_runs = max_runs
        self.max_stderr_output = max_stderr_output
        self.auto_install = auto_install
        self.config = Config()
        self.cost_tracker = cost_tracker or BudgetChecker()

        # Load prompts
        self.prompts = self.config.prompt_template.coder_prompt

    def setup_aider(
        self, model: str, fnames: List[str], chat_history: Optional[str] = None
    ) -> None:
        """Setup Aider coder with the specified model."""
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Disable chat history to avoid IO recursion issues in web environment
        io = InputOutput(yes=True, chat_history_file=None)

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
    ) -> Tuple[bool, str, Optional[str]]:
        # Ensure a clean slate for every run
        print(f"[System] Cleaning experiment directory: {self.output_dir}")
        if osp.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
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
            return False, self.output_dir, "Experiment generation failed"

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

        self.cost_tracker.report()

        return True, self.output_dir, None

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
            cost_tracker=self.cost_tracker,
            task_name="_format_experiment_for_prompt",
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
                if "ModuleNotFoundError" in result.stderr and getattr(
                    self, "auto_install", True
                ):
                    missing_pkg = self._extract_missing_package(result.stderr)
                    print(
                        f"[System] Missing package detected: {missing_pkg}. Attempting to install..."
                    )

                    # Install package with proper wait and error handling
                    try:
                        install_result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", missing_pkg],
                            capture_output=True,
                            text=True,
                            timeout=300,  # 5 minutes timeout for installation
                            check=True,
                        )
                        print(f"[System] Successfully installed {missing_pkg}")
                        print(f"[System] Install output: {install_result.stdout}")

                        # Small delay to ensure the package is fully available
                        time.sleep(2)

                        print("[System] Re-running after installing dependency...")
                        return self._run_single_experiment(run_num, timeout=timeout)

                    except subprocess.TimeoutExpired:
                        print(
                            f"[System] Package installation timed out after 5 minutes for {missing_pkg}"
                        )
                        return 1, f"Package installation timeout for {missing_pkg}"

                    except subprocess.CalledProcessError as e:
                        print(f"[System] Package installation failed for {missing_pkg}")
                        print(f"[System] Installation error: {e.stderr}")
                        return (
                            1,
                            f"Package installation failed for {missing_pkg}: {e.stderr}",
                        )

                    except Exception as e:
                        print(
                            f"[System] Unexpected error during package installation: {str(e)}"
                        )
                        return 1, f"Unexpected installation error: {str(e)}"

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

    def _extract_missing_package(self, stderr: str) -> str:
        for line in stderr.splitlines():
            if "ModuleNotFoundError" in line:
                parts = line.split("'")
                if len(parts) >= 2:
                    return parts[1]
        return "unknown-package"
