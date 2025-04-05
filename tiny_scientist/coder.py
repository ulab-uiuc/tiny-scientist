import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
from typing import Any, Dict, List, Optional, Tuple

import yaml
from aider.coders import Coder as AiderCoder
from aider.io import InputOutput
from aider.models import Model

from .tool import CodeSearchTool


class Coder:
    def __init__(
        self,
        base_dir: str,
        model: str,
        chat_history: Optional[str] = None,
        max_iters: int = 4,
        max_runs: int = 5,
        max_stderr_output: int = 1500
    ):
        """Initialize the ExperimentCoder with configuration and Aider setup."""
        self.model = model
        self.base_dir = osp.abspath(base_dir)
        self.max_iters = max_iters
        self.max_runs = max_runs
        self.max_stderr_output = max_stderr_output
        self.searcher = CodeSearchTool()

        # Load prompts
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "coder_prompt.yaml")
        with open(yaml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def setup_aider(self,
                    model: str,
                    fnames: List[str],
                    chat_history: Optional[str] = None) -> None:
        """Setup Aider coder with the specified model."""
        io = InputOutput(
            yes=True,
            chat_history_file=chat_history or f"{self.base_dir}/aider.txt"
        )

        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
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

    def perform_experiments(
        self,
        idea: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> bool:
        """Run the complete experiment workflow."""
        # Set files for this operation

        fnames = [
            osp.join(self.base_dir, "experiment.py"),
            osp.join(self.base_dir, "notes.txt")
        ]

        self.setup_aider(self.model, fnames)

        # Run experiments
        success = self._run_experiment_loop(idea, baseline_results)

        if not success:
            return False

        # # Create plots
        # success = self._create_plots()
        # if not success:
        #     print("[System] Plotting failed. Please check the logs.")
        #     return False

        # Update notes
        self._update_notes()

        result_summary = {}
        for run_num in range(1, self.max_runs + 1):
            run_dir = osp.join(self.base_dir, f"run_{run_num}")
            result_path = osp.join(run_dir, "final_info.json")
            if osp.exists(result_path):
                with open(result_path, "r") as f:
                    result_summary[f"run_{run_num}"] = json.load(f)

        # Save combined results
        save_path = osp.join(self.base_dir, "experiment_results.txt")
        with open(save_path, "w") as f:
            json.dump(result_summary, f, indent=2)

        print(f"[System] All experiment results saved to {save_path}")

        return True

    def _run_experiment_loop(
        self,
        idea: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> bool:
        """Run the experiment loop with multiple iterations if needed."""
        current_iter = 0
        run_time = 1

        # Initial prompt
        next_prompt = self.prompts["experiment_prompt"].format(
            title=idea["Title"],
            idea=idea["Experiment"],
            max_runs=self.max_runs,
            baseline_results=baseline_results,
        )

        while run_time < self.max_runs + 1:
            if current_iter >= self.max_iters:
                print("Max iterations reached")
                return False

            coder_out = self.coder.run(next_prompt)
            exp_path = osp.join(self.base_dir, "experiment.py")

            if "ALL_COMPLETED" in coder_out:
                return True

            if osp.exists(exp_path):
                with open(exp_path) as f:
                    content = f.read()
                    if "..." in content:
                        print("[System] Placeholder '...' detected. Attempting fix.")
                        self.coder.run("Please replace all placeholders (`...`) in experiment.py with complete runnable code.")

            return_code, message = self._run_single_experiment(run_time)

            if return_code == 0:
                run_time += 1
                current_iter = 0
                next_prompt = message
            else:
                print("[System] Experiment run failed. Attempting fix with Aider...")
                next_prompt = self.prompts["experiment_error_prompt"].format(
                    message=message,
                    Title=idea["Title"],
                    Experiment=idea["Experiment"],
                    run_time=run_time,
                    max_runs=self.max_runs,
                )

                current_iter += 1

        return current_iter < self.max_iters

    def _run_single_experiment(
        self,
        run_num: int,
        timeout: int = 7200
    ) -> Tuple[int, str]:
        """Run a single experiment iteration."""

        shutil.copy(
            osp.join(self.base_dir, "experiment.py"),
            osp.join(self.base_dir, f"run_{run_num}.py"),
        )

        # Run experiment
        command = ["python", "experiment.py", f"--out_dir=run_{run_num}"]

        try:
            result = subprocess.run(
                command,
                cwd=self.base_dir,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )

            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if result.returncode != 0:
                print(f"Run {run_num} failed with return code {result.returncode}")
                self._cleanup_failed_run(run_num)

                stderr_output = result.stderr
                if len(stderr_output) > self.max_stderr_output:
                    stderr_output = "..." + stderr_output[-self.max_stderr_output:]

                return 1, stderr_output

            # Load and format results
            with open(osp.join(self.base_dir, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)

            results = {
                k: v["means"] if isinstance(v, dict) and "means" in v else v
                for k, v in results.items()
            }

            return 0, self.prompts["experiment_success_prompt"].format(
                run_num=run_num,
                results=results,
                next_run=run_num + 1
            )

        except TimeoutExpired:
            print(f"Run {run_num} timed out after {timeout} seconds")
            self._cleanup_failed_run(run_num)
            return 1, self.prompts["experiment_timeout_prompt"].format(
                timeout=timeout
            )

    def _create_plots(self, timeout: int = 600) -> bool:
        """Create plots from experimental results."""
        # Set files for this operation
        self.coder.fnames = [
            osp.join(self.base_dir, "plot.py")
        ]

        current_iter = 0
        next_prompt = self.prompts["plot_initial_prompt"]

        while True:
            self.coder.run(next_prompt)
            return_code, next_prompt = self._run_plotting(timeout)

            current_iter += 1
            if return_code == 0 or current_iter >= self.max_iters:
                break

        return return_code == 0

    def _run_plotting(self, timeout: int) -> Tuple[int, str]:
        """Run the plotting script."""
        command = ["python", "plot.py"]

        try:
            result = subprocess.run(
                command,
                cwd=self.base_dir,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )

            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if result.returncode != 0:
                print(f"Plotting failed with return code {result.returncode}")
                return 1, self.prompts["plot_error_prompt"].format(
                    error=result.stderr
                )

            return 0, ""

        except TimeoutExpired:
            print(f"Plotting timed out after {timeout} seconds")
            return 1, self.prompts["plot_timeout_prompt"].format(
                timeout=timeout
            )

    def _update_notes(self) -> None:
        """Update notes.txt with plot descriptions."""
        # Set files for this operation
        self.coder.fnames = [
            osp.join(self.base_dir, "notes.txt")
        ]
        self.coder.run(self.prompts["notes_prompt"])

    def _cleanup_failed_run(self, run_num: int) -> None:
        """Clean up files from a failed run."""
        run_dir = osp.join(self.base_dir, f"run_{run_num}")
        if osp.exists(run_dir):
            shutil.rmtree(run_dir)
