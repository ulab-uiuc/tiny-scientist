import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from subprocess import TimeoutExpired
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from aider.coders import Coder as AiderCoder
from aider.io import InputOutput
from aider.models import Model
from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .tool import DockerExperimentRunner
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
        use_docker: bool = True,
    ) -> None:
        """Initialize the ExperimentCoder with configuration and Aider setup."""
        self.client, self.model = create_client(model)
        self.output_dir = osp.abspath(output_dir)
        self.max_iters = max_iters
        self.max_runs = max_runs
        self.max_stderr_output = max_stderr_output
        self.auto_install = auto_install
        self.config = Config()
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.use_docker = use_docker
        
        # Initialize Docker runner if needed
        if self.use_docker:
            self.docker_runner = DockerExperimentRunner()
        else:
            self.docker_runner = None

        # Load prompts
        self.prompts = self.config.prompt_template.coder_prompt

    def setup_aider(
        self, model: str, fnames: List[str], chat_history: Optional[str] = None
    ) -> None:
        """Setup Aider coder with the specified model."""
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Set environment variables to make Aider non-interactive
        os.environ['AIDER_YES'] = '1'
        os.environ['AIDER_QUIET'] = '1'
        
        # Try to make Aider completely non-interactive
        io = InputOutput(
            yes=True, 
            input_history_file=None,
            chat_history_file=None
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

            try:
                # Try to use a non-interactive approach first
                coder_out = self._run_non_interactive_aider(next_prompt)
            except Exception as e:
                print(f"[System] Non-interactive Aider failed: {e}")
                try:
                    coder_out = self.coder.run(next_prompt)
                except Exception as e2:
                    print(f"[System] Interactive Aider also failed: {e2}")
                    # If Aider fails, try to continue with what we have
                    coder_out = "CONTINUE"
            exp_path = osp.join(self.output_dir, "experiment.py")

            if "ALL_COMPLETED" in coder_out:
                return True

            if osp.exists(exp_path):
                with open(exp_path) as f:
                    content = f.read()
                    if "..." in content:
                        print("[System] Placeholder '...' detected. Attempting fix.")
                        try:
                            self._run_non_interactive_aider(
                                "Please replace all placeholders (`...`) in experiment.py with complete runnable code."
                            )
                        except Exception as e:
                            print(f"[System] Failed to fix placeholders: {e}")

            return_code, message = self._run_single_experiment(run_time)

            if return_code == 0:
                run_time += 1
                current_iter = 0
                next_prompt = message
            else:
                print("[System] Experiment run failed. Attempting fix with Aider...")
                try:
                    next_prompt = self.prompts.experiment_error_prompt.format(
                        message=message,
                        Title=idea["Title"],
                        Experiment=idea["Experiment"],
                        run_time=run_time,
                        max_runs=self.max_runs,
                    )
                    # Try to run non-interactive Aider to fix the issue
                    self._run_non_interactive_aider(next_prompt)
                except Exception as e:
                    print(f"[System] Non-interactive Aider fix attempt failed: {e}")
                    # If Aider fails, just continue to next iteration
                
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

        # Read experiment code
        with open(osp.join(self.output_dir, "experiment.py"), "r") as f:
            experiment_code = f.read()

        # Try Docker first if available
        if self.use_docker and self.docker_runner and self.docker_runner.use_docker:
            return_code, logs = self.docker_runner.run_experiment_in_docker(
                experiment_code, run_num, self.output_dir, timeout
            )
            if return_code is not None:  # Docker was used
                return return_code, logs

        # Fallback to local execution
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
                if "ModuleNotFoundError" in result.stderr and getattr(self, "auto_install", True):
                    missing_pkg = DockerExperimentRunner.extract_missing_package(result.stderr)
                    print(f"[System] Missing package detected: {missing_pkg}. Attempting to install...")
                    try:
                        install_result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", missing_pkg],
                            capture_output=True,
                            text=True,
                            timeout=300,
                            check=True,
                        )
                        print(f"[System] Successfully installed {missing_pkg}")
                        print(f"[System] Install output: {install_result.stdout}")
                        time.sleep(2)
                        print("[System] Re-running after installing dependency...")
                        return self._run_single_experiment(run_num, timeout=timeout)
                    except subprocess.TimeoutExpired:
                        print(f"[System] Package installation timed out after 5 minutes for {missing_pkg}")
                        return 1, f"Package installation timeout for {missing_pkg}"
                    except subprocess.CalledProcessError as e:
                        print(f"[System] Package installation failed for {missing_pkg}")
                        print(f"[System] Installation error: {e.stderr}")
                        return 1, f"Package installation failed for {missing_pkg}: {e.stderr}"
                    except Exception as e:
                        print(f"[System] Unexpected error during package installation: {str(e)}")
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
        try:
            # Use non-interactive approach for notes too
            notes_path = osp.join(self.output_dir, "notes.txt")
            current_notes = ""
            if osp.exists(notes_path):
                with open(notes_path, 'r') as f:
                    current_notes = f.read()
            
            full_prompt = f"""
{self.prompts.notes_prompt}

Current notes:
{current_notes}

Please provide the complete updated notes content.
"""
            
            response, _ = get_response_from_llm(
                msg=full_prompt,
                client=self.client,
                model=self.model,
                system_message="You are a technical writer. Provide only the notes content without any markdown formatting.",
                cost_tracker=self.cost_tracker,
                task_name="update_notes",
            )
            
            with open(notes_path, 'w') as f:
                f.write(response.strip())
                
        except Exception as e:
            print(f"[System] Failed to update notes: {e}")

    def _cleanup_failed_run(self, run_num: int) -> None:
        """Clean up files from a failed run."""
        run_dir = osp.join(self.output_dir, f"run_{run_num}")
        if osp.exists(run_dir):
            shutil.rmtree(run_dir)

    def _run_non_interactive_aider(self, prompt: str) -> str:
        """Run Aider in a non-interactive mode by directly calling the LLM."""
        try:
            # Get the current experiment.py content
            exp_path = osp.join(self.output_dir, "experiment.py")
            current_content = ""
            if osp.exists(exp_path):
                with open(exp_path, 'r') as f:
                    current_content = f.read()
            
            # Create a prompt that asks for the complete file content
            full_prompt = f"""
{prompt}

Please provide the complete content for experiment.py. If the file already exists, please provide the corrected/improved version.

Current file content:
{current_content}

Please respond with the complete file content only, no explanations or markdown formatting.
"""
            
            # Call the LLM directly
            response, _ = get_response_from_llm(
                msg=full_prompt,
                client=self.client,
                model=self.model,
                system_message="You are a Python code generator. Provide only the complete Python code without any markdown formatting or explanations.",
                cost_tracker=self.cost_tracker,
                task_name="non_interactive_aider",
            )
            
            # Clean the response to extract just the code
            code_content = self._extract_code_from_response(response)
            
            # Write the code to the file
            with open(exp_path, 'w') as f:
                f.write(code_content)
            
            return "CONTINUE"
            
        except Exception as e:
            print(f"[System] Non-interactive Aider failed: {e}")
            raise
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Remove markdown code blocks if present
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        return response.strip()

    def cleanup_docker_images(self) -> None:
        """Clean up Docker images created during experiments."""
        if self.docker_runner:
            self.docker_runner.cleanup_docker_images()
