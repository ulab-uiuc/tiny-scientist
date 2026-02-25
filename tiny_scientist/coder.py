import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import time
from subprocess import TimeoutExpired
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, Runner
from rich import print

from .budget_checker import BudgetChecker
from .configs import Config
from .smolagents_tools import (
    DockerExperimentRunner,
    ReadFileTool,
    RunExperimentTool,
    WriteFileTool,
)
from .tools.agent_tools import (
    build_research_tools,
    make_read_file_tool,
    make_run_experiment_tool,
    make_write_file_tool,
)
from .utils.llm import create_client, extract_json_between_markers, get_response_from_llm
from .utils.sdk_client import configure_openai_agents_for_model, track_sdk_cost


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
        """Initialize the ExperimentCoder with configuration and openai-agents setup."""
        self.client, self.model = create_client(model)
        self.output_dir = osp.abspath(output_dir)
        self.max_iters = max_iters
        self.max_runs = 1
        self.max_stderr_output = max_stderr_output
        self.auto_install = auto_install
        self.config = Config()
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.use_docker = use_docker

        # Initialize Docker runner if needed
        self.docker_runner: Optional[DockerExperimentRunner]
        if self.use_docker:
            self.docker_runner = DockerExperimentRunner()
        else:
            self.docker_runner = None

        # Load prompts
        self.prompts = self.config.prompt_template.coder_prompt

        # openai-agents Agent (initialized in setup_agent)
        self.agent: Optional[Agent] = None
        self.planner_agent: Optional[Agent] = None
        self.validation_agent: Optional[Agent] = None

    def setup_agent(self) -> None:
        """Setup openai-agents Agent with tools for code generation."""
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure the SDK client for the chosen model
        configure_openai_agents_for_model(self.model)

        # Create function_tool wrappers for the file/experiment tools
        write_file = make_write_file_tool(WriteFileTool(self.output_dir))
        read_file = make_read_file_tool(ReadFileTool(self.output_dir))
        run_experiment = make_run_experiment_tool(
            RunExperimentTool(self.output_dir, self.docker_runner)
        )
        research_tools = build_research_tools(model=self.model, include_drawer=False)

        # Create the openai-agents Agent
        self.agent = Agent(
            name="ExperimentCoder",
            instructions=(
                "You are an expert Python code generator for machine learning experiments. "
                "Generate COMPLETE, RUNNABLE code with REAL data loading, model training, and evaluation. "
                "NEVER use random numbers, dummy data, or hardcoded results. "
                "All metrics must come from actual model execution. "
                "Use write_file to save experiment.py, read_file to inspect it, "
                "and run_experiment to execute it. "
                "Use web_search for up-to-date implementation details, paper_search for"
                " experiment design references, and code_search for baseline repos."
            ),
            tools=[write_file, read_file, run_experiment, *research_tools],
            model=self.model,
        )
        self.planner_agent = Agent(
            name="ExperimentPlanner",
            instructions=(
                "You are a coding planner for ML experiments. "
                "Return only a compact JSON array of steps. "
                "Each item must be: {step, name, description}. "
                "Do not include markdown fences."
            ),
            tools=research_tools,
            model=self.model,
        )
        self.validation_agent = Agent(
            name="ExperimentValidator",
            instructions=(
                "You validate experiment run outputs against the thinker-provided experiment table. "
                "Return ONLY JSON with keys: valid (bool), summary (string), issues (array of strings), "
                "matched_rows (array of strings), missing_rows (array of strings). "
                "Mark valid=false if metrics look like placeholders, NaN/inf, or unrelated to the table rows."
            ),
            tools=research_tools,
            model=self.model,
        )

    def run(
        self, idea: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        # Ensure a clean slate for every run
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup the openai-agents agent
        self.setup_agent()

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
        self._write_search_links_manifest(idea)

        result_summary: Dict[str, Any] = {}
        run_dir = osp.join(self.output_dir, "run")
        result_path = osp.join(run_dir, "final_info.json")
        if osp.exists(result_path):
            with open(result_path, "r") as f:
                result_summary["run"] = json.load(f)

        # Save combined results
        save_path = osp.join(self.output_dir, "experiment_results.txt")
        with open(save_path, "w") as f:
            json.dump(result_summary, f, indent=2)

        print(f"[System] All experiment results saved to {save_path}")

        self.cost_tracker.report()

        return True, self.output_dir, None

    def _format_experiment_for_prompt(
        self, exp: Dict[str, Any]
    ) -> Tuple[str, str, str, str, str, str]:
        model_section = self._serialize_for_prompt(exp.get("Model", ""))
        dataset_section = self._serialize_for_prompt(exp.get("Dataset", ""))
        metric_section = self._serialize_for_prompt(exp.get("Metric", ""))

        llm_prompt = self.prompts.experiment_keyword_prompt.format(
            model=model_section, dataset=dataset_section, metric=metric_section
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

        return (
            model_kw,
            dataset_kw,
            metric_kw,
            model_section,
            dataset_section,
            metric_section,
        )

    @staticmethod
    def _serialize_for_prompt(section: Any) -> str:
        if isinstance(section, (dict, list)):
            try:
                return json.dumps(section, indent=2, ensure_ascii=False)
            except TypeError:
                return str(section)
        return str(section)

    def _summarize_to_bullets(self, paragraph: str) -> str:
        # Simple sentence-splitting bullet conversion
        lines = paragraph.strip().split(". ")
        return "\n".join(f"- {line.strip().rstrip('.')}" for line in lines if line)

    def _plan(
        self,
        idea: Dict[str, Any],
        model_text: str,
        dataset_text: str,
        metric_text: str,
        experiment_table: str,
        baseline_results: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Use planner agent to generate a checklist of implementation steps."""
        baseline_note = (
            f"\n\nBaseline results for reference:\n{json.dumps(baseline_results, indent=2)}"
            if baseline_results
            else ""
        )
        prompt = self.prompts.experiment_plan_prompt.format(
            title=idea["Title"],
            problem=idea["Problem"],
            approach=idea["Approach"],
            model_details=model_text,
            dataset_details=dataset_text,
            metric_details=metric_text,
            experiment_table=experiment_table,
        ) + baseline_note

        if self.planner_agent is None:
            raise RuntimeError("Planner agent is not initialized for coder.")
        result = Runner.run_sync(self.planner_agent, prompt)
        track_sdk_cost(result, self.cost_tracker, self.model, "experiment_plan")
        response = result.final_output or ""

        # Parse JSON list from response
        try:
            clean = response.strip()
            # Strip markdown code fence if present
            if "```" in clean:
                start = clean.find("[", clean.find("```"))
                end = clean.rfind("]") + 1
                clean = clean[start:end]
            checklist = json.loads(clean)
            if isinstance(checklist, list) and checklist:
                normalized: List[Dict[str, Any]] = []
                for idx, item in enumerate(checklist, start=1):
                    if not isinstance(item, dict):
                        continue
                    row_refs = item.get("row_refs", [])
                    if not isinstance(row_refs, list):
                        row_refs = []
                    normalized.append(
                        {
                            "step": int(item.get("step", idx)),
                            "name": str(item.get("name", f"Step {idx}")),
                            "description": str(item.get("description", "")),
                            "row_refs": [str(r).strip() for r in row_refs if str(r).strip()],
                        }
                    )
                if normalized:
                    return normalized
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"[Planner][Coder] invalid TODO JSON: {e}") from e
        raise RuntimeError("[Planner][Coder] empty TODO returned by planner.")

    def _code_step(
        self,
        step: Dict[str, Any],
        idea: Dict[str, Any],
        total_steps: int,
        model_text: str,
        dataset_text: str,
        metric_text: str,
        experiment_table: str,
        todo_content: str,
    ) -> None:
        """Code a single checklist step, updating experiment.py incrementally."""
        exp_path = osp.join(self.output_dir, "experiment.py")
        current_code = ""
        if osp.exists(exp_path):
            with open(exp_path, "r") as f:
                current_code = f.read()

        prompt = self.prompts.experiment_step_prompt.format(
            step_num=step["step"],
            total_steps=total_steps,
            step_name=step["name"],
            step_description=step["description"],
            title=idea["Title"],
            problem=idea["Problem"],
            approach=idea["Approach"],
            model_details=model_text,
            dataset_details=dataset_text,
            metric_details=metric_text,
            experiment_table=experiment_table,
            todo_plan=todo_content,
            current_code=current_code if current_code else "(empty — create the file)",
        )

        self._generate_experiment(prompt)

    def _run_experiment_loop(
        self, idea: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Run the experiment loop: plan → code each step → execute runs."""
        current_iter = 0
        run_time = 1

        experiment_table = str(idea.get("ExperimentTable", "")).strip()
        if not experiment_table:
            raise ValueError(
                "Idea is missing ExperimentTable. Coder requires Thinker's table as the execution blueprint."
            )
        table_rows = self._extract_table_rows(experiment_table)

        experiment_spec = idea["Experiment"]
        _, _, _, model_text, dataset_text, metric_text = self._format_experiment_for_prompt(experiment_spec)

        # Phase 1: Plan — generate implementation checklist
        print("[Planner][Coder] Generating experiment TODO...")
        checklist = self._plan(
            idea,
            model_text,
            dataset_text,
            metric_text,
            experiment_table,
            baseline_results,
        )
        print(
            f"[Planner][Coder] TODO ({len(checklist)} steps), blueprint rows={len(table_rows)}"
        )
        for item in checklist:
            refs = self._resolve_row_refs(item, table_rows)
            refs_text = ", ".join(refs) if refs else "unmapped"
            print(
                f"[TODO][Coder] [ ] Step {item['step']}: {item['name']} — {item['description']} [rows: {refs_text}]"
            )

        # Persist a TODO plan so the agent can track execution state across steps.
        self._write_todo(checklist=checklist, completed_steps=0, table_rows=table_rows)

        # Phase 2: Code — implement each step incrementally
        total_steps = len(checklist)
        for step in checklist:
            print(f"\n[TODO][Coder] [{step['step']}/{total_steps}] {step['name']}")
            self._write_todo(
                checklist=checklist,
                completed_steps=step["step"] - 1,
                table_rows=table_rows,
            )
            todo_content = self._read_todo()
            self._code_step(
                step,
                idea,
                total_steps,
                model_text,
                dataset_text,
                metric_text,
                experiment_table,
                todo_content,
            )
            self._write_todo(
                checklist=checklist, completed_steps=step["step"], table_rows=table_rows
            )

        # Phase 3: Fix any remaining placeholders after step-coding
        exp_path = osp.join(self.output_dir, "experiment.py")
        if osp.exists(exp_path):
            with open(exp_path) as f:
                content = f.read()
            if "..." in content:
                raise RuntimeError(
                    "experiment.py still contains placeholder '...'; strict mode does not auto-fix."
                )

        # Phase 4: Run a single experiment ("run") with fix/retry logic.
        next_prompt = ""
        while current_iter < self.max_iters:
            if next_prompt:
                self._generate_experiment(next_prompt)

            return_code, message = self._run_single_experiment(
                run_num=run_time,
                idea=idea,
                experiment_table=experiment_table,
                table_rows=table_rows,
            )

            if return_code == 0:
                return True

            print("[System] Experiment run failed. Attempting fix...")
            next_prompt = self.prompts.experiment_error_prompt.format(
                message=message,
                Title=idea["Title"],
                Experiment=idea["Experiment"],
                run_time=run_time,
                max_runs=self.max_runs,
            )
            current_iter += 1

        print("Max iterations reached")
        return False

    def _generate_experiment(self, prompt: str) -> str:
        """Use openai-agents Agent to generate experiment code."""
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call setup_agent() first.")

        # Build a task prompt for the agent
        task_prompt = (
            "Your task is to generate COMPLETE, RUNNABLE Python code for an experiment.\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "1. Generate REAL code with actual data loading, model training, and evaluation\n"
            "2. NEVER use random numbers, dummy data, or hardcoded results\n"
            "3. All metrics must come from actual model execution\n"
            "4. The code must be self-contained and runnable\n"
            "5. Save results to a JSON file using argparse --out_dir argument\n\n"
            f"TASK:\n{prompt}\n\n"
            "Use write_file to save experiment.py with complete, working code.\n"
            "Use TODO.md as your execution checklist and keep task focus by current step.\n"
            "The code should accept --out_dir argument and save final_info.json with results.\n"
            'After writing the code, respond with "CONTINUE" to proceed.'
        )

        result = Runner.run_sync(self.agent, task_prompt)
        track_sdk_cost(result, self.cost_tracker, self.model, "generate_experiment")
        return result.final_output or "CONTINUE"

    def _write_todo(
        self,
        checklist: List[Dict[str, Any]],
        completed_steps: int,
        table_rows: List[str],
    ) -> None:
        """Write a markdown TODO tracker for current implementation progress."""
        todo_path = osp.join(self.output_dir, "TODO.md")
        lines = ["# Experiment TODO Plan", ""]
        for item in checklist:
            try:
                step_num = int(item.get("step", 0))
            except (TypeError, ValueError):
                continue
            name = str(item.get("name", f"Step {step_num}"))
            desc = str(item.get("description", ""))
            refs = self._resolve_row_refs(item, table_rows)
            refs_text = ", ".join(refs) if refs else "unmapped"
            checked = "x" if step_num <= completed_steps else " "
            lines.append(f"- [{checked}] Step {step_num}: {name} [rows: {refs_text}]")
            lines.append(f"  - {desc}")
        with open(todo_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")

    @staticmethod
    def _extract_table_rows(experiment_table: str) -> List[str]:
        rows: List[str] = []
        for raw in experiment_table.splitlines():
            line = raw.strip()
            if not line.startswith("|"):
                continue
            if re.fullmatch(r"\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", line):
                continue
            cols = [c.strip() for c in line.strip("|").split("|")]
            if not cols:
                continue
            head = cols[0]
            if head.lower() in {"component", "step", "item"}:
                continue
            if head:
                rows.append(head)
        seen = set()
        ordered: List[str] = []
        for row in rows:
            key = row.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(row)
        return ordered

    @staticmethod
    def _resolve_row_refs(step: Dict[str, Any], table_rows: List[str]) -> List[str]:
        row_refs = step.get("row_refs", [])
        if isinstance(row_refs, list) and row_refs:
            refs = [str(r).strip() for r in row_refs if str(r).strip()]
            if refs:
                return refs
        text = f"{step.get('name', '')} {step.get('description', '')}".lower()
        inferred = [r for r in table_rows if r.lower() in text]
        return inferred[:3]

    def _read_todo(self) -> str:
        """Read TODO tracker text for prompt context."""
        todo_path = osp.join(self.output_dir, "TODO.md")
        if not osp.exists(todo_path):
            return "(No TODO plan available)"
        with open(todo_path, "r", encoding="utf-8") as f:
            return f.read()

    def _run_single_experiment(
        self,
        run_num: int,
        idea: Dict[str, Any],
        experiment_table: str,
        table_rows: List[str],
        timeout: int = 7200,
    ) -> Tuple[int, str]:
        """Run a single experiment iteration."""
        _ = run_num
        shutil.copy(
            osp.join(self.output_dir, "experiment.py"),
            osp.join(self.output_dir, "run.py"),
        )

        with open(osp.join(self.output_dir, "experiment.py"), "r") as f:
            experiment_code = f.read()

        # Try Docker first if available
        if self.use_docker and self.docker_runner and self.docker_runner.use_docker:
            docker_result = self.docker_runner.run_experiment_in_docker(
                experiment_code, run_num, self.output_dir, timeout
            )
            if docker_result is not None:
                return_code, logs = docker_result
                return return_code, logs

        # Fallback to local execution
        command = ["python", "experiment.py", "--out_dir=run"]

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
                print(f"Run failed with return code {result.returncode}")
                if "ModuleNotFoundError" in result.stderr and getattr(
                    self, "auto_install", True
                ):
                    missing_pkg = DockerExperimentRunner.extract_missing_package(
                        result.stderr
                    )
                    print(
                        f"[System] Missing package detected: {missing_pkg}. Attempting to install..."
                    )
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
                        return self._run_single_experiment(
                            run_num=run_num,
                            idea=idea,
                            experiment_table=experiment_table,
                            table_rows=table_rows,
                            timeout=timeout,
                        )
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
            results_path = osp.join(self.output_dir, "run", "final_info.json")
            if osp.exists(results_path):
                with open(results_path, "r") as f:
                    results = json.load(f)
            else:
                results = {}

            if isinstance(results, dict):
                results = {
                    k: v["means"] if isinstance(v, dict) and "means" in v else v
                    for k, v in results.items()
                }
            elif isinstance(results, list):
                results = {f"entry_{i+1}": entry for i, entry in enumerate(results)}
            else:
                results = {}

            valid, validation_message = self._validate_run_results(
                run_num=run_num,
                idea=idea,
                experiment_table=experiment_table,
                table_rows=table_rows,
                run_results=results,
            )
            if not valid:
                self._cleanup_failed_run(run_num)
                return 1, validation_message

            return 0, self.prompts.experiment_success_prompt.format(
                run_num=run_num, results=results, next_run=run_num + 1
            )

        except TimeoutExpired:
            print(f"Run timed out after {timeout} seconds")
            self._cleanup_failed_run(run_num)
            return 1, self.prompts.experiment_timeout_prompt.format(timeout=timeout)

    def _validate_run_results(
        self,
        run_num: int,
        idea: Dict[str, Any],
        experiment_table: str,
        table_rows: List[str],
        run_results: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if self.validation_agent is None:
            raise RuntimeError("Validation agent is not initialized for coder.")

        _ = run_num
        results_path = osp.join(self.output_dir, "run", "final_info.json")
        experiment_py_path = osp.join(self.output_dir, "experiment.py")
        experiment_code = ""
        if osp.exists(experiment_py_path):
            with open(experiment_py_path, "r", encoding="utf-8") as f:
                experiment_code = f.read()

        prompt = (
            "Validate whether this run output is scientifically and structurally correct.\n\n"
            f"Idea title: {idea.get('Title', '')}\n"
            f"Problem: {idea.get('Problem', '')}\n"
            f"Approach: {idea.get('Approach', '')}\n\n"
            "Experiment table (authoritative):\n"
            f"{experiment_table}\n\n"
            f"Table rows: {json.dumps(table_rows, ensure_ascii=False)}\n\n"
            "Run label: run\n"
            f"Result file path: {results_path}\n"
            f"Run results JSON:\n{json.dumps(run_results, ensure_ascii=False, indent=2)}\n\n"
            "Current experiment.py:\n"
            f"{experiment_code[:12000]}\n\n"
            "Validation rules:\n"
            "1) Results must be non-empty numeric metrics (no placeholder text, no NaN/inf).\n"
            "2) Results must align with experiment table rows and stated metrics.\n"
            "3) Detect suspicious outputs (all zeros, repeated constants, clearly dummy values).\n"
            "4) If invalid, provide concrete issues and missing_rows.\n"
            "Return JSON only."
        )
        result = Runner.run_sync(self.validation_agent, prompt)
        track_sdk_cost(result, self.cost_tracker, self.model, "validate_experiment_run")
        text = result.final_output or ""
        parsed: Any = extract_json_between_markers(text)
        if not isinstance(parsed, dict):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
        if not isinstance(parsed, dict):
            return False, "[Validator][Coder] invalid validator output format."

        report_path = osp.join(self.output_dir, "run", "validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        is_valid = bool(parsed.get("valid", False))
        summary = str(parsed.get("summary", "")).strip()
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        issue_text = "; ".join(str(i) for i in issues if str(i).strip())
        if is_valid:
            print(f"[Validator][Coder] run PASSED: {summary or 'validated'}")
            return True, summary or "validated"

        msg = summary or issue_text or "validation failed"
        print(f"[Validator][Coder] run FAILED: {msg}")
        return False, f"[Validator][Coder] run failed: {msg}"

    def _update_notes(self) -> None:
        """Update notes.txt with plot descriptions."""
        try:
            notes_path = osp.join(self.output_dir, "notes.txt")
            current_notes = ""
            if osp.exists(notes_path):
                with open(notes_path, "r") as f:
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

            with open(notes_path, "w") as f:
                f.write(response.strip())

        except Exception as e:
            print(f"[System] Failed to update notes: {e}")

    def _write_search_links_manifest(self, idea: Dict[str, Any]) -> None:
        links: List[Dict[str, str]] = []
        seen: set[str] = set()

        def _add_link(title: str, url: str, source_type: str) -> None:
            clean_url = (url or "").strip()
            if not clean_url.startswith(("http://", "https://")):
                return
            key = clean_url.lower()
            if key in seen:
                return
            seen.add(key)
            links.append(
                {
                    "title": (title or "Untitled").strip() or "Untitled",
                    "url": clean_url,
                    "source_type": (source_type or "unknown").strip() or "unknown",
                }
            )

        citations = idea.get("Citations", [])
        if isinstance(citations, list):
            for c in citations:
                if not isinstance(c, dict):
                    continue
                _add_link(
                    title=str(c.get("title", "Untitled")),
                    url=str(c.get("url", "")),
                    source_type=str(c.get("source_type", "thinker")),
                )

        grounding = idea.get("ResearchGrounding", {})
        if isinstance(grounding, dict):
            grounded = grounding.get("citations", [])
            if isinstance(grounded, list):
                for c in grounded:
                    if not isinstance(c, dict):
                        continue
                    _add_link(
                        title=str(c.get("title", "Untitled")),
                        url=str(c.get("url", "")),
                        source_type=str(c.get("source_type", "grounding")),
                    )

        manifest_path = osp.join(self.output_dir, "search_links.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"search_links": links}, f, indent=2, ensure_ascii=False)

        notes_path = osp.join(self.output_dir, "notes.txt")
        section_lines = [
            "",
            "## Search Links",
        ]
        if links:
            for link in links:
                section_lines.append(
                    f"- {link['title']} ({link['source_type']}): {link['url']}"
                )
        else:
            section_lines.append("- No search links available.")
        addition = "\n".join(section_lines) + "\n"

        existing = ""
        if osp.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                existing = f.read()
        if "## Search Links" in existing:
            existing = existing.split("## Search Links")[0].rstrip() + "\n"
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(existing + addition)

    def _cleanup_failed_run(self, run_num: int) -> None:
        """Clean up files from a failed run."""
        _ = run_num
        run_dir = osp.join(self.output_dir, "run")
        if osp.exists(run_dir):
            shutil.rmtree(run_dir)

    def cleanup_docker_images(self) -> None:
        """Clean up Docker images created during experiments."""
        if self.docker_runner:
            self.docker_runner.cleanup_docker_images()
