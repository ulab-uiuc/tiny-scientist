import builtins
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Union

import eventlet

eventlet.monkey_patch()

from flask import Flask, Response, jsonify, request, send_file, session  # noqa: E402
from flask_cors import CORS  # noqa: E402
from flask_socketio import SocketIO  # noqa: E402

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


original_print = builtins.print

_LOG_BUFFER: list[dict[str, Any]] = []
_MAX_LOG_BUFFER_LENGTH = 500


def _push_log(message: str, level: str = "info") -> None:
    message = message.strip()
    if not message:
        return
    payload = {
        "message": message,
        "level": level,
        "timestamp": time.time(),
    }
    _LOG_BUFFER.append(payload)
    if len(_LOG_BUFFER) > _MAX_LOG_BUFFER_LENGTH:
        del _LOG_BUFFER[0]
    try:
        if "socketio" in globals():
            socketio.emit("log", payload)
    except Exception:
        pass


class SocketIOLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        _push_log(message, record.levelname.lower())


def flush_log_buffer() -> None:
    global _LOG_BUFFER
    if not _LOG_BUFFER:
        return
    try:
        for payload in _LOG_BUFFER:
            socketio.emit("log", payload)
    except Exception:
        return
    finally:
        _LOG_BUFFER = []


def websocket_print(*args: Any, **kwargs: Any) -> None:
    message = " ".join(str(arg) for arg in args)
    _push_log(message)
    original_print(*args, **kwargs)


# Override print globally before importing tiny_scientist modules
builtins.print = websocket_print

# Also need to override rich.print which is used in many modules
try:
    import rich

    rich.print = websocket_print
    # Also override the console print if rich.console exists
    if hasattr(rich, "console"):
        rich.console.print = websocket_print  # type: ignore
except ImportError:
    pass


from backend.demo_cache import DemoCacheError, DemoCacheService  # noqa: E402
from tiny_scientist.budget_checker import BudgetChecker  # noqa: E402
from tiny_scientist.coder import Coder  # noqa: E402
from tiny_scientist.reviewer import Reviewer  # noqa: E402
from tiny_scientist.scientist import TinyScientist  # noqa: E402
from tiny_scientist.thinker import Thinker  # noqa: E402
from tiny_scientist.writer import Writer  # noqa: E402


# Patch print in the imported modules - this needs to happen after import
# The modules use "from rich import print" so we need to patch their local print
def patch_module_print() -> None:
    import sys

    modules_to_patch = [
        sys.modules.get("tiny_scientist.thinker"),
        sys.modules.get("tiny_scientist.coder"),
        sys.modules.get("tiny_scientist.writer"),
        sys.modules.get("tiny_scientist.reviewer"),
    ]

    for module in modules_to_patch:
        if module and hasattr(module, "print"):
            # Replace with our websocket print
            module.print = websocket_print  # type: ignore
            print(f"✅ Patched print in {module.__name__}")


# Call the patching function
patch_module_print()


def _demo_log(message: str, level: str = "info") -> None:
    _push_log(message, level)


DEMO_CACHE_MODE = os.environ.get("DEMO_CACHE_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEMO_CACHE_DIR = os.path.abspath(
    os.environ.get("DEMO_CACHE_DIR")
    or os.path.join(project_root, "frontend", "demo_cache")
)

try:
    demo_cache = DemoCacheService(
        DEMO_CACHE_DIR,
        enabled=DEMO_CACHE_MODE,
        log_fn=_demo_log,
    )
    if demo_cache.enabled:
        print(f"🗄️  Demo cache enabled using data at {DEMO_CACHE_DIR}")
except DemoCacheError as exc:
    print(f"⚠️ Demo cache disabled: {exc}")
    demo_cache = DemoCacheService(
        DEMO_CACHE_DIR,
        enabled=False,
        log_fn=_demo_log,
    )

app = Flask(__name__)
app.secret_key = "your-secret-key-here"
CORS(
    app,
    supports_credentials=True,
    origins=[
        "https://app.auto-research.dev",
        "http://app.auto-research.dev",
        "http://localhost:3000",
    ],
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
root_logger = logging.getLogger()
if not any(isinstance(handler, SocketIOLogHandler) for handler in root_logger.handlers):
    socketio_handler = SocketIOLogHandler()
    socketio_handler.setLevel(logging.INFO)
    socketio_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(socketio_handler)
if root_logger.level > logging.INFO:
    root_logger.setLevel(logging.INFO)

# Print override is now active
print("🚀 Backend server starting with WebSocket logging enabled!")


thinker: Optional[Thinker] = None
coder: Optional[Coder] = None
writer: Optional[Writer] = None
reviewer: Optional[Reviewer] = None
global_cost_tracker: Optional[BudgetChecker] = None


def format_name_for_display(name: Optional[str]) -> str:
    """Formats a name"""
    if not name:
        return "Untitled"
    return " ".join(word.capitalize() for word in name.split("_"))


# Initialize the Thinker
@app.route("/api/configure", methods=["POST"])
def configure() -> Union[Response, tuple[Response, int]]:
    """Configure model and API key"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if demo_cache.enabled:
        try:
            response_payload = demo_cache.apply_config(session)
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)
    model = data.get("model")
    api_key = data.get("api_key")
    budget = data.get("budget")
    budget_preference = data.get("budget_preference")

    if not model or not api_key:
        return jsonify({"error": "Model and API key are required"}), 400

    try:
        (
            resolved_budget,
            resolved_preference,
            allocation,
        ) = TinyScientist.resolve_budget_settings(budget, budget_preference)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Map models to their environment variables
    env_var_map = {
        "deepseek-chat": "DEEPSEEK_API_KEY",
        "deepseek-reasoner": "DEEPSEEK_API_KEY",
        "gpt-4o": "OPENAI_API_KEY",
        "gpt-o1": "OPENAI_API_KEY",
        "claude-3-5-sonnet": "ANTHROPIC_API_KEY",
        "claude-sonnet-4-5": "ANTHROPIC_API_KEY",
        "claude-haiku-4-5": "ANTHROPIC_API_KEY",
    }

    # Set the appropriate environment variable
    env_var = env_var_map.get(model)
    if env_var:
        os.environ[env_var] = api_key

    # Store in session
    session["model"] = model
    session["api_key"] = api_key
    session["configured"] = True
    session["budget"] = resolved_budget
    session["budget_preference"] = resolved_preference

    # Initialize all components with same parameters as TinyScientist
    global thinker, coder, writer, reviewer, global_cost_tracker

    # Use absolute paths outside React's file watching but still accessible
    experiments_dir = os.path.join(project_root, "generated", "experiments")
    papers_dir = os.path.join(project_root, "generated", "papers")

    # Ensure directories exist
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(papers_dir, exist_ok=True)

    print(f"Backend directory: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Experiments directory: {os.path.abspath(experiments_dir)}")
    print(f"Papers directory: {os.path.abspath(papers_dir)}")

    global_cost_tracker = BudgetChecker(budget=resolved_budget)

    thinker = Thinker(
        model=model,
        tools=[],
        iter_num=0,
        output_dir="./",
        search_papers=False,
        generate_exp_plan=True,
        cost_tracker=BudgetChecker(
            budget=allocation.get("thinker"), parent=global_cost_tracker
        ),
    )
    coder = Coder(
        model=model,
        output_dir=experiments_dir,
        max_iters=4,
        max_runs=3,
        cost_tracker=BudgetChecker(
            budget=allocation.get("coder"), parent=global_cost_tracker
        ),
    )

    return jsonify(
        {
            "status": "configured",
            "model": model,
            "budget": resolved_budget,
            "budget_preference": resolved_preference,
        }
    )


@app.route("/api/generate-initial", methods=["POST"])
def generate_initial() -> Union[Response, tuple[Response, int]]:
    """Generate initial ideas from an intent (handleAnalysisIntentSubmit)"""
    flush_log_buffer()  # Emit any buffered logs from module initialization
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    intent = data.get("intent")
    num_ideas = data.get("num_ideas", 3)

    if demo_cache.enabled:
        try:
            response_payload = demo_cache.get_initial_ideas(intent)
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)

    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    # Generate ideas
    ideas = thinker.run(intent=intent, num_ideas=num_ideas)

    # Return in the format expected by TreePlot
    response = {
        "ideas": [
            {
                "title": format_name_for_display(
                    idea.get("Name") if isinstance(idea, dict) else None
                ),
                "content": (
                    format_idea_content(idea) if isinstance(idea, dict) else str(idea)
                ),
                "originalData": idea,  # Preserve complete thinker JSON for coder/writer
            }
            for idea in ideas
        ]
    }

    return jsonify(response)


@app.route("/api/set-system-prompt", methods=["POST"])
def set_system_prompt() -> Union[Response, tuple[Response, int]]:
    """Set the system prompt for the Thinker"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    system_prompt = data.get("system_prompt")

    if demo_cache.enabled:
        try:
            demo_cache.update_system_prompt(system_prompt)
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"status": "success", "message": "System prompt updated"})

    global thinker

    if not thinker:
        return jsonify({"error": "Thinker not configured"}), 400

    # If empty string or None, reset to default
    if not system_prompt:
        thinker.set_system_prompt(None)  # This will reset to default
    else:
        thinker.set_system_prompt(system_prompt)

    return jsonify({"status": "success", "message": "System prompt updated"})


@app.route("/api/set-criteria", methods=["POST"])
def set_criteria() -> Union[Response, tuple[Response, int]]:
    """Set evaluation criteria for a specific dimension"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    dimension = data.get("dimension")  # 'novelty', 'feasibility', or 'impact'
    criteria = data.get("criteria")

    if dimension not in ["novelty", "feasibility", "impact"]:
        return jsonify({"error": "Invalid dimension"}), 400

    if demo_cache.enabled:
        try:
            demo_cache.update_criteria(dimension, criteria)
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(
            {
                "status": "success",
                "message": f"{dimension.capitalize()} criteria updated",
            }
        )

    global thinker

    if not thinker:
        return jsonify({"error": "Thinker not configured"}), 400

    # If empty string or None, reset to default
    if not criteria:
        thinker.set_criteria(dimension, None)  # This will reset to default
    else:
        thinker.set_criteria(dimension, criteria)

    return jsonify(
        {"status": "success", "message": f"{dimension.capitalize()} criteria updated"}
    )


@app.route("/api/get-prompts", methods=["GET"])
def get_prompts() -> Union[Response, tuple[Response, int]]:
    """Get current prompts and criteria"""
    if demo_cache.enabled:
        try:
            return jsonify(demo_cache.get_prompts())
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409

    global thinker

    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    return jsonify(
        {
            "system_prompt": thinker.get_system_prompt(),
            "criteria": {
                "novelty": thinker.get_criteria("novelty"),
                "feasibility": thinker.get_criteria("feasibility"),
                "impact": thinker.get_criteria("impact"),
            },
            "defaults": {
                "system_prompt": thinker.default_system_prompt,
                "novelty": thinker.default_novelty_criteria,
                "feasibility": thinker.default_feasibility_criteria,
                "impact": thinker.default_impact_criteria,
            },
        }
    )


@app.route("/api/generate-children", methods=["POST"])
def generate_children() -> Union[Response, tuple[Response, int]]:
    """Generate child ideas (generateChildNodes)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if demo_cache.enabled:
        try:
            response_payload = demo_cache.get_child_ideas()
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    parent_content = data.get("parent_content")
    context = data.get("context", "")

    # Combine parent content and context as the intent
    combined_intent = f"{parent_content}\nAdditional Context: {context}"
    ideas = thinker.run(intent=combined_intent, num_ideas=3)

    # Return in the format expected by TreePlot
    response = {
        "ideas": [
            {
                "title": format_name_for_display(
                    idea.get("Name") if isinstance(idea, dict) else None
                ),
                "content": (
                    format_idea_content(idea) if isinstance(idea, dict) else str(idea)
                ),
                "originalData": idea,  # Preserve complete thinker JSON for coder/writer
            }
            for idea in ideas
        ]
    }

    return jsonify(response)


@app.route("/api/modify", methods=["POST"])
def modify_idea() -> Union[Response, tuple[Response, int]]:
    """Modify an idea (modifyIdeaBasedOnModifications)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if demo_cache.enabled:
        try:
            response_payload = demo_cache.get_modified_idea()
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    original_idea = data.get("original_idea")
    modifications = data.get("modifications")
    behind_idea = data.get("behind_idea")

    # Use original data directly (no conversion needed)
    thinker_original = original_idea
    thinker_behind = behind_idea
    # Convert modifications to Thinker format
    thinker_mods = []
    for mod in modifications:
        thinker_mods.append(
            {"metric": mod.get("metric"), "direction": mod.get("direction")}
        )

    # Modify the idea
    modified_idea = thinker.modify_idea(
        original_idea=thinker_original,
        modifications=thinker_mods,
        behind_idea=thinker_behind,
    )
    # Return in the format expected by TreePlot
    response = {
        "title": format_name_for_display(modified_idea.get("Name")),
        "content": format_idea_content(modified_idea),
        "originalData": modified_idea,  # Preserve complete thinker JSON for coder/writer
    }
    return jsonify(response)


@app.route("/api/merge", methods=["POST"])
def merge_ideas() -> Union[Response, tuple[Response, int]]:
    """Merge two ideas (mergeIdeas)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if demo_cache.enabled:
        try:
            response_payload = demo_cache.get_merged_idea()
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    idea_a = data.get("idea_a")
    idea_b = data.get("idea_b")
    # Use original data directly (no conversion needed)
    thinker_idea_a = idea_a
    thinker_idea_b = idea_b
    # Merge ideas
    merged_idea = thinker.merge_ideas(idea_a=thinker_idea_a, idea_b=thinker_idea_b)

    # Return in the format expected by TreePlot
    response = {
        "title": format_name_for_display(
            merged_idea.get("Name") if isinstance(merged_idea, dict) else None
        ),
        "content": (
            format_idea_content(merged_idea)
            if isinstance(merged_idea, dict)
            else str(merged_idea)
        ),
        "originalData": merged_idea,  # Preserve complete thinker JSON for coder/writer
    }

    return jsonify(response)


@app.route("/api/evaluate", methods=["POST"])
def evaluate_ideas() -> Union[Response, tuple[Response, int]]:
    """Evaluate ideas (evaluateIdeas)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    ideas = data.get("ideas") or []
    intent = data.get("intent")

    if demo_cache.enabled:
        try:
            response_payload = demo_cache.evaluate(ideas)
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    # Use original data directly (no conversion needed)
    thinker_ideas = ideas

    # Store original IDs in order - LLM doesn't preserve titles, use index mapping
    original_ids = [idea.get("id") for idea in ideas]

    # Rank ideas
    scored_ideas = thinker.rank(ideas=thinker_ideas, intent=intent)

    # Return in the format expected by TreePlot
    # Use index-based mapping since LLM changes titles but preserves order
    response = []
    for i, idea in enumerate(scored_ideas):
        # Use index to map back to original ID
        original_id = original_ids[i] if i < len(original_ids) else f"idea_{i}"

        response.append(
            {
                "id": original_id,  # Use index-based mapping
                "noveltyScore": idea.get("NoveltyScore"),
                "noveltyReason": idea.get("NoveltyReason", ""),
                "feasibilityScore": idea.get("FeasibilityScore"),
                "feasibilityReason": idea.get("FeasibilityReason", ""),
                "impactScore": idea.get("ImpactScore"),
                "impactReason": idea.get("ImpactReason", ""),
            }
        )
    return jsonify(response)


def format_idea_content(idea: Union[Dict[str, Any], str]) -> str:
    """
    Formats the Thinker's idea JSON into a single content string for the frontend.
    This is now simplified, as the frontend will pull most data from originalData.
    We only format the main sections for basic display.
    """
    if isinstance(idea, str):
        return idea

    description = idea.get("Description", "")
    importance = idea.get("Importance", "")
    difficulty = idea.get("Difficulty", "")
    novelty = idea.get("NoveltyComparison", "")

    content_sections = [
        f"**Description:**\n{description}",
        f"**Impact:**\n{importance}",
        f"**Feasibility:**\n{difficulty}",
        f"**Novelty:**\n{novelty}",
    ]

    return "\n\n".join(content_sections)


@app.route("/api/code", methods=["POST"])
def generate_code() -> Union[Response, tuple[Response, int]]:
    """Generate code synchronously and return when complete"""
    flush_log_buffer()  # Emit any buffered logs
    global coder

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    idea_data = data.get("idea")
    baseline_results = data.get("baseline_results", {})
    idea_id = data.get("idea_id")
    idea_name = None
    if isinstance(idea_data, dict):
        if not idea_id:
            idea_candidate = idea_data.get("id")
            if isinstance(idea_candidate, str):
                idea_id = idea_candidate
        idea_name = (
            idea_data.get("Name")
            or idea_data.get("Title")
            or idea_data.get("name")
            or idea_data.get("title")
        )

    if demo_cache.enabled:
        try:
            response_payload = demo_cache.get_code_result(
                idea_id=idea_id,
                idea_name=idea_name,
            )
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)

    if coder is None:
        return jsonify({"error": "Coder not configured"}), 400

    print("💻 Starting synchronous code generation...")

    if not idea_data:
        return jsonify({"error": "No idea provided"}), 400

    try:
        # Extract the original idea data
        if isinstance(idea_data, dict) and "originalData" in idea_data:
            idea = idea_data["originalData"]
        else:
            idea = idea_data

        print(f"Using pre-configured Coder with model: {coder.model}")
        print(f"Idea keys: {list(idea.keys())}")
        print(f"Idea has Experiment field: {'Experiment' in idea}")

        # Call coder.run() exactly like TinyScientist does
        print("Starting coder.run()...")
        print("This may take several minutes - please wait...")

        sys.stdout.flush()  # Ensure logs are written immediately

        status, exp_path, error_details = coder.run(
            idea=idea, baseline_results=baseline_results
        )

        print(f"Coder completed with status: {status}")
        if not status:
            print(f"❌ Experiment failed. Please check {exp_path} for details.")
            if error_details:
                print(f"Final error message:\n{error_details}")
        # --- START OF FIX ---
        # The path cleaning logic is the primary issue.
        # We need to make the absolute `exp_path` relative to the `generated` directory
        # so the frontend can build a correct API URL.
        generated_base_dir = os.path.join(project_root, "generated")
        clean_exp_path = os.path.relpath(exp_path, generated_base_dir)
        # On Unix-like systems, this ensures we use forward slashes for the URL
        clean_exp_path = clean_exp_path.replace(os.path.sep, "/")
        # --- END OF FIX ---

        response = {
            "status": status,
            "experiment_dir": clean_exp_path,
            "success": status is True,
            "message": (
                "Code generation completed successfully"
                if status
                else "Code generation failed"
            ),
            "error_details": error_details,
        }

        return jsonify(response)

    except Exception as e:
        print(f"ERROR in code generation: {e}")
        import traceback

        traceback.print_exc()

        return (
            jsonify(
                {
                    "error": str(e),
                    "success": False,
                    "error_details": traceback.format_exc(),
                }
            ),
            500,
        )


@app.route("/api/write", methods=["POST"])
def generate_paper() -> Union[Response, tuple[Response, int]]:
    """Generate a paper from an idea using the Writer class"""
    # global writer

    print("📝 Paper generation request received")

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    print(f"Request data: {data}")

    idea_data = data.get("idea")
    experiment_dir = data.get("experiment_dir", None)
    idea_id = data.get("idea_id")
    idea_name = None
    if isinstance(idea_data, dict):
        if not idea_id:
            idea_candidate = idea_data.get("id")
            if isinstance(idea_candidate, str):
                idea_id = idea_candidate
        idea_name = (
            idea_data.get("Name")
            or idea_data.get("Title")
            or idea_data.get("name")
            or idea_data.get("title")
        )

    if demo_cache.enabled:
        try:
            response_payload = demo_cache.get_paper_result(
                idea_id=idea_id,
                idea_name=idea_name,
                experiment_hint=experiment_dir,
            )
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)

    s2_api_key = data.get("s2_api_key", None)
    if isinstance(s2_api_key, str):
        s2_api_key = s2_api_key.strip() or None

    if not idea_data:
        print("ERROR: No idea provided in request")
        return jsonify({"error": "No idea provided"}), 400

    try:
        writer_model = session.get("model", "deepseek-chat")  # Get model from session
        papers_dir = os.path.join(project_root, "generated", "papers")

        try:
            _, _, session_allocation = TinyScientist.resolve_budget_settings(
                session.get("budget"), session.get("budget_preference")
            )
        except ValueError:
            session_allocation = TinyScientist.compute_budget_allocation(None)

        writer = Writer(
            model=writer_model,
            output_dir=papers_dir,
            template="acl",
            s2_api_key=s2_api_key,  # Pass the key here
            cost_tracker=BudgetChecker(
                budget=session_allocation.get("writer"), parent=global_cost_tracker
            ),
        )
        print(f"Writer initialized for this request with model: {writer.model}")
        if not s2_api_key:
            print(
                "Proceeding without Semantic Scholar API key; using fallback sources."
            )

        # Extract the original idea data
        if isinstance(idea_data, dict) and "originalData" in idea_data:
            idea = idea_data["originalData"]
        else:
            idea = idea_data

        print(f"Using pre-configured Writer with model: {writer.model}")

        # Check if this is an experimental idea
        is_experimental = idea.get("is_experimental", False)
        print(f"Idea is experimental: {is_experimental}")

        abs_experiment_dir = None
        if is_experimental and experiment_dir:
            # Convert experiment_dir to absolute path for experimental ideas
            generated_base = os.path.join(project_root, "generated")
            abs_experiment_dir = os.path.abspath(
                os.path.join(generated_base, experiment_dir)
            )

            print(f"Absolute experiment directory: {abs_experiment_dir}")

            # Verify the experiment.py file exists
            experiment_file = os.path.join(abs_experiment_dir, "experiment.py")
            if not os.path.exists(experiment_file):
                raise FileNotFoundError(
                    f"Experiment file not found at: {experiment_file}"
                )

            print(f"Found experiment file: {experiment_file}")

            # Check if experiment_results.txt exists, create placeholder if not
            os.path.join(abs_experiment_dir, "experiment_results.txt")

        elif is_experimental and not experiment_dir:
            print("WARNING: Experimental idea but no experiment_dir provided")
        else:
            print("Non-experimental idea - proceeding without experiment files")

        # Call writer.run() exactly like TinyScientist does
        pdf_path, paper_name = writer.run(idea=idea, experiment_dir=abs_experiment_dir)

        print(
            f"Check the generated paper named as {paper_name} and saved at {pdf_path}"
        )
        print("✅ Paper written.")

        # Convert absolute path to API-accessible path
        generated_base = os.path.join(project_root, "generated")
        if pdf_path.startswith(generated_base):
            relative_path = os.path.relpath(pdf_path, generated_base)
            api_pdf_path = f"/api/files/{relative_path.replace(os.path.sep, '/')}"
        else:
            # Fallback
            api_pdf_path = f"/api/files/papers/{os.path.basename(pdf_path)}"

        response = {
            "pdf_path": api_pdf_path,
            "local_pdf_path": pdf_path,  # Keep original path for debugging
            "paper_name": paper_name,
            "success": True,
        }

        return jsonify(response)

    except Exception as e:
        print(f"ERROR in paper generation: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/files/<path:file_path>", methods=["GET"])
def serve_experiment_file(file_path: str) -> Union[Response, tuple[Response, int]]:
    """Serve generated experiment files"""
    try:
        if demo_cache.enabled:
            generated_base = demo_cache.generated_base
            full_path = demo_cache.resolve_generated_path(file_path)
        else:
            generated_base = os.path.join(project_root, "generated")
            full_path = os.path.abspath(os.path.join(generated_base, file_path))
            if not full_path.startswith(os.path.abspath(generated_base)):
                return jsonify({"error": "Access denied"}), 403

        if not os.path.exists(full_path):
            return jsonify({"error": "File not found"}), 404

        # For text files, return content as JSON
        if file_path.endswith((".py", ".txt", ".md", ".json")):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return jsonify({"content": content})
        else:
            # For other files, serve directly
            return send_file(full_path)

    except DemoCacheError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as e:
        print(f"Error serving file {file_path}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/review", methods=["POST"])
def review_paper() -> Union[Response, tuple[Response, int]]:
    """Review a paper using the Reviewer class"""

    print("📝 Paper review request received")

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    pdf_path = data.get("pdf_path")
    s2_api_key = data.get("s2_api_key")
    idea_id = data.get("idea_id")
    idea_name = data.get("idea_name")

    if demo_cache.enabled:
        try:
            response_payload = demo_cache.get_review_result(
                idea_id=idea_id,
                idea_name=idea_name,
                pdf_path=pdf_path,
            )
        except DemoCacheError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(response_payload)

    if not pdf_path:
        return jsonify({"error": "No PDF path provided"}), 400

    try:
        # Convert API path to absolute path with security checks
        if pdf_path.startswith("/api/files/"):
            # Remove /api/files/ prefix
            relative_path = pdf_path[len("/api/files/") :]
            generated_base = os.path.join(project_root, "generated")
            absolute_pdf_path = os.path.abspath(
                os.path.join(generated_base, relative_path)
            )

            # Security check: ensure the file is within the allowed directory
            if not absolute_pdf_path.startswith(os.path.abspath(generated_base)):
                return (
                    jsonify({"error": "Access denied - path traversal not allowed"}),
                    403,
                )
        else:
            # For security, only allow paths that start with /api/files/
            # This prevents arbitrary file access on the server
            return (
                jsonify({"error": "Invalid path - only /api/files/ paths are allowed"}),
                403,
            )

        print(f"Reviewing paper at: {absolute_pdf_path}")

        # Check if file exists
        if not os.path.exists(absolute_pdf_path):
            return jsonify({"error": "PDF file not found"}), 404
        reviewer_model = session.get("model", "deepseek-chat")  # Get model from session
        print("🔍 Starting paper review...")
        try:
            _, _, session_allocation = TinyScientist.resolve_budget_settings(
                session.get("budget"), session.get("budget_preference")
            )
        except ValueError:
            session_allocation = TinyScientist.compute_budget_allocation(None)

        reviewer = Reviewer(
            model=reviewer_model,
            tools=[],
            num_reviews=1,
            num_reflections=1,
            temperature=0.75,
            s2_api_key=s2_api_key,
            cost_tracker=BudgetChecker(
                budget=session_allocation.get("reviewer"),
                parent=global_cost_tracker,
            ),
        )
        # Call reviewer.review() to get a single review
        review_result = reviewer.review(absolute_pdf_path)

        print("✅ Review completed successfully")

        # Parse the JSON result
        import json

        review_data = json.loads(review_result)

        response = {
            "review": review_data,
            "success": True,
            "message": "Paper review completed successfully",
        }

        return jsonify(response)

    except Exception as e:
        print(f"ERROR in paper review: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    # Configure Flask for long-running requests
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    port = int(os.environ.get("PORT", "5000"))
    socketio.run(
        app,
        debug=True,
        use_reloader=False,
        port=port,
        host="0.0.0.0",
        allow_unsafe_werkzeug=True,
    )
