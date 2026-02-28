import builtins
import io
import json
import os
import sys
import uuid
from typing import Any, Dict, Optional, Union

from flask import Flask, Response, jsonify, request, send_file, session
from flask_cors import CORS
from flask_socketio import SocketIO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

original_print = builtins.print
original_stdout = sys.stdout

# Buffer to store messages until socketio is ready
log_buffer = []


class WebSocketCapture(io.StringIO):
    def write(self, text: str) -> int:
        # Also write to original stdout
        original_stdout.write(text)
        # Store for WebSocket emission
        if text.strip():  # Only non-empty messages
            log_buffer.append(text.strip())
        return len(text)


def websocket_print(*args: Any, **kwargs: Any) -> None:
    # Call original print
    original_print(*args, **kwargs)
    # Also emit via WebSocket in real-time
    message = " ".join(str(arg) for arg in args)
    if message.strip():
        emit_log_realtime(message.strip())


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


# Create a function to emit buffered logs when socketio is ready
def emit_buffered_logs() -> None:
    global log_buffer
    try:
        for message in log_buffer:
            socketio.emit(
                "log",
                {
                    "message": message,
                    "level": "info",
                    "timestamp": __import__("time").time(),
                },
            )
        log_buffer = []  # Clear buffer after emitting
    except Exception:
        pass


# Create a function to emit logs in real-time
def emit_log_realtime(message: str, level: str = "info") -> None:
    try:
        # Check if socketio is available
        if "socketio" in globals():
            socketio.emit(
                "log",
                {
                    "message": message,
                    "level": level,
                    "timestamp": __import__("time").time(),
                },
            )
    except Exception:
        pass


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
socketio = SocketIO(app, cors_allowed_origins="*")

# Print override is now active
print("🚀 Backend server starting with WebSocket logging enabled!")


thinker: Optional[Thinker] = None
coder: Optional[Coder] = None
writer: Optional[Writer] = None
reviewer: Optional[Reviewer] = None
global_cost_tracker: Optional[BudgetChecker] = None

global_idea_storage: Dict[str, Any] = {}

# Hierarchical ID counters
# root_counter: next number for root-level ideas (1, 2, 3, ...)
# child_counters: {parent_id: next_child_number}
# modify_counters: {parent_id: next_X_number}
_root_counter: int = 0
_child_counters: Dict[str, int] = {}
_modify_counters: Dict[str, int] = {}


def _next_root_id() -> str:
    global _root_counter
    _root_counter += 1
    return str(_root_counter)


def _next_child_id(parent_id: str) -> str:
    if parent_id not in _child_counters:
        _child_counters[parent_id] = 0
    _child_counters[parent_id] += 1
    return f"{parent_id}-{_child_counters[parent_id]}"


def _next_modify_id(parent_id: str) -> str:
    if parent_id not in _modify_counters:
        _modify_counters[parent_id] = 0
    _modify_counters[parent_id] += 1
    return f"{parent_id}-X{_modify_counters[parent_id]}"


def _merge_id(id_a: str, id_b: str) -> str:
    return f"{id_a}-Y-{id_b}-Y"


def _reset_id_counters() -> None:
    global _root_counter, _child_counters, _modify_counters
    _root_counter = 0
    _child_counters = {}
    _modify_counters = {}


def _store_or_update_idea(idea: Dict[str, Any]) -> None:
    iid = idea.get("id")
    if not iid:
        return
    existing = global_idea_storage.get(iid, {})
    merged = {**existing, **idea}
    global_idea_storage[iid] = merged


def _list_stored_ideas() -> list[Dict[str, Any]]:
    return list(global_idea_storage.values())


def _bulk_update_scores(scored_payloads: list[Dict[str, Any]]) -> None:
    for s in scored_payloads:
        iid = s.get("id")
        if not iid or iid not in global_idea_storage:
            continue

        update_dict: Dict[str, Any] = {}

        incoming_scores = s.get("scores")
        if incoming_scores and isinstance(incoming_scores, dict):
            existing_scores = global_idea_storage[iid].get("scores")
            if not isinstance(existing_scores, dict):
                existing_scores = {}
            update_dict["scores"] = {**existing_scores, **incoming_scores}

        # Persist the most recent dimension scores/reasons as convenience fields
        # (the canonical per-dimension values remain stored in `scores`).
        for idx in (1, 2, 3):
            score_key_in = f"dimension{idx}Score"
            score_key_out = f"Dimension{idx}Score"
            reason_key_out = f"Dimension{idx}Reason"
            if s.get(score_key_in) is not None:
                update_dict[score_key_out] = s.get(score_key_in)
            if s.get(reason_key_out) is not None:
                update_dict[reason_key_out] = s.get(reason_key_out, "")

        global_idea_storage[iid].update(update_dict)


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
        "gpt-5.2": "OPENAI_API_KEY",
        "gpt-5.2-pro": "OPENAI_API_KEY",
        "gpt-5-mini": "OPENAI_API_KEY",
        "claude-opus-4-6": "ANTHROPIC_API_KEY",
        "claude-sonnet-4-5": "ANTHROPIC_API_KEY",
        "deepseek-chat": "DEEPSEEK_API_KEY",
        "deepseek-reasoner": "DEEPSEEK_API_KEY",
        "gemini-3-pro": "GOOGLE_API_KEY",
        "gemini-3-flash": "GOOGLE_API_KEY",
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
    emit_buffered_logs()
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    intent = data.get("intent")

    idea = thinker.run(intent=intent, num_ideas=1)

    if not idea or not isinstance(idea, dict):
        return jsonify({"error": "Failed to generate idea"}), 500

    # Ensure experiment plan is generated for experimental ideas (thinker.run may skip it on failure)
    if idea.get("is_experimental", True) and not idea.get("Experiment"):
        enriched_json = thinker.generate_experiment_plan(json.dumps(idea))
        idea = json.loads(enriched_json)

    print(f"[generate-initial] Returning idea keys: {list(idea.keys())}")
    print(f"[generate-initial] Has Experiment: {'Experiment' in idea}")

    new_id = _next_root_id()
    response = {
        "ideas": [
            {
                "id": new_id,
                "title": format_name_for_display(idea.get("Name")),
                "content": format_idea_content(idea),
                "originalData": idea,
            }
        ]
    }

    return jsonify(response)


@app.route("/api/set-system-prompt", methods=["POST"])
def set_system_prompt() -> Union[Response, tuple[Response, int]]:
    """Set the system prompt for the Thinker"""
    global thinker

    if not thinker:
        return jsonify({"error": "Thinker not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    system_prompt = data.get("system_prompt")

    # If empty string or None, reset to default
    if not system_prompt:
        thinker.set_system_prompt(None)  # This will reset to default
    else:
        thinker.set_system_prompt(system_prompt)

    return jsonify({"status": "success", "message": "System prompt updated"})


@app.route("/api/get-prompts", methods=["GET"])
def get_prompts() -> Union[Response, tuple[Response, int]]:
    """Get current system prompt (dynamic dimensions are configured client-side)."""
    global thinker

    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    return jsonify(
        {
            "system_prompt": thinker.get_system_prompt(),
            "defaults": {
                "system_prompt": thinker.default_system_prompt,
            },
        }
    )


@app.route("/api/generate-children", methods=["POST"])
def generate_children() -> Union[Response, tuple[Response, int]]:
    """Generate child ideas (generateChildNodes)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    parent_content = data.get("parent_content")
    parent_id = data.get("parent_id", "root")
    context = data.get("context", "")

    combined_intent = f"{parent_content}\nAdditional Context: {context}"
    idea = thinker.run(intent=combined_intent, num_ideas=1)

    if not idea or not isinstance(idea, dict):
        return jsonify({"error": "Failed to generate idea"}), 500

    # Ensure experiment plan is generated for experimental ideas (thinker.run may skip it on failure)
    if idea.get("is_experimental", True) and not idea.get("Experiment"):
        enriched_json = thinker.generate_experiment_plan(json.dumps(idea))
        idea = json.loads(enriched_json)

    child_id = _next_child_id(parent_id)

    response = {
        "ideas": [
            {
                "id": child_id,
                "title": format_name_for_display(idea.get("Name")),
                "content": format_idea_content(idea),
                "originalData": idea,
            }
        ]
    }

    return jsonify(response)


@app.route("/api/suggest-dimensions", methods=["POST"])
def suggest_dimensions() -> Union[Response, tuple[Response, int]]:
    """Suggest dimension pairs for evaluating ideas based on research intent."""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    intent = data.get("intent")
    if not intent:
        return jsonify({"error": "Intent is required"}), 400

    dimension_pairs = thinker.suggest_dimensions(intent=intent)
    return jsonify({"dimension_pairs": dimension_pairs})


@app.route("/api/evaluate-dimension", methods=["POST"])
def evaluate_single_dimension() -> Union[Response, tuple[Response, int]]:
    """Evaluate ideas on a single dimension pair only."""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    ideas = data.get("ideas", [])
    intent = data.get("intent")
    dimension_pair = data.get("dimension_pair")
    dimension_index = data.get("dimension_index", 0)

    print("[DEBUG] Single dimension evaluation request:")
    print(f"  - Ideas count: {len(ideas)}")
    print(f"  - Dimension pair: {dimension_pair}")
    print(f"  - Dimension index: {dimension_index}")

    if not dimension_pair:
        return jsonify({"error": "dimension_pair is required"}), 400

    try:
        scores = thinker.rank_single_dimension(
            ideas=ideas,
            intent=intent,
            dimension_pair=dimension_pair,
            dimension_index=dimension_index,
        )
        print(f"[DEBUG] Single dimension evaluation returned {len(scores)} scores")
        return jsonify(
            {
                "scores": scores,
                "dimension_pair": dimension_pair,
                "dimension_index": dimension_index,
            }
        )
    except Exception as e:
        print(f"[ERROR] Single dimension evaluation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return (
            jsonify({"error": f"Error during single dimension evaluation: {str(e)}"}),
            500,
        )


@app.route("/api/modify", methods=["POST"])
def modify_idea() -> Union[Response, tuple[Response, int]]:
    """Modify an idea (modifyIdeaBasedOnModifications)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    original_idea = data.get("original_idea")
    modifications = data.get("modifications")
    behind_idea = data.get("behind_idea")
    original_id = data.get("original_id", "")
    dimension_pairs = data.get("dimension_pairs") or []

    if not isinstance(modifications, list) or len(modifications) == 0:
        return jsonify({"error": "modifications must be a non-empty list"}), 400

    if not isinstance(dimension_pairs, list) or len(dimension_pairs) < 3:
        return jsonify({"error": "dimension_pairs (length >= 3) is required"}), 400

    if not isinstance(original_idea, dict):
        return jsonify({"error": "original_idea must be an object"}), 400
    if not isinstance(modifications, list) or not modifications:
        return jsonify({"error": "modifications must be a non-empty list"}), 400

    # Use original data directly (no conversion needed)
    thinker_original = original_idea
    thinker_behind = behind_idea
    thinker_mods = [
        mod
        for mod in modifications
        if isinstance(mod, dict) and isinstance(mod.get("metric"), str)
    ]

    if not thinker_mods:
        return jsonify({"error": "No valid modifications provided"}), 400

    # Modify the idea
    modified_idea = thinker.modify_idea(
        original_idea=thinker_original,
        modifications=thinker_mods,
        behind_idea=thinker_behind,
        dimension_pairs=dimension_pairs,
    )

    if not isinstance(modified_idea, dict):
        return (
            jsonify(
                {
                    "error": "Failed to modify idea after retries",
                    "details": "thinker.modify_idea returned no valid idea",
                }
            ),
            502,
        )

    # Apply the requested score adjustments onto the modified idea's dynamic score map.
    # This keeps the UI consistent even though modify_idea is a content rewrite step.
    base_scores: Dict[str, Any] = {}
    if isinstance(thinker_original, dict) and isinstance(
        thinker_original.get("scores"), dict
    ):
        base_scores.update(thinker_original.get("scores") or {})
    if isinstance(modified_idea, dict) and isinstance(
        modified_idea.get("scores"), dict
    ):
        base_scores.update(modified_idea.get("scores") or {})

    for mod in modifications:
        if not isinstance(mod, dict):
            continue
        metric = mod.get("metric")
        new_score = mod.get("newScore")
        if isinstance(metric, str) and metric.strip() and new_score is not None:
            base_scores[metric] = new_score

    if isinstance(modified_idea, dict):
        modified_idea["scores"] = base_scores

        def _pair_key(pair: Dict[str, Any]) -> str:
            return f"{str(pair.get('dimensionA', '')).strip()}-{str(pair.get('dimensionB', '')).strip()}"

        for idx, pair in enumerate(dimension_pairs[:3]):
            key = _pair_key(pair)
            score_val = base_scores.get(key)
            modified_idea[f"Dimension{idx + 1}Score"] = score_val

    # Generate experiment plan for modified idea (only if experimental)
    if modified_idea.get("is_experimental", True) and not modified_idea.get(
        "Experiment"
    ):
        enriched_json = thinker.generate_experiment_plan(json.dumps(modified_idea))
        modified_idea = json.loads(enriched_json)

    # Generate hierarchical ID for the modified idea
    modified_id = _next_modify_id(original_id) if original_id else str(uuid.uuid4())

    # Return in the format expected by TreePlot
    response = {
        "id": modified_id,
        "title": format_name_for_display(modified_idea.get("Name")),
        "content": format_idea_content(modified_idea),
        "originalData": modified_idea,  # Preserve complete thinker JSON for coder/writer
        "scores": modified_idea.get("scores")
        if isinstance(modified_idea, dict)
        else {},
        "dimension1Score": (
            modified_idea.get("Dimension1Score")
            if isinstance(modified_idea, dict)
            else None
        ),
        "dimension2Score": (
            modified_idea.get("Dimension2Score")
            if isinstance(modified_idea, dict)
            else None
        ),
        "dimension3Score": (
            modified_idea.get("Dimension3Score")
            if isinstance(modified_idea, dict)
            else None
        ),
        "Dimension1Reason": (
            modified_idea.get("Dimension1Reason", "")
            if isinstance(modified_idea, dict)
            else ""
        ),
        "Dimension2Reason": (
            modified_idea.get("Dimension2Reason", "")
            if isinstance(modified_idea, dict)
            else ""
        ),
        "Dimension3Reason": (
            modified_idea.get("Dimension3Reason", "")
            if isinstance(modified_idea, dict)
            else ""
        ),
    }
    return jsonify(response)


@app.route("/api/merge", methods=["POST"])
def merge_ideas() -> Union[Response, tuple[Response, int]]:
    """Merge two ideas (mergeIdeas)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    idea_a = data.get("idea_a")
    idea_b = data.get("idea_b")
    idea_a_id = data.get("idea_a_id", "")
    idea_b_id = data.get("idea_b_id", "")
    # Use original data directly (no conversion needed)
    thinker_idea_a = idea_a
    thinker_idea_b = idea_b
    # Merge ideas
    merged_idea = thinker.merge_ideas(idea_a=thinker_idea_a, idea_b=thinker_idea_b)

    # Generate experiment plan for merged idea (only if experimental)
    if (
        isinstance(merged_idea, dict)
        and merged_idea.get("is_experimental", True)
        and not merged_idea.get("Experiment")
    ):
        enriched_json = thinker.generate_experiment_plan(json.dumps(merged_idea))
        merged_idea = json.loads(enriched_json)

    # Generate hierarchical merged ID
    if idea_a_id and idea_b_id:
        merged_id = _merge_id(idea_a_id, idea_b_id)
    else:
        merged_id = str(uuid.uuid4())

    # Return in the format expected by TreePlot
    response = {
        "id": merged_id,
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
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    incoming_ideas = data.get("ideas", [])
    intent = data.get("intent")
    dimension_pairs = data.get("dimension_pairs", [])
    user_score_corrections = data.get("userScoreCorrections", [])
    mode = data.get("mode", "incremental")
    explicit_target_ids = set(data.get("targetIds", []) or [])

    if not isinstance(dimension_pairs, list) or len(dimension_pairs) < 3:
        return (
            jsonify({"error": "dimension_pairs (length >= 3) is required"}),
            400,
        )

    def _pair_keys(pair: Dict[str, Any]) -> tuple[str, str]:
        a = str(pair.get("dimensionA", "")).strip()
        b = str(pair.get("dimensionB", "")).strip()
        return f"{a}-{b}", f"{b}-{a}"

    def _has_score_for_pair(idea: Dict[str, Any], pair: Dict[str, Any]) -> bool:
        scores = idea.get("scores")
        if not isinstance(scores, dict):
            return False
        k1, k2 = _pair_keys(pair)
        v1 = scores.get(k1)
        v2 = scores.get(k2)
        return v1 is not None or v2 is not None

    def _has_scores_for_pairs(
        idea: Dict[str, Any], pairs: list[Dict[str, Any]]
    ) -> bool:
        return all(_has_score_for_pair(idea, pair) for pair in pairs)

    print("[DEBUG] Evaluation request received:")
    print(f"  - Mode: {mode}")
    print(f"  - Incoming ideas count: {len(incoming_ideas)}")
    print(f"  - Incoming idea IDs: {[i.get('id', 'NO_ID') for i in incoming_ideas]}")
    print(f"  - Intent: {intent[:50] if intent else 'None'}...")
    print(f"  - Target IDs: {list(explicit_target_ids)}")
    print(f"  - Dimension pairs: {len(dimension_pairs)}")

    existing_ids = {i["id"] for i in _list_stored_ideas()}
    new_ids = []
    for inc in incoming_ideas:
        inc_id = inc.get("id") or str(uuid.uuid4())
        inc["id"] = inc_id
        if inc_id not in existing_ids:
            new_ids.append(inc_id)
        _store_or_update_idea({**inc})

    stored_list = _list_stored_ideas()

    if mode == "full":
        target_ids_local = {idea.get("id") for idea in stored_list}
    else:
        target_ids_local = set(new_ids) | explicit_target_ids
        if not explicit_target_ids:
            for idea in stored_list:
                if not _has_scores_for_pairs(idea, dimension_pairs[:3]):
                    target_ids_local.add(idea.get("id"))

    if mode == "full":
        evaluation_input = stored_list
    else:
        evaluation_input = []
        for idea in stored_list:
            iid = idea.get("id")
            has_scores = _has_scores_for_pairs(idea, dimension_pairs[:3])
            idea_copy = idea.copy()
            if iid in target_ids_local:
                idea_copy.pop("AlreadyScored", None)
                evaluation_input.append(idea_copy)
            else:
                if has_scores:
                    idea_copy["AlreadyScored"] = True
                evaluation_input.append(idea_copy)

    print("[DEBUG] About to call thinker.rank with:")
    print(f"  - Evaluation input count: {len(evaluation_input)}")
    print(f"  - Evaluation input IDs: {[i.get('id') for i in evaluation_input]}")
    print(f"  - Intent length: {len(intent) if intent else 0}")
    print(f"  - Partial mode: {mode != 'full'}")

    try:
        scored_ideas = thinker.rank(
            ideas=evaluation_input,
            intent=intent,
            dimension_pairs=dimension_pairs if dimension_pairs else None,
            user_score_corrections=user_score_corrections,
            partial=(mode != "full"),
        )
        print(
            f"[DEBUG] thinker.rank returned {len(scored_ideas) if scored_ideas else 0} scored ideas"
        )
        if scored_ideas:
            print(
                f"[DEBUG] First scored idea: {scored_ideas[0].get('Title', 'NO_TITLE')}"
            )
    except Exception as e:
        print(f"[ERROR] thinker.rank failed: {str(e)}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback

        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        scored_ideas = []

    title_maps = [{}, {}, {}]
    for idea in stored_list:
        iid = idea.get("id")
        if not iid:
            continue

        for key in [idea.get("Title"), idea.get("Name"), idea.get("title")]:
            if key and key.strip():
                title_maps[0][key.strip()] = iid

        for raw_name in [idea.get("Name"), idea.get("Title")]:
            if raw_name and raw_name.strip():
                formatted = format_name_for_display(raw_name)
                title_maps[1][formatted] = iid

        for key in [idea.get("Title"), idea.get("Name")]:
            if key and key.strip():
                variations = [
                    key.strip().lower(),
                    key.strip().replace(" ", "_").lower(),
                    key.strip().replace("_", " ").lower(),
                ]
                for variation in variations:
                    title_maps[2][variation] = iid

    print("[DEBUG] Title maps built:")
    print(f"  - Map 0 (original): {list(title_maps[0].keys())}")
    print(f"  - Map 1 (formatted): {list(title_maps[1].keys())}")
    print(f"  - Map 2 (variations): {list(title_maps[2].keys())}")

    def find_idea_id(scored_idea):
        scored_titles = [
            scored_idea.get("Title", "").strip(),
            scored_idea.get("Name", "").strip(),
            scored_idea.get("title", "").strip(),
        ]

        print(f"[DEBUG] Finding ID for scored idea with titles: {scored_titles}")

        for title in scored_titles:
            if not title:
                continue

            for i, title_map in enumerate(title_maps):
                if title in title_map:
                    print(
                        f"[DEBUG] Found exact match in map {i}: '{title}' -> {title_map[title]}"
                    )
                    return title_map[title]

            title_lower = title.lower()
            for i, title_map in enumerate(title_maps):
                for stored_title, stored_id in title_map.items():
                    if stored_title.lower() == title_lower:
                        print(
                            f"[DEBUG] Found case-insensitive match in map {i}: '{title}' -> {stored_id}"
                        )
                        return stored_id

        print(f"[DEBUG] No match found for titles: {scored_titles}")
        return None

    payloads = []
    unmatched_ideas = []
    matched_ids = set()

    pair_count = len(dimension_pairs)

    for i, scored in enumerate(scored_ideas):
        iid = find_idea_id(scored)

        if not iid and mode == "incremental" and i < len(new_ids):
            potential_id = new_ids[i]
            if potential_id in target_ids_local and potential_id not in matched_ids:
                iid = potential_id
                print(f"[DEBUG] Fallback ID match: scored[{i}] -> {iid}")

        if not iid:
            unmatched_ideas.append(
                {
                    "title": scored.get("Title", scored.get("Name", "Unknown")),
                    "scored_keys": list(scored.keys()),
                    "index": i,
                }
            )
            continue

        if mode == "incremental" and iid not in target_ids_local:
            continue

        matched_ids.add(iid)
        if iid and ("-CUSTOM-" in iid or iid.startswith("C-")):
            print(f"[DEBUG] Custom idea {iid} reasoning fields:")
            print(f"  Dimension1Reason: {scored.get('Dimension1Reason', '')}")
            print(f"  Dimension2Reason: {scored.get('Dimension2Reason', '')}")
            if pair_count >= 3:
                print(f"  Dimension3Reason: {scored.get('Dimension3Reason', '')}")
            print(f"  All scored keys: {list(scored.keys())}")

        payload = {
            "id": iid,
            "scores": scored.get("scores", {})
            if isinstance(scored.get("scores"), dict)
            else {},
        }
        payload["dimension1Score"] = scored.get("Dimension1Score")
        payload["dimension2Score"] = scored.get("Dimension2Score")
        payload["Dimension1Reason"] = scored.get("Dimension1Reason", "")
        payload["Dimension2Reason"] = scored.get("Dimension2Reason", "")
        if pair_count >= 3:
            payload["dimension3Score"] = scored.get("Dimension3Score")
            payload["Dimension3Reason"] = scored.get("Dimension3Reason", "")
        payloads.append(payload)

    print("[DEBUG] Evaluation matching results:")
    print(f"  - Mode: {mode}")
    print(f"  - Stored ideas: {len(stored_list)}")
    print(f"  - Scored ideas from LLM: {len(scored_ideas)}")
    print(f"  - Target IDs: {list(target_ids_local)}")
    print(f"  - New IDs: {new_ids}")
    print(f"  - Successful matches: {len(payloads)}")
    print(f"  - Failed matches: {len(unmatched_ideas)}")

    if unmatched_ideas:
        print(f"[WARNING] Could not match {len(unmatched_ideas)} ideas:")
        for um in unmatched_ideas[:3]:
            print(
                f"  - [{um.get('index', '?')}] Title: '{um['title']}', Keys: {um['scored_keys']}"
            )
        print(f"[DEBUG] Available stored titles: {list(title_maps[0].keys())}")

    if payloads:
        print("[DEBUG] Matched payloads:")
        for p in payloads[:2]:
            print(
                f"  - ID: {p['id']}, Scores: D1={p.get('dimension1Score')}, D2={p.get('dimension2Score')}, D3={p.get('dimension3Score')}"
            )

    _bulk_update_scores(payloads)

    updated_all = _list_stored_ideas()

    client_return = []
    for i in updated_all:
        item = {
            "id": i.get("id"),
            "title": format_name_for_display(i.get("Name") or i.get("Title")),
            "content": format_idea_content(i),
            "originalData": i,
        }

        if i.get("scores"):
            item["scores"] = i.get("scores")
            item["dimension1Score"] = i.get("Dimension1Score")
            item["dimension2Score"] = i.get("Dimension2Score")
            item["dimension3Score"] = i.get("Dimension3Score")
            item["Dimension1Reason"] = i.get("Dimension1Reason", "")
            item["Dimension2Reason"] = i.get("Dimension2Reason", "")
            item["Dimension3Reason"] = i.get("Dimension3Reason", "")

        client_return.append(item)

    custom_ideas_in_return = [
        item
        for item in client_return
        if item["id"] and ("-CUSTOM-" in item["id"] or item["id"].startswith("C-"))
    ]
    if custom_ideas_in_return:
        print("[DEBUG] Custom ideas in client return:")
        for item in custom_ideas_in_return:
            print(f"  - ID: {item['id']}")
            print(
                f"    D1Reason: {item.get('Dimension1Reason', 'MISSING')[:50]}..., D2Reason: {item.get('Dimension2Reason', 'MISSING')[:50]}..., D3Reason: {item.get('Dimension3Reason', 'MISSING')[:50]}..."
            )

    meta = {
        "mode": mode,
        "scoredCount": len(payloads),
        "totalIdeas": len(updated_all),
        "targets": list(target_ids_local),
    }
    return jsonify({"ideas": client_return, "meta": meta})


def format_idea_content(idea: Union[Dict[str, Any], str]) -> str:
    """
    Formats the Thinker's idea JSON into a single content string for the frontend.
    This is now simplified, as the frontend will pull most data from originalData.
    We only format the main sections for basic display.
    """
    if isinstance(idea, str):
        return idea

    problem = idea.get("Problem", "") or idea.get("Description", "")
    importance = idea.get("Importance", "")
    difficulty = idea.get("Difficulty", "")
    novelty = idea.get("NoveltyComparison", "")

    content_sections = [
        f"**Problem:**\n{problem}",
        f"**Importance:**\n{importance}",
        f"**Difficulty:**\n{difficulty}",
        f"**Novelty Comparison:**\n{novelty}",
    ]

    return "\n\n".join(content_sections)


@app.route("/api/code", methods=["POST"])
def generate_code() -> Union[Response, tuple[Response, int]]:
    """Generate code synchronously and return when complete"""
    emit_buffered_logs()  # Emit any buffered logs
    global coder

    if coder is None:
        return jsonify({"error": "Coder not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    idea_data = data.get("idea")
    baseline_results = data.get("baseline_results", {})

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

    s2_api_key = data.get("s2_api_key", None) or os.environ.get("S2_API_KEY")

    if not idea_data:
        print("ERROR: No idea provided in request")
        return jsonify({"error": "No idea provided"}), 400

    try:
        writer_model = (
            data.get("model")
            or session.get("model")
            or os.environ.get("DEMO_CACHE_MODEL")
            or "gpt-5-mini"
        )
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

        # Extract the original idea data
        if isinstance(idea_data, dict) and "originalData" in idea_data:
            idea = idea_data["originalData"]
        else:
            idea = idea_data

        if not isinstance(idea, dict):
            return jsonify({"error": "idea must be an object"}), 400

        print(f"Using pre-configured Writer with model: {writer.model}")

        # Check if this is an experimental idea
        is_experimental = bool(idea.get("is_experimental", experiment_dir is not None))
        print(f"Idea is experimental: {is_experimental}")

        writer_idea = {**idea, "is_experimental": is_experimental}

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
        pdf_path, paper_name = writer.run(
            idea=writer_idea, experiment_dir=abs_experiment_dir
        )

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
        # The base directory for all generated content
        generated_base = os.path.join(project_root, "generated")

        # Construct the full path securely
        full_path = os.path.abspath(os.path.join(generated_base, file_path))

        # Security check: ensure the file is within the allowed directory
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
    s2_api_key = data.get("s2_api_key") or os.environ.get("S2_API_KEY")

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
        reviewer_model = (
            data.get("model")
            or session.get("model")
            or os.environ.get("DEMO_CACHE_MODEL")
            or "gpt-5-mini"
        )
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


@app.route("/api/clear-session", methods=["POST"])
def clear_session() -> Union[Response, tuple[Response, int]]:
    """Clear session state and reset ID counters for a fresh start."""
    global global_idea_storage
    global_idea_storage = {}
    _reset_id_counters()
    session.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    # Configure Flask for long-running requests
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    socketio.run(
        app,
        debug=True,
        use_reloader=False,
        port=5000,
        host="0.0.0.0",
        allow_unsafe_werkzeug=True,
    )
