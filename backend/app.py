import os
import sys
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, request, send_file, session
from flask_cors import CORS

from tiny_scientist.coder import Coder
from tiny_scientist.thinker import Thinker
from tiny_scientist.writer import Writer

# Ensure we import from local tiny_scientist package, not installed one
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


app = Flask(__name__)
app.secret_key = "your-secret-key-here"
CORS(app, supports_credentials=True)


thinker: Optional[Thinker] = None
coder: Optional[Coder] = None
writer: Optional[Writer] = None


def format_name_for_display(name: Optional[str]) -> str:
    """Formats a name"""
    if not name:
        return "Untitled"
    return " ".join(word.capitalize() for word in name.split("_"))


# Initialize the Thinker
@app.route("/api/configure", methods=["POST"])
def configure() -> Response:
    """Configure model and API key"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    model = data.get("model")
    api_key = data.get("api_key")

    if not model or not api_key:
        return jsonify({"error": "Model and API key are required"}), 400

    # Map models to their environment variables
    env_var_map = {
        "deepseek-chat": "DEEPSEEK_API_KEY",
        "deepseek-reasoner": "DEEPSEEK_API_KEY",
        "gpt-4o": "OPENAI_API_KEY",
        "gpt-o1": "OPENAI_API_KEY",
        "claude-3.5-sonnet": "ANTHROPIC_API_KEY",
    }

    # Set the appropriate environment variable
    env_var = env_var_map.get(model)
    if env_var:
        os.environ[env_var] = api_key

    # Store in session
    session["model"] = model
    session["api_key"] = api_key
    session["configured"] = True

    # Initialize all components with same parameters as TinyScientist
    global thinker, coder, writer

    # Use absolute paths outside React's file watching but still accessible
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(current_dir, "generated", "experiments")
    papers_dir = os.path.join(current_dir, "generated", "papers")

    # Ensure directories exist
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(papers_dir, exist_ok=True)

    print(f"Backend directory: {current_dir}")
    print(f"Experiments directory: {os.path.abspath(experiments_dir)}")
    print(f"Papers directory: {os.path.abspath(papers_dir)}")

    thinker = Thinker(
        model=model,
        tools=[],
        iter_num=0,
        output_dir="./",
        search_papers=False,
        generate_exp_plan=True,
    )
    coder = Coder(
        model=model,
        output_dir=experiments_dir,
        max_iters=4,
        max_runs=3,
    )
    writer = Writer(
        model=model,
        output_dir=papers_dir,
        template="acl",
    )
    return jsonify({"status": "configured", "model": model})


@app.route("/api/set-env", methods=["POST"])
def set_environment_variable() -> Response:
    """Set an environment variable"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    key = data.get("key")
    value = data.get("value")

    if not key or not value:
        return jsonify({"error": "Both key and value are required"}), 400

    # Set the environment variable
    os.environ[key] = value

    return jsonify(
        {"status": "success", "message": f"Environment variable {key} set successfully"}
    )


@app.route("/api/generate-initial", methods=["POST"])
def generate_initial() -> Response:
    """Generate initial ideas from an intent (handleAnalysisIntentSubmit)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    intent = data.get("intent")
    num_ideas = data.get("num_ideas", 3)

    # Generate ideas
    ideas = thinker.run(intent=intent, num_ideas=num_ideas)

    # Return in the format expected by TreePlot
    response = {
        "ideas": [
            {
                "title": format_name_for_display(idea.get("Name")),
                "content": format_idea_content(idea),
                "originalData": idea,  # Preserve complete thinker JSON for coder/writer
            }
            for idea in ideas
        ]
    }

    return jsonify(response)


@app.route("/api/set-system-prompt", methods=["POST"])
def set_system_prompt() -> Response:
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


@app.route("/api/set-criteria", methods=["POST"])
def set_criteria() -> Response:
    """Set evaluation criteria for a specific dimension"""
    global thinker

    if not thinker:
        return jsonify({"error": "Thinker not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    dimension = data.get("dimension")  # 'novelty', 'feasibility', or 'impact'
    criteria = data.get("criteria")

    if dimension not in ["novelty", "feasibility", "impact"]:
        return jsonify({"error": "Invalid dimension"}), 400

    # If empty string or None, reset to default
    if not criteria:
        thinker.set_criteria(dimension, None)  # This will reset to default
    else:
        thinker.set_criteria(dimension, criteria)

    return jsonify(
        {"status": "success", "message": f"{dimension.capitalize()} criteria updated"}
    )


@app.route("/api/get-prompts", methods=["GET"])
def get_prompts() -> Response:
    """Get current prompts and criteria"""
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
def generate_children() -> Response:
    """Generate child ideas (generateChildNodes)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
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
                "title": format_name_for_display(idea.get("Name")),
                "content": format_idea_content(idea),
                "originalData": idea,  # Preserve complete thinker JSON for coder/writer
            }
            for idea in ideas
        ]
    }

    return jsonify(response)


@app.route("/api/modify", methods=["POST"])
def modify_idea() -> Response:
    """Modify an idea (modifyHypothesisBasedOnModifications)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
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
def merge_ideas() -> Response:
    """Merge two ideas (mergeHypotheses)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
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
        "title": format_name_for_display(merged_idea.get("Name")),
        "content": format_idea_content(merged_idea),
        "originalData": merged_idea,  # Preserve complete thinker JSON for coder/writer
    }

    return jsonify(response)


@app.route("/api/evaluate", methods=["POST"])
def evaluate_ideas() -> Response:
    """Evaluate ideas (evaluateHypotheses)"""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400
    ideas = data.get("ideas")
    intent = data.get("intent")

    # Debug: Print the incoming ideas to see their structure
    print("DEBUG: Incoming ideas for evaluation:")
    for idea in ideas:
        print(f"ID: {idea.get('id')}, Title: {idea.get('title', idea.get('Title'))}")
        print("---")

    # Use original data directly (no conversion needed)
    thinker_ideas = ideas

    # Store original IDs in order - LLM doesn't preserve titles, use index mapping
    original_ids = [idea.get("id") for idea in ideas]
    print(f"Original IDs in order: {original_ids}")

    # Rank ideas
    scored_ideas = thinker.rank(ideas=thinker_ideas, intent=intent)

    print(f"Number of input ideas: {len(ideas)}")
    print(f"Number of scored ideas: {len(scored_ideas)}")

    # Debug: Print the scored ideas to check if scores are present
    print("DEBUG: Scored ideas:")
    for i, idea in enumerate(scored_ideas):
        title = idea.get("Title", idea.get("title", "No Title"))
        print(f"Index {i}, Title: {title}")
        print(f"NoveltyScore: {idea.get('NoveltyScore')}")
        print(f"FeasibilityScore: {idea.get('FeasibilityScore')}")
        print(f"ImpactScore: {idea.get('ImpactScore')}")
        print("---")

    # Return in the format expected by TreePlot
    # Use index-based mapping since LLM changes titles but preserves order
    response = []
    for i, idea in enumerate(scored_ideas):
        # Use index to map back to original ID
        original_id = original_ids[i] if i < len(original_ids) else f"idea_{i}"
        print(f"Index {i} -> ID: {original_id}")

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

    print("DEBUG: API Response:")
    print(response)
    return jsonify(response)


def format_idea_content(idea: Dict[str, Any]) -> str:
    """Format Thinker idea into content for TreePlot - with standardized section headers"""
    # Get content and ensure no trailing ** in any of the content sections
    description = idea.get("Description", "").strip().rstrip("*")
    importance = idea.get("Importance", "").strip().rstrip("*")
    feasibility = idea.get("Difficulty", "").strip().rstrip("*")
    novelty = idea.get("NoveltyComparison", "").strip().rstrip("*")

    return "\n\n".join(
        [
            f"Description: {description}",
            f"Impact: {importance}",
            f"Feasibility: {feasibility}",
            f"Novelty: {novelty}",
        ]
    )


@app.route("/api/code", methods=["POST"])
def generate_code() -> Response:
    """Generate code synchronously and return when complete"""
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
        print(f"Idea has Experiment field: {'Experiment' in idea}")

        # Call coder.run() exactly like TinyScientist does
        print("Starting coder.run()...")
        print("This may take several minutes - please wait...")

        import sys

        sys.stdout.flush()  # Ensure logs are written immediately

        status, exp_path = coder.run(idea=idea, baseline_results=baseline_results)

        print(f"Coder completed with status: {status}")

        if status:
            print(f"✅ Experiment completed successfully. Results saved at {exp_path}")
        else:
            print(f"❌ Experiment failed. Please check {exp_path} for details.")

        # Clean the experiment directory path for frontend
        clean_exp_path = exp_path
        if clean_exp_path.startswith(os.path.dirname(__file__)):
            # Remove the backend directory prefix
            backend_dir = os.path.dirname(__file__)
            clean_exp_path = os.path.relpath(exp_path, backend_dir)

        response = {
            "status": status,
            "experiment_dir": clean_exp_path,
            "success": status is True,
            "message": (
                "Code generation completed successfully"
                if status
                else "Code generation failed"
            ),
        }

        return jsonify(response)

    except Exception as e:
        print(f"ERROR in code generation: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/write", methods=["POST"])
def generate_paper() -> Response:
    """Generate a paper from an idea using the Writer class"""
    global writer

    print("📝 Paper generation request received")
    print(f"Writer configured: {writer is not None}")

    if writer is None:
        print("ERROR: Writer not configured")
        return jsonify({"error": "Writer not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    print(f"Request data: {data}")

    idea_data = data.get("idea")
    experiment_dir = data.get("experiment_dir", None)

    print("📝 Writing paper...")
    print(f"Idea data: {idea_data}")
    print(f"Idea data keys: {list(idea_data.keys()) if idea_data else 'None'}")
    print(f"Experiment dir: {experiment_dir}")

    if not idea_data:
        print("ERROR: No idea provided in request")
        return jsonify({"error": "No idea provided"}), 400

    try:
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
            if not os.path.isabs(experiment_dir):
                # If relative path, make it relative to backend directory
                backend_dir = os.path.dirname(os.path.abspath(__file__))
                abs_experiment_dir = os.path.join(backend_dir, experiment_dir)
            else:
                abs_experiment_dir = experiment_dir

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
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        if pdf_path.startswith(backend_dir):
            # Remove backend directory prefix and create API path
            relative_path = os.path.relpath(pdf_path, backend_dir)
            api_pdf_path = f"/api/files/{relative_path}"
        else:
            # Fallback: just use the filename if path structure is unexpected
            api_pdf_path = f"/api/files/generated/papers/{os.path.basename(pdf_path)}"

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
def serve_experiment_file(file_path: str) -> Response:
    """Serve generated experiment files"""
    try:
        # Get the backend directory to construct full paths
        backend_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct full path relative to backend directory
        if file_path.startswith("generated/"):
            full_path = os.path.join(backend_dir, file_path)
        else:
            # For backward compatibility, assume it's in generated folder
            full_path = os.path.join(backend_dir, "generated", file_path)

        full_path = os.path.abspath(full_path)
        allowed_base = os.path.abspath(os.path.join(backend_dir, "generated"))

        # Security check: ensure the file is within the allowed directory
        if not full_path.startswith(allowed_base):
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


if __name__ == "__main__":
    # Configure Flask for long-running requests
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.run(debug=True, port=8080, host="0.0.0.0", threaded=True)
