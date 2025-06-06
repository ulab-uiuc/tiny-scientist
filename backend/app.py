import os
from typing import Any, Dict, Optional, Union

from flask import Flask, Response, jsonify, request, session
from flask_cors import CORS

from tiny_scientist.thinker import Thinker

app = Flask(__name__)
app.secret_key = "your-secret-key-here"
CORS(app, supports_credentials=True)


thinker: Optional[Thinker] = None


# Initialize the Thinker
@app.route("/api/configure", methods=["POST"])  # type: ignore[misc]
def configure() -> Union[Response, tuple[Response, int]]:
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

    # Initialize thinker with new model
    global thinker
    thinker = Thinker(
        model=model,
        tools=[],
        iter_num=0,
        output_dir="./",
        search_papers=False,
        generate_exp_plan=False,
    )

    return jsonify({"status": "configured", "model": model})


@app.route("/api/generate-initial", methods=["POST"])  # type: ignore[misc]
def generate_initial() -> Union[Response, tuple[Response, int]]:
    """Generate initial ideas from an intent (handleAnalysisIntentSubmit)"""
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    intent = data.get("intent")
    num_ideas = data.get("num_ideas", 3)

    # Generate ideas
    ideas = thinker.run(intent=intent, num_ideas=num_ideas)

    # Return in the format expected by TreePlot
    response = {
        "ideas": [
            {
                "title": (
                    idea.get("Title", idea.get("Name", "Untitled"))
                    if isinstance(idea, dict)
                    else "Untitled"
                ),
                "content": format_idea_content(idea),
            }
            for idea in ideas
        ]
    }

    return jsonify(response)


@app.route("/api/generate-children", methods=["POST"])  # type: ignore[misc]
def generate_children() -> Union[Response, tuple[Response, int]]:
    """Generate child ideas (generateChildNodes)"""
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    parent_content = data.get("parent_content")
    context = data.get("context", "")

    # Combine parent content and context as the intent
    combined_intent = f"{parent_content}\nAdditional Context: {context}"
    ideas = thinker.run(intent=combined_intent, num_ideas=3)

    # Return in the format expected by TreePlot
    response = {
        "ideas": [
            {
                "title": (
                    idea.get("Title", idea.get("Name", "Untitled"))
                    if isinstance(idea, dict)
                    else "Untitled"
                ),
                "content": format_idea_content(idea),
            }
            for idea in ideas
        ]
    }

    return jsonify(response)


@app.route("/api/modify", methods=["POST"])  # type: ignore[misc]
def modify_idea() -> Union[Response, tuple[Response, int]]:
    """Modify an idea (modifyHypothesisBasedOnModifications)"""
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    original_idea = data.get("original_idea")
    modifications = data.get("modifications")
    behind_idea = data.get("behind_idea")

    # Convert TreePlot format to Thinker format
    thinker_original = convert_to_thinker_format(original_idea)
    thinker_behind = convert_to_thinker_format(behind_idea) if behind_idea else None
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
        "title": (
            modified_idea.get("Title", modified_idea.get("Name", "Untitled"))
            if modified_idea
            else "Untitled"
        ),
        "content": format_idea_content(modified_idea),
    }
    return jsonify(response)


@app.route("/api/merge", methods=["POST"])  # type: ignore[misc]
def merge_ideas() -> Union[Response, tuple[Response, int]]:
    """Merge two ideas (mergeHypotheses)"""
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    idea_a = data.get("idea_a")
    idea_b = data.get("idea_b")
    # Convert TreePlot format to Thinker format
    thinker_idea_a = convert_to_thinker_format(idea_a)
    thinker_idea_b = convert_to_thinker_format(idea_b)
    # Merge ideas
    merged_idea = thinker.merge_ideas(idea_a=thinker_idea_a, idea_b=thinker_idea_b)

    # Return in the format expected by TreePlot
    response = {
        "title": (
            merged_idea.get("Title", merged_idea.get("Name", "Untitled"))
            if merged_idea
            else "Untitled"
        ),
        "content": format_idea_content(merged_idea),
    }

    return jsonify(response)


@app.route("/api/evaluate", methods=["POST"])  # type: ignore[misc]
def evaluate_ideas() -> Union[Response, tuple[Response, int]]:
    """Evaluate ideas (evaluateHypotheses)"""
    if thinker is None:
        return jsonify({"error": "Thinker not configured"}), 400

    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    ideas = data.get("ideas")
    intent = data.get("intent")

    # Convert TreePlot format to Thinker format
    thinker_ideas = [convert_to_thinker_format(idea) for idea in ideas]

    # Rank ideas
    ranked_ideas = thinker.rank(ideas=thinker_ideas, intent=intent)

    # Return in the format expected by TreePlot
    # Include rankings that TreePlot will convert to scores
    response = []
    for idea in ranked_ideas:
        response.append(
            {
                "id": idea.get("id"),
                "novelty_rank": idea.get("NoveltyRanking"),
                "novelty_rank_reason": idea.get("NoveltyReason", ""),
                "feasibility_rank": idea.get("FeasibilityRanking"),
                "feasibility_rank_reason": idea.get("FeasibilityReason", ""),
                "impact_rank": idea.get("ImpactRanking"),
                "impact_rank_reason": idea.get("ImpactReason", ""),
            }
        )

    return jsonify(response)


def format_idea_content(idea: Any) -> str:
    """Format Thinker idea into content for TreePlot - with standardized section headers"""
    if not isinstance(idea, dict):
        return "No content available"

    # Get content and ensure no trailing ** in any of the content sections
    problem = idea.get("Problem", "").strip().rstrip("*")
    importance = idea.get("Importance", "").strip().rstrip("*")
    feasibility = idea.get("Difficulty", "").strip().rstrip("*")
    novelty = idea.get("NoveltyComparison", "").strip().rstrip("*")

    return "\n\n".join(
        [
            f"Problem: {problem}",
            f"Impact: {importance}",
            f"Feasibility: {feasibility}",
            f"Novelty: {novelty}",
        ]
    )


def convert_to_thinker_format(treeplot_idea: Any) -> Dict[str, Any]:
    """Convert TreePlot idea format to Thinker format"""
    if not isinstance(treeplot_idea, dict):
        return {}

    # Extract sections from content if possible
    content = treeplot_idea.get("content", "")

    problem = ""
    importance = ""
    difficulty = ""  # Maps to Feasibility in frontend
    novelty_comparison = ""  # Maps to Novelty in frontend
    approach = ""

    # Try to extract sections with more flexible pattern matching
    if content:
        sections = content.split("\n\n")
        for section in sections:
            # Remove all formatting variations and normalize
            section_lower = section.lower()

            if "problem" in section_lower:
                # Extract content after any form of "Problem:" heading
                problem = extract_section_content(section)
            if "impact" in section_lower:
                importance = extract_section_content(section)
            if "feasibility" in section_lower:
                # In frontend it's called Feasibility, in backend it's Difficulty
                difficulty = extract_section_content(section)
            if "novelty" in section_lower:
                # In frontend it can be Novelty or Novelty Comparison
                novelty_comparison = extract_section_content(section)
            if "approach" in section_lower:
                approach = extract_section_content(section)

    # Create Thinker format
    thinker_idea = {
        "id": treeplot_idea.get("id"),
        "Name": treeplot_idea.get("title"),
        "Title": treeplot_idea.get("title"),
        "Problem": problem,
        "Importance": importance,
        "Difficulty": difficulty,  # Maps to Feasibility in frontend
        "NoveltyComparison": novelty_comparison,  # Maps to Novelty in frontend
        "Approach": approach,
    }

    return thinker_idea


def extract_section_content(section: str) -> str:
    """Helper function to extract content after section heading regardless of format"""
    # Check if section contains a colon (indicating a header)
    if ":" in section:
        # Split at the first colon to separate header from content
        parts = section.split(":", 1)
        if len(parts) > 1:
            # Return just the content part, removing any asterisks
            return parts[1].replace("**", "").strip()

    # If there's no colon or we couldn't extract properly,
    # just clean any formatting and return the whole section
    return section.replace("**", "").strip()


if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")
