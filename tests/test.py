import json
import os
from unittest.mock import Mock, patch

import pytest
from ai_scientist.scientist import Scientist


@pytest.fixture
def mock_client() -> Mock:
    return Mock()

@pytest.fixture
def mock_model():
    return "gpt-4-test"

@pytest.fixture
def test_base_dir(tmp_path) -> str:
    # Create temporary test directory
    base_dir = tmp_path / "test_project"
    base_dir.mkdir()

    # Create required files
    experiment_py = base_dir / "experiment.py"
    experiment_py.write_text("print('Test experiment')")

    prompt_json = base_dir / "prompt.json"
    prompt_json.write_text(json.dumps({
        "task_description": "Test task",
        "system": "Test system prompt"
    }))

    seed_ideas_json = base_dir / "seed_ideas.json"
    seed_ideas_json.write_text(json.dumps([
        {
            "Name": "test_idea",
            "Title": "Test Idea",
            "Experiment": "Test experiment description"
        }
    ]))

    return str(base_dir)

@pytest.fixture
def scientist(mock_client, mock_model, test_base_dir) -> Scientist:
    return Scientist(
        model=mock_model,
        client=mock_client,
        base_dir=test_base_dir
    )

def test_think_generates_ideas(scientist, mock_client) -> None:
    # Mock LLM response for idea generation
    mock_client.chat.completions.create.return_value.choices[0].message.content = """
    THOUGHT: Test thought

    NEW IDEA JSON:
    ```json
    {
        "Name": "test_idea",
        "Title": "Test Title",
        "Experiment": "Test experiment",
        "Interestingness": 8,
        "Feasibility": 7,
        "Novelty": 6
    }
    ```
    """

    # Test idea generation
    ideas = scientist.think(
        task_description="Test task",
        code="Test code",
        max_num_generations=1
    )

    assert len(ideas) == 1
    assert ideas[0]["Name"] == "test_idea"
    assert ideas[0]["Title"] == "Test Title"
    mock_client.chat.completions.create.assert_called()

def test_think_next_builds_on_previous(scientist, mock_client) -> None:
    # Mock LLM response
    mock_client.chat.completions.create.return_value.choices[0].message.content = """
    THOUGHT: Test thought

    NEW IDEA JSON:
    ```json
    {
        "Name": "next_idea",
        "Title": "Next Test Title",
        "Experiment": "Next test experiment",
        "Interestingness": 8,
        "Feasibility": 7,
        "Novelty": 6
    }
    ```
    """

    prev_ideas = [{
        "Name": "prev_idea",
        "Title": "Previous Test",
        "Experiment": "Previous experiment"
    }]

    ideas = scientist.think_next(
        prev_ideas=prev_ideas,
        num_reflections=1
    )

    assert len(ideas) == 2  # Previous + new idea
    assert ideas[-1]["Name"] == "next_idea"
    mock_client.chat.completions.create.assert_called()

@patch('subprocess.run')
def test_code_executes_experiments(mock_subprocess, scientist, mock_client) -> None:
    # Mock Aider coder responses
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Test implementation"

    # Mock subprocess success
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stderr = ""

    # Create mock results file
    os.makedirs(os.path.join(scientist.base_dir, "run_1"), exist_ok=True)
    with open(os.path.join(scientist.base_dir, "run_1", "final_info.json"), "w") as f:
        json.dump({"test_metric": {"means": 0.5}}, f)

    success = scientist.code(
        idea={
            "Name": "test_idea",
            "Title": "Test Title",
            "Experiment": "Test experiment"
        },
        baseline_results={"baseline": 0.4}
    )

    assert success
    mock_subprocess.assert_called()

@patch('subprocess.run')
def test_write_generates_paper(mock_subprocess, scientist, mock_client) -> None:
    # Mock successful LaTeX compilation
    mock_subprocess.return_value.returncode = 0

    # Mock Aider coder responses
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Test paper content"

    # Create LaTeX directory
    latex_dir = os.path.join(scientist.base_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    scientist.write(
        idea={
            "Name": "test_idea",
            "Title": "Test Title",
            "Experiment": "Test experiment"
        },
        folder_name=scientist.base_dir
    )

    mock_subprocess.assert_called()
    assert os.path.exists(os.path.join(latex_dir, "template.tex"))

def test_review_evaluates_paper(scientist, mock_client) -> None:
    # Mock LLM review response
    mock_client.chat.completions.create.return_value.choices[0].message.content = """
    THOUGHT: Test review thought

    REVIEW JSON:
    ```json
    {
        "Summary": "Test summary",
        "Strengths": ["Strength 1"],
        "Weaknesses": ["Weakness 1"],
        "Originality": 3,
        "Quality": 3,
        "Clarity": 3,
        "Significance": 3,
        "Questions": ["Question 1"],
        "Limitations": ["Limitation 1"],
        "Ethical Concerns": false,
        "Soundness": 3,
        "Presentation": 3,
        "Contribution": 3,
        "Overall": 7,
        "Confidence": 4,
        "Decision": "Accept"
    }
    ```
    """

    review = scientist.review(
        text="Test paper content",
        num_reflections=1
    )

    assert review["Summary"] == "Test summary"
    assert review["Decision"] == "Accept"
    mock_client.chat.completions.create.assert_called()

def test_end_to_end_workflow(scientist, mock_client, mock_subprocess) -> None:
    # This test demonstrates the complete workflow
    with patch('subprocess.run') as mock_subprocess:
        # Mock successful subprocess calls
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # 1. Generate ideas
        mock_client.chat.completions.create.return_value.choices[0].message.content = """
        THOUGHT: Test thought

        NEW IDEA JSON:
        ```json
        {
            "Name": "test_workflow",
            "Title": "Test Workflow",
            "Experiment": "Test workflow experiment",
            "Interestingness": 8,
            "Feasibility": 7,
            "Novelty": 6
        }
        ```
        """

        ideas = scientist.think(
            task_description="Test workflow",
            code="Test code",
            max_num_generations=1
        )

        assert len(ideas) == 1
        idea = ideas[0]

        # 2. Implement experiments
        # Create mock results
        os.makedirs(os.path.join(scientist.base_dir, "run_1"), exist_ok=True)
        with open(os.path.join(scientist.base_dir, "run_1", "final_info.json"), "w") as f:
            json.dump({"test_metric": {"means": 0.5}}, f)

        success = scientist.code(
            idea=idea,
            baseline_results={"baseline": 0.4}
        )

        assert success

        # 3. Write paper
        latex_dir = os.path.join(scientist.base_dir, "latex")
        os.makedirs(latex_dir, exist_ok=True)

        scientist.write(
            idea=idea,
            folder_name=scientist.base_dir
        )

        # 4. Review paper
        mock_client.chat.completions.create.return_value.choices[0].message.content = """
        THOUGHT: Test review

        REVIEW JSON:
        ```json
        {
            "Summary": "Test workflow summary",
            "Strengths": ["Good workflow"],
            "Weaknesses": ["Could be improved"],
            "Originality": 3,
            "Quality": 3,
            "Clarity": 3,
            "Significance": 3,
            "Questions": ["Any improvements planned?"],
            "Limitations": ["Limited scope"],
            "Ethical Concerns": false,
            "Soundness": 3,
            "Presentation": 3,
            "Contribution": 3,
            "Overall": 7,
            "Confidence": 4,
            "Decision": "Accept"
        }
        ```
        """

        review = scientist.review("Test paper content")

        assert review["Decision"] == "Accept"
        assert review["Summary"] == "Test workflow summary"
