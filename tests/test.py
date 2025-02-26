import json
import os
from unittest.mock import Mock, patch

import pytest
from tiny_scientist.reviewer import Reviewer


@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def mock_model():
    return "gpt-4-test"

@pytest.fixture
def test_base_dir(tmp_path):
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
def scientist(mock_client, mock_model, test_base_dir):
    return Scientist(
        model=mock_model,
        client=mock_client,
        base_dir=test_base_dir
    )

def test_think_generates_ideas(scientist, mock_client):
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

def test_think_next_builds_on_previous(scientist, mock_client):
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
def test_code_executes_experiments(mock_subprocess, scientist, mock_client):
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
def test_write_generates_paper(mock_subprocess, scientist, mock_client):
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

def test_review_evaluates_paper(scientist, mock_client):
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

def test_reviewer_independent_functionality(mock_client, mock_model):
    from tiny_scientist.reviewer import Reviewer

    # Initialize the Reviewer with mock model and client
    reviewer = Reviewer(model=mock_model, client=mock_client)

    # Mock LLM review response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = """
    THOUGHT: Independent review thought

    REVIEW JSON:
    ```json
    {
        "Summary": "Independent test summary",
        "Strengths": ["Independent strength 1"],
        "Weaknesses": ["Independent weakness 1"],
        "Originality": 4,
        "Quality": 4,
        "Clarity": 4,
        "Significance": 4,
        "Questions": ["Independent question 1"],
        "Limitations": ["Independent limitation 1"],
        "Ethical Concerns": false,
        "Soundness": 4,
        "Presentation": 4,
        "Contribution": 4,
        "Overall": 8,
        "Confidence": 5,
        "Decision": "Strong Accept"
    }
    ```
    """
    mock_client.chat.completions.create.return_value = mock_response

    # Mock the LLM response function
    with patch('tiny_scientist.llm.get_response_from_llm', return_value=(mock_response.choices[0].message.content, [])):
        # Mock the file loading methods
        with patch.object(Reviewer, 'load_paper', return_value='Mock paper content'):
            with patch.object(Reviewer, 'load_review', return_value='Mock review content'):
                # Perform the review
                review = reviewer.perform_review(text="Independent test paper content", num_reflections=1)

    # Assertions to ensure the review is as expected
    assert review["Summary"] == "Independent test summary"
    assert review["Decision"] == "Strong Accept"
    mock_client.chat.completions.create.assert_called()
def test_end_to_end_workflow(scientist, mock_client, mock_subprocess):
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
