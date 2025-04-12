import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def mock_model() -> str:
    return "gpt-4-test"


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    # Create a subdirectory under tmp_path
    output_dir = tmp_path / "test_scientist"
    output_dir.mkdir()

    # Create required files using Path methods
    experiment_py = output_dir / "experiment.py"
    experiment_py.write_text("print('Test experiment')")

    prompt_json = output_dir / "prompt.json"
    prompt_json.write_text(
        json.dumps({"task_description": "Test task", "system": "Test system prompt"})
    )

    seed_ideas_json = output_dir / "seed_ideas.json"
    seed_ideas_json.write_text(
        json.dumps(
            [
                {
                    "Name": "test_idea",
                    "Title": "Test Idea",
                    "Experiment": "Test experiment description",
                }
            ]
        )
    )

    # Return a Path object, not a string
    return output_dir


def test_mock() -> bool:
    return True
