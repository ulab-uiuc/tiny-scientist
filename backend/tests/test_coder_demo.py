import json
import os
from pathlib import Path

import pytest

from backend.app import app


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _load_demo_idea() -> dict:
    idea_path = Path(__file__).resolve().parents[2] / "demo_test" / "idea.json"
    if not idea_path.exists():
        raise FileNotFoundError(f"Idea file not found: {idea_path}")
    return json.loads(idea_path.read_text())


def _configure_backend(client) -> None:
    model = "gpt-4o"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping coder integration test")

    response = client.post(
        "/api/configure",
        json={
            "model": model,
            "api_key": api_key,
            "budget": 10.0,
            "budget_preference": "balanced",
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)


def test_coder_with_demo_idea(client):
    """
    Integration test: run backend /api/code with the demo idea.
    Ensures coder executes and produces experiment outputs.
    """
    _configure_backend(client)

    idea_payload = _load_demo_idea()
    response = client.post(
        "/api/code",
        json={"idea": {"originalData": idea_payload}},
    )

    assert response.status_code == 200, response.get_data(as_text=True)

    data = response.get_json()
    assert data is not None, "No JSON body returned"
    assert data.get("success") is True, data

    experiment_dir = data.get("experiment_dir")
    assert experiment_dir, data

    generated_base = Path(__file__).resolve().parents[2] / "generated"
    abs_experiment_dir = generated_base / experiment_dir
    assert abs_experiment_dir.exists(), f"Experiment dir missing: {abs_experiment_dir}"

    expected_files = {
        "experiment.py",
        "notes.txt",
        "experiment_results.txt",
    }
    missing = [
        name for name in expected_files if not (abs_experiment_dir / name).exists()
    ]
    assert not missing, f"Missing files in {abs_experiment_dir}: {missing}"
