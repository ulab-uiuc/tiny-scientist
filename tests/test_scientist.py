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
    main_py = output_dir / "main.py"
    main_py.write_text("print('Test experiment')")

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


def test_mock() -> None:
    assert True


def test_coder_recovers_from_max_turns_with_existing_workspace(
    test_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = pytest.importorskip("agents")
    from tiny_scientist.coder import Coder

    MaxTurnsExceeded = agents.exceptions.MaxTurnsExceeded
    coder = Coder(
        model="gpt-4-test",
        output_dir=str(test_output_dir),
        use_docker=False,
    )

    idea = {
        "Title": "Test Idea",
        "Problem": "Test problem",
        "Approach": "Test approach",
        "Experiment": {"Model": {}, "Dataset": {}, "Metric": {}},
        "ExperimentTable": "| Row | Details |\n| --- | --- |\n| Baselines | Test |\n",
    }

    monkeypatch.setattr(coder, "setup_agent", lambda: None)
    monkeypatch.setattr(
        coder,
        "_run_experiment_loop",
        lambda _idea, _baseline=None: (_ for _ in ()).throw(MaxTurnsExceeded("max turns")),
    )
    monkeypatch.setattr(coder, "_format_experiment_for_prompt", lambda _exp: ("", "", "", "", "", ""))
    monkeypatch.setattr(
        coder,
        "_run_single_experiment",
        lambda run_num, idea, experiment_table, table_rows, timeout=7200: (0, "ok"),
    )
    monkeypatch.setattr(coder, "_update_notes", lambda: None)
    monkeypatch.setattr(coder, "_write_search_links_manifest", lambda _idea: None)

    status, exp_dir, note = coder.run(idea=idea)

    assert status is True
    assert exp_dir == str(test_output_dir)
    assert note is not None
    assert "max-turn limit" in note


def test_scientist_code_marks_partial_success(
    test_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("agents")
    from tiny_scientist.scientist import TinyScientist

    scientist = TinyScientist(
        model="gpt-4-test",
        output_dir=str(test_output_dir),
        enable_safety_check=False,
        use_docker=False,
        agent_sdk="openai",
    )
    monkeypatch.setattr(
        scientist.coder,
        "run",
        lambda idea, baseline_results=None: (
            True,
            str(test_output_dir),
            "Recovered by running the current workspace.",
        ),
    )

    status, exp_dir = scientist.code(
        idea={
            "Title": "Test Idea",
            "Experiment": {"Model": {}, "Dataset": {}, "Metric": {}},
        }
    )

    assert status is True
    assert exp_dir == str(test_output_dir)
