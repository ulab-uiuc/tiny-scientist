from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = ROOT / relative_path
    package_parts = module_name.split(".")
    for index in range(1, len(package_parts)):
        package_name = ".".join(package_parts[:index])
        if package_name not in sys.modules:
            package = importlib.util.module_from_spec(
                importlib.util.spec_from_loader(package_name, loader=None)
            )
            package.__path__ = [str((ROOT / "/".join(package_parts[:index])).resolve())]
            sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_skill_loader_reads_agents_and_claude_dirs(tmp_path: Path) -> None:
    skill_loader = _load_module(
        "skill_loader_module", "tiny_scientist/utils/skill_loader.py"
    )
    agents_dir = tmp_path / ".agents" / "skills" / "thinker-agent"
    claude_dir = tmp_path / ".claude" / "skills" / "thinker-claude"
    agents_dir.mkdir(parents=True)
    claude_dir.mkdir(parents=True)
    (agents_dir / "SKILL.md").write_text("agent skill", encoding="utf-8")
    (claude_dir / "SKILL.md").write_text("claude skill", encoding="utf-8")

    skill_loader._SKILL_DIRS = (tmp_path / ".agents" / "skills", tmp_path / ".claude" / "skills")

    combined = skill_loader.load_skill("thinker")

    assert "thinker-agent" in combined
    assert "thinker-claude" in combined


def test_sdk_mcp_writes_config_and_tool_names(tmp_path: Path) -> None:
    sdk_mcp = _load_module("sdk_mcp_module", "tiny_scientist/utils/sdk_mcp.py")

    config_path = sdk_mcp.ensure_mcp_config(str(tmp_path), include_drawer=False)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tool_names = sdk_mcp.claude_allowed_mcp_tools(include_drawer=False)

    assert "mcpServers" in config
    assert "tiny_scientist_research" in config["mcpServers"]
    assert "mcp__tiny_scientist_research__paper_search" in tool_names
    assert "mcp__tiny_scientist_research__claim_verifier" in tool_names


def test_claude_agent_runner_passes_model_and_native_skill_sources() -> None:
    captured_options = []

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs):
            captured_options.append(kwargs)

    async def fake_query(prompt: str, options: object):
        del prompt, options
        yield types.SimpleNamespace(
            result="done",
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = FakeClaudeAgentOptions
    fake_sdk.query = fake_query
    sys.modules["claude_agent_sdk"] = fake_sdk

    budget_module = _load_module(
        "tiny_scientist.budget_checker",
        "tiny_scientist/budget_checker.py",
    )
    runner_module = _load_module(
        "tiny_scientist.utils.claude_agent_runner",
        "tiny_scientist/utils/claude_agent_runner.py",
    )
    runner = runner_module.ClaudeAgentRunner(
        instructions="system",
        allowed_tools=["Skill"],
        cwd=".",
        model="claude-3-5-sonnet-20241022",
        cost_tracker=budget_module.BudgetChecker(),
    )

    result = asyncio.run(runner._run_async("prompt", "task"))

    assert result == "done"
    assert captured_options[0]["model"] == "claude-3-5-sonnet-20241022"
    assert captured_options[0]["setting_sources"] == ["user", "project"]


def test_claude_agent_runner_uses_supported_mcp_option_name() -> None:
    captured_options = []

    class FakeClaudeAgentOptions:
        def __init__(
            self,
            *,
            model: str,
            cwd: str,
            allowed_tools: list[str],
            permission_mode: str,
            setting_sources: list[str],
            mcp_config_path: str,
        ):
            captured_options.append(
                {
                    "model": model,
                    "cwd": cwd,
                    "allowed_tools": allowed_tools,
                    "permission_mode": permission_mode,
                    "setting_sources": setting_sources,
                    "mcp_config_path": mcp_config_path,
                }
            )

    async def fake_query(prompt: str, options: object):
        del prompt, options
        yield types.SimpleNamespace(result="done", usage=None)

    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = FakeClaudeAgentOptions
    fake_sdk.query = fake_query
    sys.modules["claude_agent_sdk"] = fake_sdk

    runner_module = _load_module(
        "tiny_scientist.utils.claude_agent_runner",
        "tiny_scientist/utils/claude_agent_runner.py",
    )
    runner = runner_module.ClaudeAgentRunner(
        instructions="system",
        allowed_tools=["Skill"],
        cwd=".",
        model="claude-sonnet-4-6",
        mcp_config_path="/tmp/test-mcp.json",
    )

    result = asyncio.run(runner._run_async("prompt", "task"))

    assert result == "done"
    assert captured_options[0]["mcp_config_path"] == "/tmp/test-mcp.json"
