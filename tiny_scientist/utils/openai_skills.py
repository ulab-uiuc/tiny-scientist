"""Helpers for OpenAI Agents SDK shell-based skill mounting."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional


def load_openai_skill_specs(stage: str) -> List[Dict[str, Any]]:
    """Load official OpenAI skill specs from environment JSON."""
    env_names = [
        f"OPENAI_AGENT_SKILLS_{stage.upper()}_JSON",
        "OPENAI_AGENT_SKILLS_JSON",
    ]
    for env_name in env_names:
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        raise ValueError(f"{env_name} must decode to a JSON object or list of objects.")
    return []


def build_openai_skill_shell_tool(
    stage: str,
    working_directory: str,
    timeout: int = 300,
) -> Optional[Any]:
    """Build a ShellTool with mounted skills when official skill specs are configured."""
    skill_specs = load_openai_skill_specs(stage)
    if not skill_specs:
        return None

    try:
        from agents import ShellTool
    except Exception:
        return None

    async def run_shell(request: Any) -> str:
        command = _extract_shell_command(request)
        if not command:
            return "[Error] Missing shell command."
        cwd = _extract_shell_cwd(request) or working_directory
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[Error] Command timed out after {timeout}s"
        except Exception as exc:
            return f"[Error] {exc}"

        output = result.stdout or ""
        if result.stderr:
            output += ("\n[stderr]\n" if output else "[stderr]\n") + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"

    environment: Dict[str, Any] = {"type": "local", "skills": skill_specs}
    return ShellTool(executor=run_shell, environment=environment)


def _extract_shell_command(request: Any) -> str:
    for attr in ("command", "input", "text"):
        value = getattr(request, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(request, dict):
        for key in ("command", "input", "text"):
            value = request.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _extract_shell_cwd(request: Any) -> str:
    value = getattr(request, "cwd", None)
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(request, dict):
        value = request.get("cwd")
        if isinstance(value, str) and value.strip():
            return value
    return ""
