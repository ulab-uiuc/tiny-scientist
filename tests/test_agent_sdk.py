from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tiny_scientist"
    / "utils"
    / "agent_sdk.py"
)
SPEC = importlib.util.spec_from_file_location("agent_sdk_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load agent SDK helper from {MODULE_PATH}")
AGENT_SDK_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGENT_SDK_MODULE)
resolve_agent_sdk = AGENT_SDK_MODULE.resolve_agent_sdk


def test_resolve_agent_sdk_defaults_to_claude() -> None:
    assert resolve_agent_sdk() == "claude"


def test_resolve_agent_sdk_accepts_claude_legacy_flag() -> None:
    assert resolve_agent_sdk(use_claude_agent_sdk=True) == "claude"


def test_resolve_agent_sdk_accepts_openai_legacy_flag() -> None:
    assert resolve_agent_sdk(use_claude_agent_sdk=False) == "openai"


def test_resolve_agent_sdk_accepts_explicit_value() -> None:
    assert resolve_agent_sdk(agent_sdk="claude") == "claude"


def test_resolve_agent_sdk_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="must be either"):
        resolve_agent_sdk(agent_sdk="invalid")


def test_resolve_agent_sdk_rejects_conflicting_inputs() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_agent_sdk(agent_sdk="openai", use_claude_agent_sdk=True)
