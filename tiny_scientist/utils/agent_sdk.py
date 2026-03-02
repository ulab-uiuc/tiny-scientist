"""Helpers for selecting the agent SDK backend."""

from __future__ import annotations

from typing import Optional

try:
    from typing import Literal
except ImportError:  # pragma: no cover - Python < 3.8
    from typing_extensions import Literal

OPENAI_AGENT_SDK: Literal["openai"] = "openai"
CLAUDE_AGENT_SDK: Literal["claude"] = "claude"
DEFAULT_AGENT_SDK: Literal["claude"] = CLAUDE_AGENT_SDK
SUPPORTED_AGENT_SDKS = (CLAUDE_AGENT_SDK, OPENAI_AGENT_SDK)

AgentSdk = Literal["openai", "claude"]


def resolve_agent_sdk(
    agent_sdk: Optional[str] = None,
    use_claude_agent_sdk: Optional[bool] = None,
) -> AgentSdk:
    """Resolve the configured agent SDK backend.

    `agent_sdk` is the primary interface. `use_claude_agent_sdk` is kept for
    backward compatibility and maps to the equivalent backend when provided.
    """
    normalized_sdk = (agent_sdk or "").strip().lower()
    if normalized_sdk:
        if normalized_sdk not in SUPPORTED_AGENT_SDKS:
            raise ValueError(
                "agent_sdk must be either 'openai' or 'claude', "
                f"got {agent_sdk!r}."
            )
        if use_claude_agent_sdk is not None:
            expected = CLAUDE_AGENT_SDK if use_claude_agent_sdk else OPENAI_AGENT_SDK
            if normalized_sdk != expected:
                raise ValueError(
                    "agent_sdk conflicts with use_claude_agent_sdk: "
                    f"got agent_sdk={agent_sdk!r}, "
                    f"use_claude_agent_sdk={use_claude_agent_sdk!r}."
                )
        return normalized_sdk

    if use_claude_agent_sdk is not None:
        return CLAUDE_AGENT_SDK if use_claude_agent_sdk else OPENAI_AGENT_SDK

    return DEFAULT_AGENT_SDK


def is_claude_agent_sdk(agent_sdk: AgentSdk) -> bool:
    return agent_sdk == CLAUDE_AGENT_SDK


def validate_agent_sdk_model_combo(agent_sdk: AgentSdk, model: str) -> None:
    """Raise a clear error for unsupported runtime/model combinations."""
    normalized_model = (model or "").strip().lower()
    if not normalized_model:
        return

    if agent_sdk == CLAUDE_AGENT_SDK and normalized_model.startswith(
        ("gpt-", "o1", "o3", "o4", "codex")
    ):
        raise ValueError(
            "agent_sdk='claude' uses Claude Code / claude_agent_sdk and cannot run "
            f"OpenAI models like {model!r}. Use agent_sdk='openai' for GPT/o/codex "
            "models, or switch the model to a Claude family model."
        )
