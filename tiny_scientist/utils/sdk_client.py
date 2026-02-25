"""Shared utilities for OpenAI Agents SDK-backed stages."""

from __future__ import annotations

import os
from typing import Any, Optional

from tiny_scientist.budget_checker import BudgetChecker


def configure_openai_agents_for_model(model: str) -> None:
    """Set openai-agents SDK global client based on model prefix.

    - GPT/o-series: SDK default (reads OPENAI_API_KEY)
    - claude-*: Anthropic's OpenAI-compatible endpoint
    - deepseek-*: DeepSeek's OpenAI-compatible endpoint
    - others: LiteLLM proxy (LITELLM_PROXY_URL env var) if set
    """
    from agents import set_default_openai_client
    from openai import AsyncOpenAI

    if model.startswith(("gpt-", "o1", "o3")):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            set_default_openai_client(AsyncOpenAI(api_key=api_key))

    elif model.startswith("claude-"):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        set_default_openai_client(
            AsyncOpenAI(
                base_url="https://api.anthropic.com/v1/",
                api_key=api_key,
            )
        )

    elif model.startswith("deepseek-"):
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        set_default_openai_client(
            AsyncOpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=api_key,
            )
        )

    else:
        litellm_url = os.environ.get("LITELLM_PROXY_URL", "")
        if litellm_url:
            set_default_openai_client(
                AsyncOpenAI(
                    base_url=litellm_url,
                    api_key=os.environ.get("LITELLM_API_KEY", "none"),
                )
            )


def track_sdk_cost(
    result: Any,
    cost_tracker: BudgetChecker,
    model: str,
    task_name: str,
) -> None:
    """Iterate result.raw_responses and call cost_tracker.add_cost() for each."""
    try:
        for resp in result.raw_responses:
            usage = getattr(resp, "usage", None)
            if usage is None:
                continue
            input_tokens = (
                getattr(usage, "input_tokens", None)
                or getattr(usage, "prompt_tokens", 0)
                or 0
            )
            output_tokens = (
                getattr(usage, "output_tokens", None)
                or getattr(usage, "completion_tokens", 0)
                or 0
            )
            cost_tracker.add_cost(model, input_tokens, output_tokens, task_name)
    except Exception:
        pass
