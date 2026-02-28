"""Model adapter for smolagents LiteLLM integration."""

from __future__ import annotations

from smolagents import LiteLLMModel

# Model name mappings from tiny-scientist format to LiteLLM format
MODEL_MAPPING = {
    # OpenAI models
    "gpt-5": "gpt-5",
    "gpt-5-pro": "gpt-5-pro",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-pro": "gpt-5.2-pro",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "o3": "o3",
    "o4-mini-deep-research": "o4-mini-deep-research",
    # Anthropic models
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "claude-opus-4-5": "anthropic/claude-opus-4-5",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    "claude-opus-4": "anthropic/claude-opus-4",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    # DeepSeek models
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-reasoner": "deepseek/deepseek-reasoner",
    # Meta Llama models via Together AI
    "llama3.1-405b": "together_ai/meta-llama/Llama-3.1-405B-Instruct-Turbo",
    "llama3.1-70b": "together_ai/meta-llama/Llama-3.1-70B-Instruct-Turbo",
    "llama3.1-8b": "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo",
    # Google models
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini/gemini-2.5-flash-lite",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
}


def create_smolagents_model(model: str) -> LiteLLMModel:
    """
    Create a smolagents LiteLLMModel from a tiny-scientist model name.

    Args:
        model: Model name in tiny-scientist format (e.g., "gpt-5", "claude-sonnet-4-5")

    Returns:
        LiteLLMModel configured for the specified model
    """
    # Map to LiteLLM format, or use as-is if not in mapping
    model_id = MODEL_MAPPING.get(model, model)
    return LiteLLMModel(model_id=model_id, model_kwargs={"drop_params": True})
