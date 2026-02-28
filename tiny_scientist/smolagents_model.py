"""Optional compatibility adapter for smolagents LiteLLM integration."""

from __future__ import annotations

# Model name mappings from tiny-scientist format to LiteLLM format
MODEL_MAPPING = {
    # OpenAI models
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    # Anthropic models
    "claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229": "anthropic/claude-3-opus-20240229",
    "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku-20240307": "anthropic/claude-3-haiku-20240307",
    # DeepSeek models
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-coder-v2-0724": "deepseek/deepseek-coder",
    # Meta Llama models via Together AI
    "llama3.1-405b": "together_ai/meta-llama/Llama-3.1-405B-Instruct-Turbo",
    "llama3.1-70b": "together_ai/meta-llama/Llama-3.1-70B-Instruct-Turbo",
    "llama3.1-8b": "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo",
    # Google models
    "gemini-pro": "gemini/gemini-pro",
    "gemini-1.5-pro": "gemini/gemini-1.5-pro",
}


def create_smolagents_model(model: str):
    """
    Create a smolagents LiteLLMModel from a tiny-scientist model name.

    Args:
        model: Model name in tiny-scientist format (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")

    Returns:
        LiteLLMModel configured for the specified model
    """
    try:
        from smolagents import LiteLLMModel
    except ImportError as exc:
        raise ImportError(
            "smolagents is not installed. Install the optional compatibility "
            "dependency if you need create_smolagents_model()."
        ) from exc

    model_id = MODEL_MAPPING.get(model, model)
    return LiteLLMModel(model_id=model_id)
