# Pricing data for each model (prices are in dollars per million (1,000,000) tokens)
import math
from typing import Dict, Iterable, Optional, Tuple, Union

MODEL_PRICING = {
    # OpenAI models - GPT-5 family
    "gpt-5": (1.25, 10.0),
    "gpt-5-pro": (21.0, 168.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.03, 0.12),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.2-pro": (21.0, 168.0),
    "gpt-5.2-codex": (1.75, 14.0),
    # OpenAI models - GPT-4.1 family
    "gpt-4.1": (1.25, 10.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.05, 0.20),
    # OpenAI models - O-series
    "o3": (10.0, 40.0),
    "o4-mini-deep-research": (2.0, 8.0),
    # OpenRouter models
    "llama3.1-405b": (3.5, 3.5),
    # Anthropic models - Claude 4.6 and 4.5 series
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    # DeepSeek models
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.28, 0.42),
    # Google Gemini models - Gemini 2.5 series
    "gemini-2.5-pro": (1.25, 5.0),
    "gemini-2.5-flash": (0.25, 1.0),
    "gemini-2.5-flash-lite": (0.1, 0.4),
    # Google Gemini models - Gemini 2.0 series
    "gemini-2.0-flash": (0.2, 0.8),
}


def calculate_pricing(model: str, input_tokens: int, output_tokens: int) -> float:
    # Check if the model exists
    if model not in MODEL_PRICING:
        found_match = False
        for m in MODEL_PRICING:
            if model.startswith(m):
                model = m
                found_match = True
                break

        if not found_match:
            raise ValueError(f"Pricing for '{model}' is not found.")

    input_price, output_price = MODEL_PRICING[model]

    # Check if pricing data is available
    if input_price is None or output_price is None:
        raise ValueError(f"Pricing for '{model}' is unavailable.")

    # The pricing is per million (1,000,000) tokens.
    input_cost = (input_tokens / 1000000) * input_price
    output_cost = (output_tokens / 1000000) * output_price

    total_cost = input_cost + output_cost
    return total_cost


def estimate_tokens_from_text(text: Optional[str]) -> int:
    """Roughly estimate token count from text length."""
    if not text:
        return 0
    # Empirical rule: ~4 characters per token
    return max(1, math.ceil(len(text) / 4))


def estimate_prompt_cost(
    model: str,
    input_texts: Iterable[Optional[str]],
    expected_output_tokens: Optional[int] = None,
) -> Optional[float]:
    """Estimate the price of a single LLM call given textual inputs."""

    texts = [text for text in input_texts if text]
    if not texts and expected_output_tokens is None:
        return None

    input_tokens = sum(estimate_tokens_from_text(text) for text in texts)

    if expected_output_tokens is None:
        longest_text = max(texts, key=len, default="")
        expected_output_tokens = max(estimate_tokens_from_text(longest_text), 256)

    try:
        return calculate_pricing(model, input_tokens, expected_output_tokens)
    except ValueError:
        return None


# ---- Budget allocation helpers ---------------------------------------------

BUDGET_MODULE_KEYS: Tuple[str, ...] = (
    "safety_checker",
    "thinker",
    "coder",
    "writer",
    "reviewer",
)

DEFAULT_BUDGET_PREFERENCE = "balanced"

BUDGET_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "safety_checker": 0.1,
        "thinker": 0.25,
        "writer": 0.25,
        "reviewer": 0.25,
        "coder": 0.15,
    },
    "write-heavy": {
        "safety_checker": 0.05,
        "thinker": 0.15,
        "writer": 0.5,
        "reviewer": 0.2,
        "coder": 0.1,
    },
    "think-heavy": {
        "safety_checker": 0.05,
        "thinker": 0.5,
        "writer": 0.15,
        "reviewer": 0.2,
        "coder": 0.1,
    },
    "review-heavy": {
        "safety_checker": 0.05,
        "thinker": 0.15,
        "writer": 0.15,
        "reviewer": 0.5,
        "coder": 0.15,
    },
}


def coerce_budget(raw_budget: Optional[Union[float, int, str]]) -> Optional[float]:
    if raw_budget is None:
        return None
    try:
        budget_value = float(raw_budget)
    except (TypeError, ValueError) as exc:
        raise ValueError("Budget must be a number if provided.") from exc
    if budget_value < 0:
        raise ValueError("Budget must be non-negative.")
    return budget_value


def normalize_budget_preference(preference: Optional[str]) -> str:
    if not preference:
        preference = DEFAULT_BUDGET_PREFERENCE
    normalized = preference.lower()
    if normalized not in BUDGET_WEIGHTS:
        raise ValueError(f"Unknown budget preference: {preference}")
    return normalized


def resolve_budget_settings(
    budget: Optional[Union[float, int, str]],
    budget_preference: Optional[str] = None,
) -> Tuple[Optional[float], str, Dict[str, Optional[float]]]:
    normalized_budget = coerce_budget(budget)
    normalized_preference = normalize_budget_preference(budget_preference)

    if normalized_budget is None:
        allocation = {key: None for key in BUDGET_MODULE_KEYS}
    else:
        weights = BUDGET_WEIGHTS[normalized_preference]
        allocation = {
            key: normalized_budget * weights[key] for key in BUDGET_MODULE_KEYS
        }

    return normalized_budget, normalized_preference, allocation


def compute_budget_allocation(
    budget: Optional[Union[float, int, str]],
    budget_preference: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    _, _, allocation = resolve_budget_settings(budget, budget_preference)
    return allocation
