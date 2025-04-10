# Pricing data for each model (prices are in dollars per million (1,000,000) tokens)
MODEL_PRICING = {
    # OpenAI models
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10),
    "o1-preview": (15, 60),
    "o1-mini": (1.1, 4.4),
    "o1": (15, 60),
    # OpenRouter models
    "llama3.1-405b": (3.5, 3.5),
    # Anthropic models
    "claude-3-sonnet-v1": (0.8, 4),
    "claude-3-sonnet": (3, 15),
    "claude-3-5-sonnet-v2": (3, 15),
    "claude-3-5-sonnet": (3, 15),
    "claude-3-haiku-v1": (0.25, 1.25),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus-v1": (15, 75),
    "claude-3-opus": (0.8, 4),
    # DeepSeek models
    "deepseek-chat": (0.07, 0.27),
    "deepseek-reasoner": (0.14, 0.55),
    # Google Gemini models
    "gemini-1.5-flash": (0.01875, 0.075),
    "gemini-1.5-pro": (0.3125, 1.25),
}


def calculate_pricing(model: str, input_tokens: int, output_tokens: int) -> float:
    # Check if the model exists
    if model not in MODEL_PRICING:
        for m in MODEL_PRICING:
            if model.startswith(m):
                model = m
            else:
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
