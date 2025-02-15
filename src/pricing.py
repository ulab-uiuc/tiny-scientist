# Pricing data for each model (prices are in dollars per million (1,000,000) tokens)
MODEL_PRICING = {
    # Anthropic models
    "claude-3-5-sonnet-20240620": (3, 15),
    "claude-3-5-sonnet-20241022": (3, 15),
    # OpenAI models
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4o-2024-05-13": (2.5, 10),
    "gpt-4o-2024-08-06": (2.5, 10),
    "o1-preview-2024-09-12": (15, 60),
    "o1-mini-2024-09-12": (1.1, 4.4),
    "o1-2024-12-17": (15, 60),
    # OpenRouter models
    "llama3.1-405b": (3.5, 3.5),
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0": (0.8, 4),
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0": (3, 15),
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0": (3, 15),
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0": (0.25, 1.25),
    "bedrock/anthropic.claude-3-opus-20240229-v1:0": (15, 75),
    "vertex_ai/claude-3-opus@20240229": (0.8, 4),
    "vertex_ai/claude-3-5-sonnet@20240620": (3, 15),
    "vertex_ai/claude-3-5-sonnet-v2@20241022": (3, 15),
    "vertex_ai/claude-3-sonnet@20240229": (3, 15),
    "vertex_ai/claude-3-haiku@20240307": (0.25, 1.25),
    # DeepSeek models
    "deepseek-chat": (0.07, 0.27),
    "deepseek-coder": (None, None),
    "deepseek-reasoner": (0.14, 0.55),
    # Google Gemini models
    "gemini-1.5-flash": (0.01875, 0.075),
    "gemini-1.5-pro": (0.3125, 1.25),
}

def calculate_pricing(model: str, input_tokens: int, output_tokens: int) -> float:
    # Check if the model exists
    if model not in MODEL_PRICING:
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