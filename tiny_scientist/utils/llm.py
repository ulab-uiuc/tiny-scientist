import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import anthropic
import backoff
import openai
import toml
from google.generativeai.types import GenerationConfig

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")
if os.path.exists(config_path):
    config = toml.load(config_path)
else:
    config = {"core": {}}

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    # Anthropic models
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    # OpenRouter models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    # Google Gemini models
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Any = None,
    temperature: float = 0.75,
    n_responses: int = 1,
) -> Tuple[List[str], List[List[Dict[str, str]]]]:
    if msg_history is None:
        msg_history = []

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)

        for i, history in enumerate(new_msg_history):
            print(f"Response {i}:")
            for j, msg in enumerate(history):
                print(msg)

        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Any = None,
    temperature: float = 0.75,
) -> Tuple[str, List[Dict[str, Any]]]:
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model in [
        "gpt-4o-mini",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4o",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            max_completion_tokens=MAX_NUM_TOKENS,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-chat", "deepseek-coder"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["deepseek-reasoner"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "gemini" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        gemini_contents = [{"role": "system", "parts": system_message}]
        for m in new_msg_history:
            gemini_contents.append({"role": m["role"], "parts": m["content"]})
        response = client.generate_content(
            contents=gemini_contents,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=MAX_NUM_TOKENS,
                candidate_count=1,
            ),
        )
        content = response.text
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)

        for i, history in enumerate(new_msg_history):
            print(f"Response {i}:")
            for j, msg in enumerate(history):
                print(msg)

        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> Optional[Dict[str, Any]]:
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = cast(Dict[str, Any], json.loads(json_string))
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                parsed_json = cast(Dict[str, Any], json.loads(json_string))
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(
    model: str,
) -> Tuple[
    Union[
        anthropic.Anthropic,
        anthropic.AnthropicBedrock,
        anthropic.AnthropicVertex,
        openai.OpenAI,
    ],
    str,
]:
    llm_api_key = config["core"].get("llm_api_key", "")
    client: Union[
        anthropic.Anthropic,
        anthropic.AnthropicBedrock,
        anthropic.AnthropicVertex,
        openai.OpenAI,
    ]

    if model.startswith("claude-"):
        api_key = os.environ.get("ANTHROPIC_API_KEY", llm_api_key)
        if not api_key:
            raise ValueError(
                f"Missing Anthropic API key to use {model}. Set ANTHROPIC_API_KEY or llm_api_key in config.toml."
            )
        client = anthropic.Anthropic(api_key=api_key)
        return client, model

    elif model.startswith("bedrock") and "claude" in model:
        client = anthropic.AnthropicBedrock()
        return client, model.split("/")[-1]

    elif model.startswith("vertex_ai") and "claude" in model:
        client = anthropic.AnthropicVertex()
        return client, model.split("/")[-1]

    elif "gpt" in model or model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
        api_key = os.environ.get("OPENAI_API_KEY", llm_api_key)
        if not api_key:
            raise ValueError(
                f"Missing OpenAI API key to use {model}. Set OPENAI_API_KEY or llm_api_key in config.toml."
            )
        client = openai.OpenAI(api_key=api_key)
        return client, model

    elif model in ["deepseek-chat", "deepseek-reasoner"]:
        api_key = os.environ.get("DEEPSEEK_API_KEY", llm_api_key)
        if not api_key:
            raise ValueError(
                f"Missing DeepSeek API key to use {model}. Set DEEPSEEK_API_KEY or llm_api_key in config.toml."
            )
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return client, model

    elif model == "llama3.1-405b":
        api_key = os.environ.get("OPENROUTER_API_KEY", llm_api_key)
        if not api_key:
            raise ValueError(
                f"Missing OpenRouter API key to use {model}. Set OPENROUTER_API_KEY or llm_api_key in config.toml."
            )
        client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        return client, "meta-llama/llama-3.1-405b-instruct"

    else:
        raise ValueError(f"Model {model} not supported.")
