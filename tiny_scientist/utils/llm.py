import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import anthropic
import backoff
import openai
import toml
from google.generativeai.types import GenerationConfig

from tiny_scientist.budget_checker import BudgetChecker

# Load config
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.toml"
)
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
    # Together AI models - Meta Llama models
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    # Together AI models - Qwen models
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen3-235B-A22B-fp8-tput",
    # Together AI models - DeepSeek models
    "deepseek-ai/DeepSeek-V3",
    # Together AI models - Mistral models
    "mistralai/Mistral-Small-24B-Instruct-2501",
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
    cost_tracker: Optional[BudgetChecker] = None,
    task_name: Optional[str] = None,
) -> Tuple[List[str], List[List[Dict[str, str]]]]:
    if msg_history is None:
        msg_history = []

    input_tokens = 0
    output_tokens = 0
    response = None
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
            if cost_tracker is not None:
                cost_tracker.add_cost(model, input_tokens, output_tokens, task_name)
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
            if cost_tracker is not None:
                cost_tracker.add_cost(model, input_tokens, output_tokens, task_name)
    elif any(
        model.startswith(prefix)
        for prefix in ["meta-llama/", "Qwen/", "deepseek-ai/", "mistralai/"]
    ):
        # Together AI models
        content = []
        new_msg_history = []
        for _ in range(n_responses):
            together_msg_history = msg_history + [{"role": "user", "content": msg}]
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *together_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            resp_content = response.choices[0].message.content
            content.append(resp_content)
            updated_history = together_msg_history + [
                {"role": "assistant", "content": resp_content}
            ]
            new_msg_history.append(updated_history)
            if hasattr(response, "usage"):
                input_tokens = getattr(response.usage, "prompt_tokens", 0)
                output_tokens = getattr(response.usage, "completion_tokens", 0)
                if cost_tracker is not None:
                    cost_tracker.add_cost(model, input_tokens, output_tokens, task_name)
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
                cost_tracker=cost_tracker,
                task_name=task_name,
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
    cost_tracker: Optional[BudgetChecker] = None,
    task_name: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if msg_history is None:
        msg_history = []

    input_tokens = 0
    output_tokens = 0
    response = None
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
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
    elif any(
        model.startswith(prefix) for prefix in ["ollama/", "lm_studio/", "openai/"]
    ):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model.split("/")[1],
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
    elif any(
        model.startswith(prefix)
        for prefix in ["meta-llama/", "Qwen/", "deepseek-ai/", "mistralai/"]
    ):
        # Together AI models
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
        if hasattr(response, "usage"):
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
    else:
        raise ValueError(f"Model {model} not supported.")

    if cost_tracker is not None:
        cost_tracker.add_cost(model, input_tokens, output_tokens, task_name)

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


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm_with_tools(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    tools: List[Dict[str, Any]],
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.75,
    n_responses: int = 1,
    cost_tracker: Optional[BudgetChecker] = None,
    task_name: Optional[str] = None,
) -> Tuple[List[Union[str, Dict[str, Any]]], List[List[Dict[str, str]]]]:
    """
    Gets batch responses from LLM, potentially including tool calls.
    Currently primarily supports OpenAI models.
    Returns a list of responses (string or tool call dict) and the updated message histories.
    """
    if msg_history is None:
        msg_history = []

    all_responses: List[Union[str, Dict[str, Any]]] = []
    all_new_histories: List[List[Dict[str, str]]] = []

    # Assuming OpenAI client for tool calling
    if isinstance(client, openai.OpenAI) and (
        "gpt" in model or model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]
    ):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        try:
            response = client.chat.completions.create(  # type: ignore[call-overload]
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                tools=tools,
                tool_choice="auto",  # Or specify a tool like {"type": "function", "function": {"name": "my_function"}}
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                stop=None,
                seed=0,  # Seed might not be available for all models or with tool use
            )

            # Extract token usage for OpenAI
            input_tokens = (
                getattr(response.usage, "prompt_tokens", 0)
                if hasattr(response, "usage")
                else 0
            )
            output_tokens = (
                getattr(response.usage, "completion_tokens", 0)
                if hasattr(response, "usage")
                else 0
            )
            if cost_tracker is not None:
                cost_tracker.add_cost(model, input_tokens, output_tokens, task_name)

            for choice in response.choices:
                response_message = choice.message
                current_history = new_msg_history + [
                    response_message.model_dump(exclude_unset=True)
                ]  # Add assistant response (tool call or text)

                if response_message.tool_calls:
                    # Store tool call information
                    tool_call_info = {
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response_message.tool_calls
                        ]
                    }
                    all_responses.append(tool_call_info)

                    # Note: History for tool calls needs the tool response added later
                    # For simplicity here, we just append the assistant's request
                    all_new_histories.append(current_history)

                else:
                    # Store text response
                    content = response_message.content or ""
                    all_responses.append(content)
                    all_new_histories.append(
                        current_history
                    )  # History is complete here

        except Exception as e:
            print(f"Error during LLM call with tools: {e}")
            # Fallback or error handling
            for _ in range(n_responses):
                all_responses.append(f"Error: {e}")
                all_new_histories.append(
                    msg_history
                    + [
                        {"role": "user", "content": msg},
                        {"role": "assistant", "content": f"Error: {e}"},
                    ]
                )

    # Handle Together AI models with function calling
    elif hasattr(client, "together") and any(
        model.startswith(prefix)
        for prefix in ["meta-llama/", "Qwen/", "deepseek-ai/", "mistralai/"]
    ):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        try:
            # Try to use function calling with Together AI models
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                tools=tools,
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
            )

            input_tokens = (
                getattr(response, "prompt_tokens", 0)
                if hasattr(response, "prompt_tokens")
                else 0
            )
            output_tokens = (
                getattr(response, "completion_tokens", 0)
                if hasattr(response, "completion_tokens")
                else 0
            )
            if cost_tracker is not None:
                cost_tracker.add_cost(model, input_tokens, output_tokens, task_name)

            for choice in response.choices:
                response_message = choice.message
                current_history = new_msg_history + [
                    {"role": "assistant", "content": response_message.content or ""}
                ]

                if (
                    hasattr(response_message, "tool_calls")
                    and response_message.tool_calls
                ):
                    # Store tool call information for Together AI
                    tool_call_info = {
                        "tool_calls": [
                            {
                                "id": tc.id if hasattr(tc, "id") else f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for i, tc in enumerate(response_message.tool_calls)
                        ]
                    }
                    all_responses.append(tool_call_info)
                    all_new_histories.append(current_history)
                else:
                    # Store text response
                    content = response_message.content or ""
                    all_responses.append(content)
                    all_new_histories.append(current_history)

        except Exception as e:
            print(f"Error during Together AI call with tools: {e}")
            for _ in range(n_responses):
                all_responses.append(f"Error with Together AI tool calls: {e}")
                all_new_histories.append(
                    msg_history
                    + [
                        {"role": "user", "content": msg},
                        {"role": "assistant", "content": f"Error: {e}"},
                    ]
                )

    else:
        # Fallback for models without direct tool support or non-OpenAI clients
        print(
            f"[WARNING] Tool calling requested for model '{model}' which might not have direct support in this implementation. Falling back to standard generation."
        )
        contents, histories = get_batch_responses_from_llm(
            msg=msg,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=print_debug,
            msg_history=msg_history,
            temperature=temperature,
            n_responses=n_responses,
            cost_tracker=cost_tracker,
            task_name=task_name,
        )
        for item in contents:
            all_responses.append(item)
        all_new_histories = histories

    if print_debug:
        print()
        print("*" * 20 + " LLM (Tools) START " + "*" * 20)
        for i, (resp, history) in enumerate(zip(all_responses, all_new_histories)):
            print(f"Response {i}: {resp}")
            # print(f"History {i}: {history}") # History might be long
        print("*" * 21 + " LLM (Tools) END " + "*" * 21)
        print()

    return all_responses, all_new_histories


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
        Any,  # Together AI client
    ],
    str,
]:
    llm_api_key = config["core"].get("llm_api_key", "")
    client: Union[
        anthropic.Anthropic,
        anthropic.AnthropicBedrock,
        anthropic.AnthropicVertex,
        openai.OpenAI,
        Any,  # Together AI client
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

    elif model.startswith("ollama"):
        base_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        client = openai.OpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        return client, model

    elif model.startswith("lm_studio"):
        base_url = os.environ.get("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
        client = openai.OpenAI(api_key="lm_studio", base_url=base_url)
        return client, model

    elif model.startswith("openai"):
        api_key = os.environ.get("OPENAI_API_KEY", llm_api_key)
        if not api_key:
            raise ValueError(
                f"Missing API key to use {model}. Set OPENAI_API_KEY or llm_api_key in config.toml."
            )
        base_url = os.environ.get("OPENAI_API_BASE", "http://localhost/v1")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return client, model

    elif any(
        model.startswith(prefix)
        for prefix in ["meta-llama/", "Qwen/", "deepseek-ai/", "mistralai/"]
    ):
        # Together AI client
        try:
            from together import Together
        except ImportError:
            raise ImportError(
                "To use Together AI models, you need to install the 'together' package: pip install together"
            )

        api_key = os.environ.get("TOGETHER_API_KEY", llm_api_key)
        if not api_key:
            raise ValueError(
                f"Missing Together AI API key to use {model}. Set TOGETHER_API_KEY or llm_api_key in config.toml."
            )

        # Create the Together client and set an attribute to identify it
        client = Together(api_key=api_key)
        client.together = True  # Add this attribute to identify Together client

        return client, model

    else:
        raise ValueError(f"Model {model} not supported.")
