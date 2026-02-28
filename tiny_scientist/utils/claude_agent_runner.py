"""Synchronous wrapper around claude_agent_sdk.query() for use in the pipeline."""

from __future__ import annotations

import asyncio
import inspect
from typing import List, Optional

from ..budget_checker import BudgetChecker


class ClaudeAgentRunner:
    """Wraps claude_agent_sdk.query() with a synchronous interface.

    Matches the usage pattern of Runner.run_sync() so the pipeline can swap
    between OpenAI and Claude runtimes without changing higher-level logic.

    Args:
        instructions: System prompt / role injected at the front of every prompt.
        allowed_tools: Built-in SDK tool names, e.g. ["Bash", "Read", "Write", "Edit"].
        cwd: Working directory for the agent (experiment output_dir).
        permission_mode: SDK permission mode, default "bypassPermissions".
        cost_tracker: Optional BudgetChecker for tracking token usage.
        model: Model identifier string used for cost attribution.
    """

    def __init__(
        self,
        instructions: str,
        allowed_tools: List[str],
        cwd: str,
        permission_mode: str = "bypassPermissions",
        cost_tracker: Optional[BudgetChecker] = None,
        model: str = "",
        mcp_config_path: Optional[str] = None,
    ) -> None:
        self.instructions = instructions
        self.allowed_tools = allowed_tools
        self.cwd = cwd
        self.permission_mode = permission_mode
        self.cost_tracker = cost_tracker
        self.model = model
        self.mcp_config_path = mcp_config_path

    def run_sync(self, prompt: str, task_name: str = "") -> str:
        """Synchronous wrapper — safe to call from non-async code.

        If there is already a running event loop (e.g. inside Jupyter or another
        async framework), the coroutine is submitted to a new thread's loop to
        avoid nesting errors.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside a running loop — run in a separate thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, self._run_async(prompt, task_name)
                )
                return future.result()
        else:
            return asyncio.run(self._run_async(prompt, task_name))

    async def _run_async(self, prompt: str, task_name: str) -> str:
        """Async implementation that streams messages from claude_agent_sdk."""
        from claude_agent_sdk import ClaudeAgentOptions, query

        option_kwargs = {
            "model": self.model,
            "cwd": self.cwd,
            "allowed_tools": self.allowed_tools,
            "permission_mode": self.permission_mode,
            # Load native Claude skills from both user and project scopes.
            "setting_sources": ["user", "project"],
        }

        option_params = inspect.signature(ClaudeAgentOptions).parameters
        if self.mcp_config_path:
            if "mcp_config" in option_params:
                option_kwargs["mcp_config"] = self.mcp_config_path
            elif "mcp_config_path" in option_params:
                option_kwargs["mcp_config_path"] = self.mcp_config_path
            elif "mcp_servers" in option_params:
                # Newer SDKs discover MCP from project settings instead of a direct config path.
                # Ignore the generated config here rather than crashing on an unknown kwarg.
                pass

        options = ClaudeAgentOptions(**option_kwargs)

        full_prompt = (
            self.instructions.rstrip() + "\n\n" + prompt
            if self.instructions
            else prompt
        )
        result_text = ""

        async for message in query(prompt=full_prompt, options=options):
            # ResultMessage carries the final text output
            if hasattr(message, "result") and message.result:
                result_text = message.result

            # Track cost from usage fields on ResultMessage
            if (
                hasattr(message, "usage")
                and message.usage is not None
                and self.cost_tracker is not None
            ):
                input_tokens = getattr(message.usage, "input_tokens", 0) or 0
                output_tokens = getattr(message.usage, "output_tokens", 0) or 0
                if input_tokens or output_tokens:
                    self.cost_tracker.add_cost(
                        self.model,
                        input_tokens,
                        output_tokens,
                        task_name,
                    )

        return result_text
