"""Tool loader utility for tiny-scientist.

This module provides a unified interface for loading tools either as:
1. Native smolagents Tool classes (default, faster)
2. MCP servers via smolagents ToolCollection (optional, for interoperability)

Usage:
    # Default: Native smolagents tools
    from tiny_scientist.tool_loader import get_paper_search_tool, get_code_search_tool

    paper_tool = get_paper_search_tool()
    code_tool = get_code_search_tool()

    # Alternative: Load from MCP servers
    from tiny_scientist.tool_loader import load_tools_from_mcp

    tools = load_tools_from_mcp(["paper_search", "code_search"])
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tiny_scientist.budget_checker import BudgetChecker

# Native smolagents tools (always available)
from tiny_scientist.smolagents_tools import (
    CodeSearchTool,
    DockerExperimentRunner,
    DrawerTool,
    PaperSearchTool,
    ReadFileTool,
    RunExperimentTool,
    WriteFileTool,
)

__all__ = [
    # Tool getter functions (native)
    "get_paper_search_tool",
    "get_code_search_tool",
    "get_drawer_tool",
    "get_docker_runner",
    "get_write_file_tool",
    "get_read_file_tool",
    "get_run_experiment_tool",
    # MCP loading
    "load_tools_from_mcp",
    "is_mcp_available",
    # Direct tool classes
    "PaperSearchTool",
    "CodeSearchTool",
    "DrawerTool",
    "DockerExperimentRunner",
    "WriteFileTool",
    "ReadFileTool",
    "RunExperimentTool",
]

PACKAGE_ROOT = Path(__file__).resolve().parent


def is_mcp_available() -> bool:
    """Check if MCP dependencies are installed."""
    try:
        import fastmcp  # noqa: F401
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Native Tool Getters (Default)
# =============================================================================


def get_paper_search_tool(
    s2_api_key: Optional[str] = None,
    engine: Optional[str] = None,
    disable_fallback: bool = False,
    cost_tracker: Optional[BudgetChecker] = None,
) -> PaperSearchTool:
    """Get a native PaperSearchTool instance."""
    return PaperSearchTool(
        s2_api_key=s2_api_key,
        engine=engine,
        disable_fallback=disable_fallback,
        cost_tracker=cost_tracker,
    )


def get_code_search_tool(
    cost_tracker: Optional[BudgetChecker] = None,
) -> CodeSearchTool:
    """Get a native CodeSearchTool instance."""
    return CodeSearchTool(cost_tracker=cost_tracker)


def get_drawer_tool(
    model: str,
    prompt_template_dir: Optional[str] = None,
    temperature: float = 0.75,
    cost_tracker: Optional[BudgetChecker] = None,
) -> DrawerTool:
    """Get a native DrawerTool instance."""
    return DrawerTool(
        model=model,
        prompt_template_dir=prompt_template_dir,
        temperature=temperature,
        cost_tracker=cost_tracker,
    )


def get_docker_runner(
    docker_image: str = "tiny-scientist-ml",
    docker_base: str = "python:3.11-slim",
) -> DockerExperimentRunner:
    """Get a DockerExperimentRunner instance."""
    return DockerExperimentRunner(
        docker_image=docker_image,
        docker_base=docker_base,
    )


def get_write_file_tool(output_dir: str) -> WriteFileTool:
    """Get a WriteFileTool instance."""
    return WriteFileTool(output_dir=output_dir)


def get_read_file_tool(output_dir: str) -> ReadFileTool:
    """Get a ReadFileTool instance."""
    return ReadFileTool(output_dir=output_dir)


def get_run_experiment_tool(
    output_dir: str,
    docker_runner: Optional[DockerExperimentRunner] = None,
) -> RunExperimentTool:
    """Get a RunExperimentTool instance."""
    return RunExperimentTool(output_dir=output_dir, docker_runner=docker_runner)


# =============================================================================
# MCP Tool Loading (Optional)
# =============================================================================

# MCP server specifications
MCP_SERVER_SPECS: Dict[str, Dict[str, Any]] = {
    "paper_search": {
        "module": "tiny_scientist.mcp.paper_search_server",
        "description": "Paper search MCP server",
    },
    "code_search": {
        "module": "tiny_scientist.mcp.code_search_server",
        "description": "Code search MCP server",
    },
    "drawer": {
        "module": "tiny_scientist.mcp.drawer_server",
        "description": "Diagram generation MCP server",
    },
    "docker_runner": {
        "module": "tiny_scientist.mcp.docker_runner_server",
        "description": "Docker experiment runner MCP server",
    },
}


def load_tools_from_mcp(
    server_names: Optional[List[str]] = None,
) -> List[Any]:
    """
    Load tools from MCP servers via smolagents ToolCollection.

    Args:
        server_names: List of server names to load. If None, loads all available.
                     Options: "paper_search", "code_search", "drawer", "docker_runner"

    Returns:
        List of smolagents tools loaded from MCP servers.

    Raises:
        ImportError: If MCP dependencies are not installed.

    Example:
        # Load specific tools
        tools = load_tools_from_mcp(["paper_search", "code_search"])

        # Load all tools
        all_tools = load_tools_from_mcp()
    """
    if not is_mcp_available():
        raise ImportError(
            "MCP dependencies not installed. Install with: "
            "pip install tiny-scientist[mcp]"
        )

    from smolagents import ToolCollection

    if server_names is None:
        server_names = list(MCP_SERVER_SPECS.keys())

    all_tools: List[Any] = []

    for name in server_names:
        if name not in MCP_SERVER_SPECS:
            raise ValueError(
                f"Unknown MCP server: {name}. "
                f"Available: {list(MCP_SERVER_SPECS.keys())}"
            )

        spec = MCP_SERVER_SPECS[name]

        try:
            # Load tools from MCP server using smolagents
            tool_collection = ToolCollection.from_mcp(
                name,
                server_parameters={
                    "command": sys.executable,
                    "args": ["-m", spec["module"]],
                    "cwd": str(PACKAGE_ROOT.parent),
                },
            )
            all_tools.extend(tool_collection.tools)
            print(f"[MCP] Loaded tools from {name} server")
        except Exception as e:
            print(f"[MCP] Warning: Failed to load {name} server: {e}")

    return all_tools


def get_tools(
    use_mcp: bool = False,
    server_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Get tools using either native smolagents or MCP approach.

    Args:
        use_mcp: If True, load tools from MCP servers. Default False (native).
        server_names: For MCP mode, which servers to load.
        **kwargs: Additional arguments passed to native tool constructors.

    Returns:
        Dictionary of tool name -> tool instance.

    Example:
        # Native tools (default)
        tools = get_tools()

        # MCP tools
        tools = get_tools(use_mcp=True)
    """
    if use_mcp:
        mcp_tools = load_tools_from_mcp(server_names)
        return {tool.name: tool for tool in mcp_tools}

    # Native tools
    return {
        "paper_search": get_paper_search_tool(
            s2_api_key=kwargs.get("s2_api_key"),
            engine=kwargs.get("engine"),
            cost_tracker=kwargs.get("cost_tracker"),
        ),
        "code_search": get_code_search_tool(
            cost_tracker=kwargs.get("cost_tracker"),
        ),
    }
