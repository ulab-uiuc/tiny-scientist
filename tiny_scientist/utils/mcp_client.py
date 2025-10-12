"""Helpers for calling Tiny Scientist MCP servers via stdio."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: list[str]
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None


SERVER_SPECS: Dict[str, MCPServerSpec] = {
    "code_search": MCPServerSpec(
        name="code_search",
        command=[sys.executable, "-m", "tiny_scientist.mcp.code_search_server"],
        cwd=PACKAGE_ROOT,
    ),
    "paper_search": MCPServerSpec(
        name="paper_search",
        command=[sys.executable, "-m", "tiny_scientist.mcp.paper_search_server"],
        cwd=PACKAGE_ROOT,
    ),
    "drawer": MCPServerSpec(
        name="drawer",
        command=[sys.executable, "-m", "tiny_scientist.mcp.drawer_server"],
        cwd=PACKAGE_ROOT,
    ),
    "docker_runner": MCPServerSpec(
        name="docker_runner",
        command=[sys.executable, "-m", "tiny_scientist.mcp.docker_runner_server"],
        cwd=PACKAGE_ROOT,
    ),
}


def _format_tool_result(result: CallToolResult) -> Any:
    if result.structuredContent is not None:
        return result.structuredContent

    chunks: list[str] = []
    for block in result.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            chunks.append(getattr(block, "text", ""))
        elif block_type == "image":
            chunks.append("<image>")
        else:
            chunks.append(str(block))

    combined = "\n".join(chunk for chunk in chunks if chunk)
    if combined:
        try:
            return json.loads(combined)
        except json.JSONDecodeError:
            return combined
    return None


async def _call_tool_async(spec: MCPServerSpec, tool: str, args: Dict[str, Any]) -> Any:
    command, *args_list = spec.command
    params = StdioServerParameters(
        command=command,
        args=args_list,
        cwd=str(spec.cwd) if spec.cwd else None,
        env=spec.env,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)

    if result.isError:
        details = _format_tool_result(result)
        raise RuntimeError(f"MCP tool {tool} returned error: {details}")

    return _format_tool_result(result)


def call_mcp_tool(server: str, tool: str, args: Dict[str, Any]) -> Any:
    spec = SERVER_SPECS.get(server)
    if spec is None:
        raise ValueError(f"Unknown MCP server '{server}'")

    return anyio.run(_call_tool_async, spec, tool, args)
