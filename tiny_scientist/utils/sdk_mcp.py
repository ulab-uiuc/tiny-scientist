"""Shared helpers for wiring Tiny Scientist MCP servers into SDK backends."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SDKMCPServerSpec:
    name: str
    command: List[str]
    tool_names: List[str]
    cwd: Path = PACKAGE_ROOT
    env: Optional[Dict[str, str]] = None


def build_research_mcp_specs(include_drawer: bool = False) -> List[SDKMCPServerSpec]:
    specs = [
        SDKMCPServerSpec(
            name="tiny_scientist_research",
            command=[sys.executable, "-m", "tiny_scientist.mcp.research_server"],
            tool_names=[
                "paper_search",
                "code_search",
                "web_search",
                "scholar_graph_search",
                "patent_search",
                "dataset_search",
                "benchmark_search",
                "arxiv_daily_watch",
                "news_search",
                "repo_runtime_probe",
                "table_extractor",
                "claim_verifier",
            ],
        )
    ]
    if include_drawer:
        specs.append(
            SDKMCPServerSpec(
                name="tiny_scientist_drawer",
                command=[sys.executable, "-m", "tiny_scientist.mcp.drawer_server"],
                tool_names=["drawer.run"],
            )
        )
    return specs


def ensure_mcp_config(
    cwd: str,
    include_drawer: bool = False,
) -> str:
    """Write a generated Claude-compatible MCP config file and return its path."""
    specs = build_research_mcp_specs(include_drawer=include_drawer)
    config = {
        "mcpServers": {
            spec.name: {
                "command": spec.command[0],
                "args": spec.command[1:],
                "cwd": str(spec.cwd),
                **({"env": spec.env} if spec.env else {}),
            }
            for spec in specs
        }
    }
    config_path = Path(cwd) / ".tiny_scientist.generated.mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path)


def claude_allowed_mcp_tools(include_drawer: bool = False) -> List[str]:
    """Return Claude Code SDK tool names for the generated MCP servers."""
    names: List[str] = []
    for spec in build_research_mcp_specs(include_drawer=include_drawer):
        for tool_name in spec.tool_names:
            names.append(f"mcp__{spec.name}__{tool_name}")
    return names
