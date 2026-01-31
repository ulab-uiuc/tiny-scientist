"""Tool module for tiny-scientist.

This module re-exports the smolagents-based tools for backward compatibility.
All tool implementations have been migrated to smolagents_tools.py.
"""

from __future__ import annotations

# Re-export all tools from smolagents_tools for backward compatibility
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
    "PaperSearchTool",
    "CodeSearchTool",
    "DrawerTool",
    "DockerExperimentRunner",
    "WriteFileTool",
    "ReadFileTool",
    "RunExperimentTool",
]
