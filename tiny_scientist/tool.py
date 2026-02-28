"""Tool module for tiny-scientist.

This module re-exports TinyScientist tool implementations for backward compatibility.
"""

from __future__ import annotations

# Re-export all tools from tool_impls for backward compatibility
from tiny_scientist.tool_impls import (
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
