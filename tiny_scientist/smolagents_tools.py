"""Backward-compatible re-export of TinyScientist tool implementations.

The runtime no longer depends on smolagents. Import from `tiny_scientist.tool_impls`
for the canonical module name.
"""

from __future__ import annotations

from tiny_scientist.tool_impls import *  # noqa: F401,F403
