"""Lightweight internal tool base class.

This keeps TinyScientist runtime independent from any specific agent framework.
"""

from __future__ import annotations

import abc
from typing import Any


class Tool(abc.ABC):
    """Framework-neutral tool protocol used by TinyScientist tools."""

    name = ""
    description = ""
    inputs: dict[str, Any] = {}
    output_type = "object"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
