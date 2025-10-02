from __future__ import annotations

from typing import Dict, Optional

from rich import print

from .utils.pricing import calculate_pricing


class BudgetExceededError(Exception):
    """Raised when a call would exceed the configured budget."""


class BudgetChecker:
    """Track API usage cost and enforce a spending budget."""

    def __init__(
        self,
        budget: Optional[float] = None,
        parent: Optional["BudgetChecker"] = None,
    ) -> None:
        self.total_cost = 0.0
        self.per_task_cost: Dict[str, float] = {}
        self.current_task: Optional[str] = None
        self.budget = budget  # in dollars
        self.parent = parent

    def start_task(self, task_name: str) -> None:
        self.current_task = task_name
        if task_name not in self.per_task_cost:
            self.per_task_cost[task_name] = 0.0

    def end_task(self, task_name: Optional[str] = None) -> None:
        if task_name is None:
            task_name = self.current_task
        self.current_task = None

    def add_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_name: Optional[str] = None,
    ) -> float:
        cost = calculate_pricing(model, input_tokens, output_tokens)
        projected_total = self.total_cost + cost

        enforce_locally = (
            self.parent is None or self.parent.budget is None
        ) and self.budget is not None

        if enforce_locally and projected_total > self.budget:
            raise BudgetExceededError(
                f"Budget exceeded! Attempted to add ${cost:.4f}, "
                f"which would bring total to ${projected_total:.4f} (budget: ${self.budget:.4f})"
            )

        if self.parent is not None:
            self.parent.add_cost(
                model,
                input_tokens,
                output_tokens,
                task_name,
            )

        self.total_cost += cost
        if task_name is None:
            task_name = self.current_task
        if task_name:
            if task_name not in self.per_task_cost:
                self.per_task_cost[task_name] = 0.0
            self.per_task_cost[task_name] += cost
        return float(cost)

    def report(self) -> None:
        print(f"Total cost: ${self.total_cost:.4f}")
        for task, cost in self.per_task_cost.items():
            print(f"  Task '{task}': ${cost:.4f}")

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_task_cost(self, task_name: str) -> float:
        return self.per_task_cost.get(task_name, 0.0)

    def get_remaining_budget(self) -> Optional[float]:
        """Return remaining budget if a limit is set."""
        if self.budget is None:
            return None
        return self.budget - self.total_cost

    def get_budget(self) -> Optional[float]:
        """Return the configured budget."""
        return self.budget

    def get_effective_remaining_budget(self) -> Optional[float]:
        """Return the minimum remaining budget considering parent trackers."""

        remaining_candidates = []
        local_remaining = self.get_remaining_budget()
        if local_remaining is not None:
            remaining_candidates.append(local_remaining)

        if self.parent is not None:
            parent_remaining = self.parent.get_effective_remaining_budget()
            if parent_remaining is not None:
                remaining_candidates.append(parent_remaining)

        if not remaining_candidates:
            return None

        return max(0.0, min(remaining_candidates))
