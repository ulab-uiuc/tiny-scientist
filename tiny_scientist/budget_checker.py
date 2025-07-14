from typing import Dict, Optional

from rich import print

from .utils.pricing import calculate_pricing


class BudgetExceededError(Exception):
    """Raised when a call would exceed the configured budget."""


class BudgetChecker:
    """Track API usage cost and enforce a spending budget."""

    def __init__(self, budget: Optional[float] = None) -> None:
        self.total_cost = 0.0
        self.per_task_cost: Dict[str, float] = {}
        self.current_task: Optional[str] = None
        self.budget = budget  # in dollars

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
        if self.budget is not None and self.total_cost + cost > self.budget:
            raise BudgetExceededError(
                f"Budget exceeded! Attempted to add ${cost:.4f}, "
                f"which would bring total to ${self.total_cost + cost:.4f} (budget: ${self.budget:.4f})"
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
