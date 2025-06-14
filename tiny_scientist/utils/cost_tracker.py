from typing import Dict, Optional

from tiny_scientist.utils.pricing import calculate_pricing


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.per_task_cost: Dict[str, float] = {}
        self.current_task: Optional[str] = None

    def start_task(self, task_name: str):
        self.current_task = task_name
        if task_name not in self.per_task_cost:
            self.per_task_cost[task_name] = 0.0

    def end_task(self, task_name: Optional[str] = None):
        if task_name is None:
            task_name = self.current_task
        self.current_task = None

    def add_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_name: Optional[str] = None,
    ):
        cost = calculate_pricing(model, input_tokens, output_tokens)
        self.total_cost += cost
        if task_name is None:
            task_name = self.current_task
        if task_name:
            if task_name not in self.per_task_cost:
                self.per_task_cost[task_name] = 0.0
            self.per_task_cost[task_name] += cost
        return cost

    def report(self):
        print(f"Total cost: ${self.total_cost:.4f}")
        for task, cost in self.per_task_cost.items():
            print(f"  Task '{task}': ${cost:.4f}")

    def get_total_cost(self):
        return self.total_cost

    def get_task_cost(self, task_name: str):
        return self.per_task_cost.get(task_name, 0.0)
