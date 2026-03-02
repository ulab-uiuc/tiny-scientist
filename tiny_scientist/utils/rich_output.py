from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table


console = Console()


def _stringify(value: Any, max_length: int = 120) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.strip().replace("\n", " ")
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=False)
        except TypeError:
            text = str(value)
        text = text.replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def print_mapping_table(
    title: str,
    data: Mapping[str, Any],
    *,
    key_header: str = "Field",
    value_header: str = "Value",
) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column(key_header, style="cyan", no_wrap=True)
    table.add_column(value_header, overflow="fold")
    for key, value in data.items():
        table.add_row(str(key), _stringify(value))
    console.print(table)


def print_rows_table(
    title: str,
    columns: Sequence[Tuple[str, str]],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    table = Table(title=title, show_lines=False)
    for key, header in columns:
        no_wrap = key in {"step", "action", "status"}
        style = "cyan" if key in {"step", "module", "action"} else None
        table.add_column(header, no_wrap=no_wrap, style=style, overflow="fold")
    for row in rows:
        table.add_row(*[_stringify(row.get(key)) for key, _ in columns])
    console.print(table)


def print_stage_progress(
    title: str,
    current: int,
    total: int,
    label: str,
    *,
    status: str = "active",
) -> None:
    safe_total = max(total, 1)
    safe_current = min(max(current, 0), safe_total)
    width = 24
    filled = int(round((safe_current / safe_total) * width))
    bar = "#" * filled + "-" * (width - filled)
    console.print(
        f"[bold cyan]{title}[/] "
        f"[{'green' if status == 'done' else 'cyan'}][{bar}][/] "
        f"{safe_current}/{safe_total} "
        f"[bold]{status.upper()}[/] "
        f"{label}"
    )


def print_task_event(scope: str, task_name: str, event: str, elapsed_s: Optional[float] = None) -> None:
    suffix = f" ({elapsed_s:.1f}s)" if elapsed_s is not None else ""
    console.print(f"[bold magenta][{scope}][/][bold] {event}[/] {task_name}{suffix}")


def print_cost_delta_summary(
    title: str,
    before_total: float,
    before_tasks: Mapping[str, float],
    after_total: float,
    after_tasks: Mapping[str, float],
    *,
    global_before_total: Optional[float] = None,
    global_before_tasks: Optional[Mapping[str, float]] = None,
    global_after_total: Optional[float] = None,
    global_after_tasks: Optional[Mapping[str, float]] = None,
) -> None:
    delta_total = after_total - before_total
    has_global = (
        global_before_total is not None
        and global_before_tasks is not None
        and global_after_total is not None
        and global_after_tasks is not None
    )
    changed_rows = []
    for task, after_cost in after_tasks.items():
        before_cost = before_tasks.get(task, 0.0)
        delta = after_cost - before_cost
        if abs(delta) > 1e-12:
            row = {
                "task": task,
                "delta": f"${delta:.4f}",
                "stage_total": f"${after_cost:.4f}",
            }
            if has_global:
                row["global_total"] = f"${global_after_tasks.get(task, after_cost):.4f}"
            changed_rows.append(row)
    if not changed_rows and abs(delta_total) <= 1e-12:
        row = {
            "task": "(no new tracked cost)",
            "delta": "$0.0000",
            "stage_total": f"${after_total:.4f}",
        }
        if has_global:
            row["global_total"] = f"${global_after_total:.4f}"
        changed_rows.append(row)
    step_row = {
        "task": "STEP TOTAL",
        "delta": f"${delta_total:.4f}",
        "stage_total": f"${after_total:.4f}",
    }
    if has_global:
        step_row["global_total"] = f"${global_after_total:.4f}"
    changed_rows.insert(0, step_row)
    columns = [("task", "Task"), ("delta", "Delta"), ("stage_total", "Stage Total")]
    if has_global:
        columns.append(("global_total", "Global Total"))
    print_rows_table(title, columns, changed_rows)


def print_todo_table(
    stage: str,
    items: List[Dict[str, Any]],
    *,
    include_refs: bool = False,
) -> None:
    columns: List[Tuple[str, str]] = [
        ("step", "Step"),
        ("action", "Action"),
        ("name", "Name"),
        ("description", "Description"),
    ]
    if include_refs:
        columns.append(("refs", "Rows"))
    print_rows_table(f"{stage} TODO", columns, items)


def summarize_idea(idea: Mapping[str, Any]) -> Dict[str, Any]:
    experiment = idea.get("Experiment")
    experiment_metric = (
        experiment.get("Metric")
        if isinstance(experiment, Mapping)
        else None
    )
    experiment_success = (
        experiment.get("Success_Criteria")
        if isinstance(experiment, Mapping)
        else None
    )
    citations = idea.get("Citations")
    return {
        "Title": idea.get("Title") or idea.get("Name"),
        "Problem": idea.get("Problem"),
        "Approach": idea.get("Approach"),
        "Experiment": experiment,
        "Metric": idea.get("Metric") or experiment_metric,
        "Success_Criteria": idea.get("Success_Criteria") or experiment_success,
        "Citations": len(citations) if isinstance(citations, list) else citations,
    }


def summarize_review(review: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "Summary": review.get("Summary") or review.get("summary"),
        "Strengths": len(review.get("Strengths", []))
        if isinstance(review.get("Strengths"), list)
        else review.get("Strengths"),
        "Weaknesses": len(review.get("Weaknesses", []))
        if isinstance(review.get("Weaknesses"), list)
        else review.get("Weaknesses"),
        "Questions": len(review.get("Questions", []))
        if isinstance(review.get("Questions"), list)
        else review.get("Questions"),
        "Score": review.get("Score") or review.get("score"),
        "Confidence": review.get("Confidence") or review.get("confidence"),
        "Decision": review.get("Decision") or review.get("decision"),
    }
