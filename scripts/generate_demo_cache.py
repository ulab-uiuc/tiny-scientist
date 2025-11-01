#!/usr/bin/env python3
"""
Run a real Tiny Scientist session and snapshot the responses into
frontend/demo_cache/session.json plus the required generated artefacts.

Usage example:
    python scripts/generate_demo_cache.py \
        --intent "Benchmarking adaptive step size strategies using a convex quadratic optimization function" \
        --model gpt-4o \
        --api-key "$OPENAI_API_KEY" \
        --operations-file demo_plan.json

The operations file is optional; if omitted the script only captures the
initial idea generation + evaluation + code/paper/review for idea 0.
See the DEFAULT_PLAN dict at the bottom for the reference format.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_PLAN = {
    "operations": [
        {"action": "generate_children", "parent_index": 0, "context": ""},
        {"action": "code", "idea_index": 0},
        {"action": "write", "idea_index": 0},
        {"action": "code", "idea_index": 1},
        {"action": "write", "idea_index": 1},
        {"action": "code", "idea_index": 2},
        {"action": "write", "idea_index": 2},
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tiny Scientist demo cache")
    parser.add_argument(
        "--base-url", default="http://localhost:5000", help="Backend base URL"
    )
    parser.add_argument(
        "--intent", required=True, help="Research intent to feed into the Thinker"
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="Model name for configure call"
    )
    parser.add_argument("--api-key", default="", help="API key for configure call")
    parser.add_argument(
        "--budget", type=float, default=None, help="Optional budget value"
    )
    parser.add_argument(
        "--budget-preference", default=None, help="Optional budget preference"
    )
    parser.add_argument(
        "--num-ideas", type=int, default=3, help="Number of initial ideas to request"
    )
    parser.add_argument(
        "--output-dir",
        default="frontend/demo_cache",
        help="Directory to dump cache artefacts (default: frontend/demo_cache)",
    )
    parser.add_argument(
        "--generated-root",
        default="generated",
        help="Root directory where backend writes artefacts",
    )
    parser.add_argument(
        "--operations-file",
        type=str,
        default=None,
        help="JSON file describing post-initial operations (generate_children/modify/merge/code/write/review)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional replacement system prompt to send before generation",
    )
    parser.add_argument(
        "--criteria-file",
        type=str,
        default=None,
        help="JSON file with dimension->criteria text to push via /api/set-criteria",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600, increase if writer/coder takes longer)",
    )
    return parser.parse_args()


@dataclass
class IdeaRecord:
    id: str
    payload: Dict[str, Any]
    slot: int  # 1-based slot for consistent folder naming


@dataclass
class CacheContext:
    base_url: str
    intent: str
    num_ideas: int
    generated_root: Path
    output_dir: Path
    timeout: int
    session: requests.Session = field(default_factory=requests.Session)
    idea_counter: int = 0
    ideas: List[IdeaRecord] = field(default_factory=list)
    idea_ids: set[str] = field(default_factory=set)
    operations_log: Dict[str, List[str]] = field(default_factory=dict)
    evaluation_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluation_default: Optional[Dict[str, Any]] = None
    configure_response: Optional[Dict[str, Any]] = None
    prompts_snapshot: Optional[Dict[str, Any]] = None
    initial_payloads: List[Dict[str, Any]] = field(default_factory=list)
    children_payloads: List[Dict[str, Any]] = field(default_factory=list)
    modify_payloads: List[Dict[str, Any]] = field(default_factory=list)
    merge_payloads: List[Dict[str, Any]] = field(default_factory=list)
    code_payloads: List[Dict[str, Any]] = field(default_factory=list)
    write_payloads: List[Dict[str, Any]] = field(default_factory=list)
    review_payloads: List[Dict[str, Any]] = field(default_factory=list)
    last_real_experiment_dir: Optional[str] = None
    last_demo_experiment_dir: Optional[str] = None
    last_real_pdf_path: Optional[str] = None
    last_demo_pdf_path: Optional[str] = None
    ideas_root: Path = field(init=False)
    generated_experiments_root: Path = field(init=False)
    generated_papers_root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.ideas_root = self.output_dir / "ideas"
        self.generated_experiments_root = self.output_dir / "generated" / "experiments"
        self.generated_papers_root = self.output_dir / "generated" / "papers"
        ensure_dir(self.ideas_root)
        ensure_dir(self.generated_experiments_root)
        ensure_dir(self.generated_papers_root)

    def _snapshot_idea(self, record: IdeaRecord) -> None:
        idea_dir = self.ideas_root / record.id
        ensure_dir(idea_dir)
        idea_path = idea_dir / "idea.json"
        with idea_path.open("w", encoding="utf-8") as fh:
            json.dump(record.payload, fh, indent=2)

    def log(self, channel: str, message: str) -> None:
        self.operations_log.setdefault(channel, []).append(message)

    def next_idea_id(self) -> str:
        self.idea_counter += 1
        return f"idea-{self.idea_counter}"

    def _ensure_unique_id(self, candidate: str) -> str:
        base = candidate
        suffix = 1
        while candidate in self.idea_ids:
            candidate = f"{base}-{suffix}"
            suffix += 1
        self.idea_ids.add(candidate)
        return candidate

    def register_ideas(
        self,
        ideas: List[Dict[str, Any]],
        *,
        preassigned_ids: Optional[List[str]] = None,
    ) -> List[IdeaRecord]:
        records: List[IdeaRecord] = []
        for idx, idea in enumerate(ideas):
            copied = copy.deepcopy(idea)
            idea_id = None
            if preassigned_ids and idx < len(preassigned_ids):
                idea_id = preassigned_ids[idx]
            idea_id = idea_id or self.next_idea_id()
            idea_id = self._ensure_unique_id(idea_id)
            copied.setdefault("originalData", {})
            copied["id"] = idea_id
            slot = len(self.ideas) + 1
            record = IdeaRecord(id=idea_id, payload=copied, slot=slot)
            records.append(record)
            self.ideas.append(record)
            self._snapshot_idea(record)
        return records

    def eval_payload(self) -> Dict[str, Any]:
        items = []
        for idea in self.ideas:
            original = copy.deepcopy(idea.payload.get("originalData") or {})
            original["id"] = idea.id
            items.append(original)
        return {"ideas": items, "intent": self.intent}

    def update_evaluation(self, entries: List[Dict[str, Any]]) -> None:
        id_to_entry = {entry["id"]: entry for entry in entries if "id" in entry}
        for idea in self.ideas:
            entry = id_to_entry.get(idea.id)
            if not entry:
                continue
            original = idea.payload.get("originalData") or {}
            name = (
                original.get("Name")
                or original.get("Title")
                or idea.payload.get("title")
                or idea.id
            )
            clean_entry = {
                "noveltyScore": entry.get("noveltyScore"),
                "noveltyReason": entry.get("noveltyReason"),
                "feasibilityScore": entry.get("feasibilityScore"),
                "feasibilityReason": entry.get("feasibilityReason"),
                "impactScore": entry.get("impactScore"),
                "impactReason": entry.get("impactReason"),
            }
            self.evaluation_map[name] = clean_entry
            if self.evaluation_default is None:
                self.evaluation_default = clean_entry


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def request_json(
    ctx: CacheContext, method: str, path: str, *, json_body: Any = None
) -> Any:
    url = f"{ctx.base_url.rstrip('/')}{path}"
    response = ctx.session.request(method, url, json=json_body, timeout=ctx.timeout)
    response.raise_for_status()
    if response.headers.get("Content-Type", "").startswith("application/json"):
        return response.json()
    return response.content


def configure_backend(ctx: CacheContext, args: argparse.Namespace) -> None:
    payload = {
        "model": args.model,
        "api_key": args.api_key,
    }
    if args.budget is not None:
        payload["budget"] = args.budget
    if args.budget_preference:
        payload["budget_preference"] = args.budget_preference
    ctx.configure_response = request_json(
        ctx, "POST", "/api/configure", json_body=payload
    )
    ctx.log("configure", "[record] configure completed")


def capture_prompts(ctx: CacheContext) -> None:
    try:
        ctx.prompts_snapshot = request_json(ctx, "GET", "/api/get-prompts")
        ctx.log("prompts", "[record] fetched prompts snapshot")
    except requests.HTTPError as exc:
        ctx.log("prompts", f"[record] get-prompts failed: {exc}")


def set_system_prompt(ctx: CacheContext, prompt: Optional[str]) -> None:
    if prompt is None:
        return
    request_json(
        ctx, "POST", "/api/set-system-prompt", json_body={"system_prompt": prompt}
    )
    ctx.log("set_system_prompt", "[record] updated system prompt")


def set_criteria(ctx: CacheContext, criteria_map: Dict[str, str]) -> None:
    for dimension, text in criteria_map.items():
        request_json(
            ctx,
            "POST",
            "/api/set-criteria",
            json_body={"dimension": dimension, "criteria": text},
        )
        ctx.log("set_criteria", f"[record] updated {dimension} criteria")


def generate_initial(ctx: CacheContext) -> None:
    payload = {"intent": ctx.intent, "num_ideas": ctx.num_ideas}
    response = request_json(ctx, "POST", "/api/generate-initial", json_body=payload)
    ideas = response.get("ideas", [])
    ctx.initial_payloads.append(response)
    start_index = len(ctx.ideas) + 1
    pre_ids = [f"idea-{start_index + idx}" for idx in range(len(ideas))]
    ctx.register_ideas(ideas, preassigned_ids=pre_ids)
    ctx.log("generate_initial", f"[record] generated {len(ideas)} initial ideas")


def evaluate_now(ctx: CacheContext) -> None:
    payload = ctx.eval_payload()
    response = request_json(ctx, "POST", "/api/evaluate", json_body=payload)
    if not isinstance(response, list):
        ctx.log("evaluate", "[record] unexpected evaluation payload")
        return
    ctx.update_evaluation(response)
    ctx.log("evaluate", f"[record] evaluated {len(response)} ideas")


def generate_children(ctx: CacheContext, parent_index: int, context: str) -> None:
    try:
        parent = ctx.ideas[parent_index]
    except IndexError:
        raise ValueError(f"Invalid parent_index {parent_index}")

    payload = {
        "parent_content": parent.payload.get("content"),
        "context": context,
    }
    response = request_json(ctx, "POST", "/api/generate-children", json_body=payload)
    ideas = response.get("ideas", [])
    ctx.children_payloads.append(response)
    pre_ids = [f"{parent.id}-{idx + 1}" for idx in range(len(ideas))]
    ctx.register_ideas(ideas, preassigned_ids=pre_ids)
    ctx.log(
        "generate_children",
        f"[record] generated {len(ideas)} child ideas from {parent.id}",
    )
    evaluate_now(ctx)


def modify_idea(
    ctx: CacheContext,
    original_index: int,
    modifications: List[Dict[str, Any]],
    behind_index: Optional[int],
) -> None:
    try:
        original = ctx.ideas[original_index]
    except IndexError:
        raise ValueError(f"Invalid original_index {original_index}")
    behind_payload = None
    if behind_index is not None:
        try:
            behind_payload = ctx.ideas[behind_index].payload.get("originalData") or {}
        except IndexError:
            raise ValueError(f"Invalid behind_index {behind_index}")

    payload = {
        "original_idea": original.payload.get("originalData") or {},
        "modifications": modifications,
        "behind_idea": behind_payload,
    }
    response = request_json(ctx, "POST", "/api/modify", json_body=payload)
    ctx.modify_payloads.append(response)
    new_id = f"{original.id}-mod"
    ctx.register_ideas([response], preassigned_ids=[new_id])
    ctx.log("modify", f"[record] modified idea {original.id}")
    evaluate_now(ctx)


def merge_ideas(ctx: CacheContext, index_a: int, index_b: int) -> None:
    try:
        idea_a = ctx.ideas[index_a]
        idea_b = ctx.ideas[index_b]
    except IndexError:
        raise ValueError(f"Invalid merge indices {index_a}, {index_b}")

    payload = {
        "idea_a": idea_a.payload.get("originalData") or {},
        "idea_b": idea_b.payload.get("originalData") or {},
    }
    response = request_json(ctx, "POST", "/api/merge", json_body=payload)
    ctx.merge_payloads.append(response)
    new_id = f"{idea_a.id}-merge-{idea_b.id}"
    ctx.register_ideas([response], preassigned_ids=[new_id])
    ctx.log("merge", f"[record] merged {idea_a.id} & {idea_b.id}")
    evaluate_now(ctx)


def generate_code(
    ctx: CacheContext, idea_index: int, baseline_results: Dict[str, Any]
) -> None:
    try:
        idea = ctx.ideas[idea_index]
    except IndexError:
        raise ValueError(f"Invalid idea_index {idea_index}")

    payload = {
        "idea": idea.payload,
        "baseline_results": baseline_results,
    }
    response = request_json(ctx, "POST", "/api/code", json_body=payload)
    original_experiment_dir = response.get("experiment_dir")
    ctx.last_real_experiment_dir = original_experiment_dir
    ctx.last_demo_experiment_dir = None
    ctx.log("code", f"[record] generated code for {idea.id}")

    if original_experiment_dir:
        src_dir = ctx.generated_root / original_experiment_dir
        dest_dir = ctx.generated_experiments_root / idea.id
        if src_dir.exists():
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
            ctx.log(
                "code",
                f"[record] copied experiment artefacts from {src_dir} -> {dest_dir}",
            )
        else:
            ctx.log("code", f"[warn] experiment directory missing: {src_dir}")
        response["experiment_dir"] = f"experiments/{idea.id}"
        ctx.last_demo_experiment_dir = response["experiment_dir"]

    ctx.code_payloads.append(response)


def generate_paper(
    ctx: CacheContext, idea_index: int, s2_api_key: Optional[str]
) -> None:
    try:
        idea = ctx.ideas[idea_index]
    except IndexError:
        raise ValueError(f"Invalid idea_index {idea_index}")

    payload = {
        "idea": idea.payload,
        "experiment_dir": ctx.last_real_experiment_dir,
    }
    if s2_api_key:
        payload["s2_api_key"] = s2_api_key

    response = request_json(ctx, "POST", "/api/write", json_body=payload)
    original_pdf_path = response.get("pdf_path")
    ctx.last_real_pdf_path = original_pdf_path
    ctx.last_demo_pdf_path = None
    ctx.log("write", f"[record] generated paper for {idea.id}")

    dest_path: Optional[Path] = None
    if original_pdf_path:
        local_pdf = response.get("local_pdf_path")
        if local_pdf and Path(local_pdf).exists():
            src_path = Path(local_pdf)
            dest_dir = ctx.generated_papers_root / idea.id
            ensure_dir(dest_dir)
            dest_path = dest_dir / src_path.name
            shutil.copy2(src_path, dest_path)
        else:
            pdf_content = request_json(ctx, "GET", original_pdf_path)
            dest_dir = ctx.generated_papers_root / idea.id
            ensure_dir(dest_dir)
            dest_path = dest_dir / Path(original_pdf_path).name
            if isinstance(pdf_content, bytes):
                dest_path.write_bytes(pdf_content)
    if dest_path:
        relative_pdf = f"/api/files/papers/{idea.id}/{dest_path.name}"
        response["pdf_path"] = relative_pdf
        response["local_pdf_path"] = str(dest_path)
        ctx.last_demo_pdf_path = relative_pdf
        ctx.log(
            "write",
            f"[record] captured paper artefact at {dest_path}",
        )
    else:
        ctx.log("write", "[warn] unable to capture demo PDF snapshot")
    ctx.write_payloads.append(response)


def review_paper(ctx: CacheContext, s2_api_key: Optional[str]) -> None:
    if not ctx.last_real_pdf_path:
        ctx.log("review", "[warn] skipping review — no pdf_path captured")
        return
    payload = {"pdf_path": ctx.last_real_pdf_path}
    if s2_api_key:
        payload["s2_api_key"] = s2_api_key
    response = request_json(ctx, "POST", "/api/review", json_body=payload)
    ctx.review_payloads.append(response)
    ctx.log("review", "[record] generated review")


def build_session_payload(ctx: CacheContext) -> Dict[str, Any]:
    configure_response = ctx.configure_response or {}
    idea_index_map = {record.id: record.slot for record in ctx.ideas}

    def attach_index(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for entry in entries:
            idea_id = None
            if isinstance(entry, dict):
                idea_data = entry.get("idea")
                if isinstance(idea_data, dict):
                    idea_id = idea_data.get("id")
                idea_id = idea_id or entry.get("id")
            record = copy.deepcopy(entry)
            if idea_id and idea_id in idea_index_map:
                record["_cached_idea_index"] = idea_index_map[idea_id]
            enriched.append(record)
        return enriched

    return {
        "intent": ctx.intent,
        "generated_root": "generated",
        "configure": {
            "session": {
                "model": configure_response.get("model"),
                "configured": True,
                "budget": configure_response.get("budget"),
                "budget_preference": configure_response.get("budget_preference"),
            },
            "response": configure_response,
        },
        "prompts": ctx.prompts_snapshot or {},
        "generate_initial": ctx.initial_payloads,
        "generate_children": ctx.children_payloads,
        "modify": attach_index(ctx.modify_payloads),
        "merge": attach_index(ctx.merge_payloads),
        "evaluation": {
            "by_name": ctx.evaluation_map,
            "default": ctx.evaluation_default,
        },
        "code": attach_index(ctx.code_payloads),
        "write": attach_index(ctx.write_payloads),
        "review": attach_index(ctx.review_payloads),
        "logs": ctx.operations_log,
    }


def load_operations(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_PLAN
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    generated_root = Path(args.generated_root).resolve()
    ensure_dir(output_dir)

    ctx = CacheContext(
        base_url=args.base_url,
        intent=args.intent,
        num_ideas=args.num_ideas,
        generated_root=generated_root,
        output_dir=output_dir,
        timeout=args.timeout,
    )

    configure_backend(ctx, args)
    capture_prompts(ctx)
    if args.system_prompt:
        set_system_prompt(ctx, args.system_prompt)
    if args.criteria_file:
        with open(args.criteria_file, "r", encoding="utf-8") as fh:
            criteria_map = json.load(fh)
        set_criteria(ctx, criteria_map)

    generate_initial(ctx)
    evaluate_now(ctx)

    plan = load_operations(args.operations_file)
    for step in plan.get("operations", []):
        action = step.get("action")
        if action == "generate_children":
            generate_children(
                ctx,
                parent_index=step.get("parent_index", 0),
                context=step.get("context", ""),
            )
        elif action == "modify":
            modify_idea(
                ctx,
                original_index=step.get("idea_index", 0),
                modifications=step.get("modifications", []),
                behind_index=step.get("behind_index"),
            )
        elif action == "merge":
            merge_ideas(
                ctx,
                index_a=step.get("idea_a_index", 0),
                index_b=step.get("idea_b_index", 1),
            )
        elif action == "evaluate":
            evaluate_now(ctx)
        elif action == "code":
            generate_code(
                ctx,
                idea_index=step.get("idea_index", 0),
                baseline_results=step.get("baseline_results", {}),
            )
        elif action == "write":
            generate_paper(
                ctx,
                idea_index=step.get("idea_index", 0),
                s2_api_key=step.get("s2_api_key"),
            )
        elif action == "review":
            review_paper(ctx, s2_api_key=step.get("s2_api_key"))
        else:
            raise ValueError(f"Unsupported action: {action}")

    session_payload = build_session_payload(ctx)
    output_path = output_dir / "session.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(session_payload, fh, indent=2)
    ctx.log("summary", f"[record] wrote session to {output_path}")

    print(f"Demo cache captured at {output_path}")
    print("Operations log:")
    for channel, messages in ctx.operations_log.items():
        for message in messages:
            print(f"  [{channel}] {message}")


if __name__ == "__main__":
    main()
