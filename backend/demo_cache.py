from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class DemoCacheError(RuntimeError):
    """Raised when demo cache data is missing or inconsistent."""


class DemoCacheService:
    """Utility to replay a pre-recorded Tiny Scientist session for demos."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        enabled: bool,
        log_fn: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.enabled = bool(enabled)
        self._log_fn = log_fn or (lambda message, level="info": None)

        self.generated_root = "generated"
        self.generated_base = self.base_dir / self.generated_root

        self._queues: Dict[str, List[Any]] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._evaluation_by_name: Dict[str, Dict[str, Any]] = {}
        self._evaluation_default: Optional[Dict[str, Any]] = None
        self._prompts_state: Dict[str, Any] = {
            "system_prompt": "",
            "criteria": {},
            "defaults": {},
        }
        self._configure_state: Dict[str, Any] = {}
        self._logs: Dict[str, List[Any]] = {}
        self.intent: Optional[str] = None
        self._idea_names_by_id: defaultdict[str, set[str]] = defaultdict(set)
        self._idea_id_by_name: Dict[str, str] = {}
        self._code_results_by_id: Dict[str, Any] = {}
        self._code_results_by_experiment: Dict[str, Any] = {}
        self._paper_results_by_id: Dict[str, Any] = {}
        self._paper_results_by_path: Dict[str, Any] = {}
        self._review_results_by_id: Dict[str, Any] = {}
        self._review_results_by_path: Dict[str, Any] = {}

        if self.enabled:
            self._load()

    def _load(self) -> None:
        session_path = self.base_dir / "session.json"
        if not session_path.exists():
            raise DemoCacheError(
                f"Demo cache session file not found at {session_path!s}"
            )

        with session_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self.intent = payload.get("intent")
        self.generated_root = payload.get("generated_root") or "generated"
        self.generated_base = self.base_dir / self.generated_root

        self._configure_state = payload.get("configure") or {}

        prompts = payload.get("prompts") or {}
        defaults = prompts.get("defaults") or {}
        if not defaults and prompts.get("criteria"):
            defaults = copy.deepcopy(prompts["criteria"])

        self._prompts_state = {
            "system_prompt": prompts.get("system_prompt", ""),
            "criteria": copy.deepcopy(prompts.get("criteria") or {}),
            "defaults": copy.deepcopy(defaults),
        }

        evaluation_payload = payload.get("evaluation") or {}
        self._evaluation_by_name = copy.deepcopy(
            evaluation_payload.get("by_name") or {}
        )
        default_entry = evaluation_payload.get("default")
        self._evaluation_default = (
            copy.deepcopy(default_entry) if default_entry else None
        )

        self._logs = {}
        raw_logs = payload.get("logs") or {}
        for key, messages in raw_logs.items():
            if isinstance(messages, list):
                self._logs[key] = list(messages)
            elif isinstance(messages, dict):
                self._logs[key] = [messages]
            else:
                self._logs[key] = [messages]

        queue_keys = [
            "generate_initial",
            "generate_children",
            "modify",
            "merge",
            "code",
            "write",
            "review",
        ]
        for key in queue_keys:
            raw_queue = payload.get(key)
            if raw_queue is None:
                self._queues[key] = []
            elif isinstance(raw_queue, dict):
                self._queues[key] = [copy.deepcopy(raw_queue)]
            else:
                self._queues[key] = [copy.deepcopy(item) for item in raw_queue]

        self._build_name_index()
        self._rebuild_result_indices()

    def _emit_logs(self, channel: str) -> None:
        if not self.enabled:
            return

        messages = self._logs.get(channel) or []
        for entry in messages:
            if isinstance(entry, dict):
                message = entry.get("message")
                level = entry.get("level", "info")
            else:
                message = str(entry)
                level = "info"

            if message:
                self._log_fn(message, level)

    def _normalize_name(self, name: Optional[str]) -> Optional[str]:
        if not name or not isinstance(name, str):
            return None
        normalized = " ".join(name.strip().split()).lower()
        return normalized or None

    def _extract_idea_id(self, value: Optional[str]) -> Optional[str]:
        if not value or not isinstance(value, str):
            return None
        sanitized = value.replace("\\", "/")
        match = re.search(r"(idea[-_][0-9a-z]+)", sanitized, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None

    def _build_name_index(self) -> None:
        self._idea_names_by_id = defaultdict(set)
        self._idea_id_by_name = {}

        initial_entries = self._queues.get("generate_initial") or []
        for entry in initial_entries:
            ideas = entry.get("ideas") if isinstance(entry, dict) else None
            if not isinstance(ideas, list):
                continue
            for idea in ideas:
                if not isinstance(idea, dict):
                    continue
                idea_id = idea.get("id")
                normalized_id = (
                    idea_id.strip().lower() if isinstance(idea_id, str) else None
                )
                if not normalized_id:
                    continue

                candidates = [
                    idea.get("title"),
                    idea.get("name"),
                    idea.get("Title"),
                    idea.get("Name"),
                ]
                original = idea.get("originalData")
                if isinstance(original, dict):
                    candidates.extend(
                        [original.get("Title"), original.get("Name")]
                    )

                for candidate in candidates:
                    normalized = self._normalize_name(candidate)
                    if normalized:
                        self._idea_names_by_id[normalized_id].add(normalized)

        for idea_id, names in self._idea_names_by_id.items():
            for normalized in names:
                self._idea_id_by_name.setdefault(normalized, idea_id)

    def _rebuild_result_indices(self) -> None:
        self._code_results_by_id = {}
        self._code_results_by_experiment = {}
        for entry in self._queues.get("code") or []:
            if not isinstance(entry, dict):
                continue
            experiment_dir = entry.get("experiment_dir")
            idea_id = self._extract_idea_id(experiment_dir)
            if idea_id:
                self._code_results_by_id.setdefault(idea_id, copy.deepcopy(entry))
            if isinstance(experiment_dir, str):
                self._code_results_by_experiment.setdefault(
                    experiment_dir, copy.deepcopy(entry)
                )

        self._paper_results_by_id = {}
        self._paper_results_by_path = {}
        for entry in self._queues.get("write") or []:
            if not isinstance(entry, dict):
                continue
            idea_id = (
                self._extract_idea_id(entry.get("pdf_path"))
                or self._extract_idea_id(entry.get("local_pdf_path"))
                or self._extract_idea_id(entry.get("paper_name"))
            )
            if idea_id:
                self._paper_results_by_id.setdefault(idea_id, copy.deepcopy(entry))
            pdf_path = entry.get("pdf_path")
            if isinstance(pdf_path, str):
                self._paper_results_by_path.setdefault(pdf_path, copy.deepcopy(entry))

        self._review_results_by_id = {}
        self._review_results_by_path = {}
        for entry in self._queues.get("review") or []:
            if not isinstance(entry, dict):
                continue
            pdf_path = entry.get("pdf_path")
            if isinstance(pdf_path, str):
                self._review_results_by_path.setdefault(pdf_path, copy.deepcopy(entry))
            idea_id = self._extract_idea_id(pdf_path)
            if idea_id:
                self._review_results_by_id.setdefault(idea_id, copy.deepcopy(entry))

    def _resolve_candidate_ids(
        self,
        idea_id: Optional[str] = None,
        idea_name: Optional[str] = None,
        experiment_hint: Optional[str] = None,
    ) -> List[str]:
        candidates: List[str] = []

        for raw in (
            idea_id,
            self._extract_idea_id(experiment_hint),
        ):
            if isinstance(raw, str):
                normalized = raw.strip().lower()
                if normalized and normalized not in candidates:
                    candidates.append(normalized)

        if idea_name:
            normalized = self._normalize_name(idea_name)
            if normalized:
                mapped = self._idea_id_by_name.get(normalized)
                if mapped and mapped not in candidates:
                    candidates.append(mapped)

        return candidates

    def _next(self, key: str) -> Any:
        queue = self._queues.get(key) or []
        if not queue:
            raise DemoCacheError(f"No cached payloads configured for '{key}'")

        index = self._counters[key]
        if index < len(queue):
            payload = queue[index]
            self._counters[key] = index + 1
        else:
            payload = queue[-1]

        self._emit_logs(key)
        return copy.deepcopy(payload)

    def apply_config(self, flask_session: Any) -> Dict[str, Any]:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        session_values = copy.deepcopy(self._configure_state.get("session") or {})
        if session_values:
            for key, value in session_values.items():
                flask_session[key] = value
        else:
            flask_session["configured"] = True

        response_payload = copy.deepcopy(
            self._configure_state.get("response")
            or {
                "status": "configured",
                "model": session_values.get("model", "demo-model"),
                "budget": session_values.get("budget"),
                "budget_preference": session_values.get("budget_preference"),
            }
        )

        self._emit_logs("configure")
        return response_payload

    def get_prompts(self) -> Dict[str, Any]:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        return {
            "system_prompt": self._prompts_state.get("system_prompt", ""),
            "criteria": copy.deepcopy(self._prompts_state.get("criteria") or {}),
            "defaults": copy.deepcopy(self._prompts_state.get("defaults") or {}),
        }

    def update_system_prompt(self, system_prompt: Optional[str]) -> None:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        default_prompt = self._prompts_state.get("defaults", {}).get(
            "system_prompt", ""
        )
        self._prompts_state["system_prompt"] = system_prompt or default_prompt
        self._emit_logs("set_system_prompt")

    def update_criteria(self, dimension: str, criteria: Optional[str]) -> None:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        defaults = self._prompts_state.get("defaults") or {}
        self._prompts_state.setdefault("criteria", {})
        if criteria:
            self._prompts_state["criteria"][dimension] = criteria
        else:
            fallback = defaults.get(dimension, "")
            if fallback:
                self._prompts_state["criteria"][dimension] = fallback
            else:
                self._prompts_state["criteria"].pop(dimension, None)
        self._emit_logs("set_criteria")

    def get_initial_ideas(self, intent: Optional[str] = None) -> Dict[str, Any]:
        _ = intent  # Intent is kept for potential validation/debugging.
        return self._next("generate_initial")

    def get_child_ideas(self) -> Dict[str, Any]:
        return self._next("generate_children")

    def get_modified_idea(self) -> Dict[str, Any]:
        return self._next("modify")

    def get_merged_idea(self) -> Dict[str, Any]:
        return self._next("merge")

    def get_code_result(
        self,
        idea_id: Optional[str] = None,
        idea_name: Optional[str] = None,
        experiment_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        for candidate in self._resolve_candidate_ids(
            idea_id=idea_id, idea_name=idea_name, experiment_hint=experiment_hint
        ):
            payload = self._code_results_by_id.get(candidate)
            if payload is not None:
                self._emit_logs("code")
                return copy.deepcopy(payload)

        return self._next("code")

    def get_paper_result(
        self,
        idea_id: Optional[str] = None,
        idea_name: Optional[str] = None,
        experiment_hint: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        if pdf_path and isinstance(pdf_path, str):
            cached = self._paper_results_by_path.get(pdf_path)
            if cached is not None:
                self._emit_logs("write")
                return copy.deepcopy(cached)

        for candidate in self._resolve_candidate_ids(
            idea_id=idea_id, idea_name=idea_name, experiment_hint=experiment_hint
        ):
            payload = self._paper_results_by_id.get(candidate)
            if payload is not None:
                self._emit_logs("write")
                return copy.deepcopy(payload)

        return self._next("write")

    def get_review_result(
        self,
        idea_id: Optional[str] = None,
        idea_name: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        if pdf_path and isinstance(pdf_path, str):
            cached = self._review_results_by_path.get(pdf_path)
            if cached is not None:
                self._emit_logs("review")
                return copy.deepcopy(cached)

        for candidate in self._resolve_candidate_ids(
            idea_id=idea_id, idea_name=idea_name, experiment_hint=pdf_path
        ):
            payload = self._review_results_by_id.get(candidate)
            if payload is not None:
                self._emit_logs("review")
                return copy.deepcopy(payload)

        return self._next("review")

    def evaluate(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        results: List[Dict[str, Any]] = []
        for idx, idea in enumerate(ideas):
            idea_id = idea.get("id") or f"idea_{idx}"
            name_candidates = [
                idea.get("Name"),
                idea.get("Title"),
                idea.get("title"),
                idea.get("name"),
            ]

            score_entry: Optional[Dict[str, Any]] = None
            for name in name_candidates:
                if isinstance(name, str) and name in self._evaluation_by_name:
                    score_entry = self._evaluation_by_name[name]
                    break

            if score_entry is None:
                score_entry = self._evaluation_default

            if score_entry is None:
                raise DemoCacheError(
                    f"No cached evaluation scores for idea '{name_candidates[0] or idea_id}'"
                )

            payload = copy.deepcopy(score_entry)
            payload["id"] = idea_id
            results.append(payload)

        self._emit_logs("evaluate")
        return results

    def resolve_generated_path(self, relative_path: str) -> Path:
        if not self.enabled:
            raise DemoCacheError("Demo cache mode is disabled")

        sanitized = relative_path.lstrip("/\\")
        candidate = (self.generated_base / sanitized).resolve()
        base = self.generated_base.resolve()
        if not str(candidate).startswith(str(base)):
            raise DemoCacheError(
                f"Attempt to access file outside demo cache: {relative_path}"
            )
        return candidate
