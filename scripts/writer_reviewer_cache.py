#!/usr/bin/env python3
"""Regenerate demo cache writer/reviewer artefacts using local Tiny Scientist classes."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tiny_scientist.reviewer import Reviewer
from tiny_scientist.writer import Writer


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def discover_ideas(
    output_dir: Path, selected: Optional[Set[str]]
) -> List[Tuple[str, Dict[str, Any]]]:
    idea_root = output_dir / "ideas"
    experiments_root = output_dir / "generated" / "experiments"
    if not experiments_root.exists():
        raise FileNotFoundError(
            f"Expected experiments directory missing: {experiments_root}"
        )

    payloads: List[Tuple[str, Dict[str, Any]]] = []
    for exp_dir in sorted(p for p in experiments_root.iterdir() if p.is_dir()):
        idea_id = exp_dir.name
        if selected and idea_id not in selected:
            continue
        idea_json = idea_root / idea_id / "idea.json"
        if not idea_json.exists():
            raise FileNotFoundError(f"Missing idea payload for {idea_id}: {idea_json}")
        payload = load_json(idea_json)
        payload.setdefault("id", idea_id)
        payloads.append((idea_id, payload))

    if selected:
        missing = selected - {idea_id for idea_id, _ in payloads}
        if missing:
            raise ValueError(
                f"Requested idea IDs not found with experiments: {', '.join(sorted(missing))}"
            )
    return payloads


def copy_experiments(
    source_root: Path, target_root: Path, idea_ids: Iterable[str]
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for idea_id in idea_ids:
        src = source_root / idea_id
        if not src.exists():
            raise FileNotFoundError(f"Experiment folder missing: {src}")
        dst = target_root / idea_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def extract_idea_id(entry: Dict[str, Any]) -> Optional[str]:
    idea = entry.get("idea")
    if isinstance(idea, dict):
        idea_id = idea.get("id")
        if isinstance(idea_id, str):
            return idea_id
    pdf_path = entry.get("pdf_path") or entry.get("local_pdf_path")
    if isinstance(pdf_path, str) and "papers/" in pdf_path:
        parts = [p for p in pdf_path.split("/") if p]
        try:
            idx = parts.index("papers")
            return parts[idx + 1]
        except (ValueError, IndexError):
            return None
    return None


def merge_entries(
    previous: Sequence[Dict[str, Any]],
    replacements: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    replaced_ids = set(replacements.keys())
    for entry in previous:
        idea_id = extract_idea_id(entry)
        if idea_id and idea_id in replaced_ids:
            continue
        merged.append(entry)
    merged.extend(replacements[idea_id] for idea_id in sorted(replacements.keys()))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay writer/reviewer over cached demo ideas without backend HTTP."
    )
    parser.add_argument(
        "--output-dir",
        default="frontend/demo_cache",
        help="Demo cache directory containing ideas/ and generated/ (default: frontend/demo_cache)",
    )
    parser.add_argument(
        "--generated-root",
        default="generated",
        help="Backend generated directory (default: generated)",
    )
    parser.add_argument(
        "--ideas",
        nargs="*",
        default=None,
        help="Optional subset of idea IDs to process (defaults to all with experiments)",
    )
    parser.add_argument(
        "--intent",
        default="Writer/Reviewer cache regeneration",
        help="Intent string stored in session.json if missing",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model for the writer (default: gpt-4o)",
    )
    parser.add_argument(
        "--writer-template",
        default="acl",
        choices=["acl", "iclr"],
        help="Writer output template (default: acl)",
    )
    parser.add_argument(
        "--prompt-template-dir",
        default=None,
        help="Optional path to prompt templates",
    )
    parser.add_argument(
        "--s2-api-key",
        default=None,
        help="Optional Semantic Scholar API key forwarded to writer/reviewer",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Replay writer only and skip reviewer",
    )
    parser.add_argument(
        "--review-model",
        default=None,
        help="Review model (defaults to --model)",
    )
    parser.add_argument(
        "--review-count",
        type=int,
        default=1,
        help="Reviewer num_reviews parameter (default: 1)",
    )
    parser.add_argument(
        "--review-reflections",
        type=int,
        default=1,
        help="Reviewer num_reflections parameter (default: 1)",
    )
    parser.add_argument(
        "--review-temperature",
        type=float,
        default=0.75,
        help="Reviewer temperature (default: 0.75)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    generated_root = Path(args.generated_root).resolve()

    session_path = output_dir / "session.json"
    session_payload = load_json(session_path)
    session_payload.setdefault("intent", args.intent)
    session_payload.setdefault("logs", {})

    selected = set(args.ideas) if args.ideas else None
    idea_payloads = discover_ideas(output_dir, selected)
    if not idea_payloads:
        print("No matching ideas found. Nothing to do.")
        return

    experiments_source = output_dir / "generated" / "experiments"
    experiments_target = generated_root / "experiments"
    copy_experiments(
        experiments_source,
        experiments_target,
        (idea_id for idea_id, _ in idea_payloads),
    )

    writer_entries: Dict[str, Dict[str, Any]] = {}
    review_entries: Dict[str, Dict[str, Any]] = {}
    logs = session_payload["logs"]
    logs.setdefault("write", [])
    if not args.skip_review:
        logs.setdefault("review", [])

    for index, (idea_id, payload) in enumerate(idea_payloads, start=1):
        idea_payload = copy.deepcopy(payload.get("originalData") or payload)
        idea_payload.setdefault("id", idea_id)

        experiment_dir = experiments_target / idea_id
        experiment_dir_path = str(experiment_dir) if experiment_dir.exists() else None

        paper_dir = generated_root / "papers" / idea_id
        ensure_dir(paper_dir)

        writer = Writer(
            model=args.model,
            output_dir=str(paper_dir),
            template=args.writer_template,
            prompt_template_dir=args.prompt_template_dir,
            s2_api_key=args.s2_api_key,
        )
        print(f"[writer] Running writer for {idea_id}")
        pdf_path_str, paper_name = writer.run(
            idea=idea_payload,
            experiment_dir=experiment_dir_path,
        )
        pdf_path = Path(pdf_path_str).resolve()
        if not pdf_path.exists():
            raise RuntimeError(
                f"Writer reported PDF at {pdf_path}, but the file does not exist."
            )
        relative_pdf = f"/api/files/papers/{idea_id}/{pdf_path.name}"
        writer_entries[idea_id] = {
            "pdf_path": relative_pdf,
            "local_pdf_path": str(pdf_path),
            "paper_name": paper_name,
            "success": True,
            "idea": copy.deepcopy(payload),
            "_cached_idea_index": index,
        }
        logs["write"].append(f"[record] regenerated paper for {idea_id}")

        if args.skip_review:
            continue

        review_dir = output_dir / "reviews" / idea_id
        ensure_dir(review_dir)

        reviewer = Reviewer(
            model=args.review_model or args.model,
            tools=[],
            num_reviews=args.review_count,
            num_reflections=args.review_reflections,
            temperature=args.review_temperature,
            prompt_template_dir=args.prompt_template_dir,
            s2_api_key=args.s2_api_key,
        )
        print(f"[review] Running reviewer for {idea_id}")
        review_payload = reviewer.run(pdf_path=str(pdf_path))
        review_path = review_dir / "review.json"
        with review_path.open("w", encoding="utf-8") as fh:
            json.dump(review_payload, fh, indent=2)
            fh.write("\n")

        review_entries[idea_id] = {
            "pdf_path": relative_pdf,
            "review": review_payload,
            "success": True,
            "_cached_idea_index": index,
        }
        logs["review"].append(f"[record] regenerated review for {idea_id}")

    session_payload["write"] = merge_entries(
        session_payload.get("write") or [], writer_entries
    )
    if not args.skip_review:
        session_payload["review"] = merge_entries(
            session_payload.get("review") or [], review_entries
        )
    save_json(session_path, session_payload)

    idea_list = ", ".join(sorted(writer_entries.keys()))
    print(f"[writer] Completed writer for: {idea_list}")
    if not args.skip_review:
        print(f"[review] Completed reviewer for: {', '.join(sorted(review_entries.keys()))}")


if __name__ == "__main__":
    main()
