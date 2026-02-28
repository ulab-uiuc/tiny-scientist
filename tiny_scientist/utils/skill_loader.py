"""Load SKILL.md files from project skill directories and inject into instructions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

_PROJECT_ROOT = Path(__file__).parents[2]
_SKILL_DIRS = (
    _PROJECT_ROOT / ".agents" / "skills",
    _PROJECT_ROOT / ".claude" / "skills",
)


def iter_skill_files(stage: str) -> Iterable[Path]:
    """Yield stage-matching SKILL.md files from all supported skill roots."""
    seen: set[Path] = set()
    for skills_dir in _SKILL_DIRS:
        if not skills_dir.exists():
            continue
        for path in sorted(skills_dir.glob(f"{stage}-*/SKILL.md")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def load_skill(stage: str) -> str:
    """Load and concatenate all SKILL.md files for a given pipeline stage.

    Files are matched by subdirectory prefix: e.g. stage='thinker' loads
    all .claude/skills/thinker-*/SKILL.md files.

    Args:
        stage: Pipeline stage ('thinker', 'coder', 'writer', 'reviewer')

    Returns:
        Concatenated skill content, or empty string if no skills found.
    """
    skill_files = list(iter_skill_files(stage))
    if not skill_files:
        return ""

    parts = []
    for path in skill_files:
        content = path.read_text(encoding="utf-8").strip()
        skill_name = path.parent.name  # directory name is the skill name
        parts.append(f"### SKILL: {skill_name}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def skill_instructions(stage: str, base_instructions: str) -> str:
    """Augment base agent instructions with relevant skill reference material.

    Appends skill documentation after the base instructions so the agent
    treats it as expert reference guidance for its tasks.

    Args:
        stage: Pipeline stage to load skills for.
        base_instructions: The agent's primary instruction string.

    Returns:
        Combined instruction string with skill content appended.
    """
    skill_content = load_skill(stage)
    if not skill_content:
        return base_instructions

    return (
        base_instructions.rstrip()
        + "\n\n"
        + "=" * 60
        + "\n## REFERENCE SKILLS\n\n"
        + "The following expert skill guides are provided as reference material "
        + "to inform your approach. Apply their frameworks and best practices "
        + "when completing your tasks.\n\n"
        + skill_content
    )


def list_skills() -> dict[str, list[str]]:
    """Return a dict mapping each pipeline stage to its loaded skill directory names."""
    result: dict[str, list[str]] = {}
    for stage in ("thinker", "coder", "writer", "reviewer"):
        dirs = list(iter_skill_files(stage))
        if dirs:
            result[stage] = [d.parent.name for d in dirs]
    return result
