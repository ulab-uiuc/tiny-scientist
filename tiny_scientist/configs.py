import os
from typing import Any, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel

from .data import CoderPrompt, ReviewerPrompt, ThinkerPrompt, WriterPrompt

T = TypeVar("T", bound=BaseModel)


class PromptTemplate(BaseModel):
    """Configuration for prompts."""

    coder_prompt: CoderPrompt
    thinker_prompt: ThinkerPrompt
    reviewer_prompt: ReviewerPrompt
    writer_prompt: WriterPrompt


class Config(BaseModel):
    prompt_template: PromptTemplate

    def __init__(self, prompt_path: Optional[str] = None, **kwargs: Any) -> None:
        if not prompt_path:
            prompt_path = self._default_config_path()

        yaml_data = {"prompt_template": self._load_from_yaml(prompt_path)}
        kwargs.update(yaml_data)
        super().__init__(**kwargs)

    def _default_config_path(self) -> str:
        this_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(this_dir, "./", "prompts"))

    def _load_from_yaml(self, prompt_path: str) -> PromptTemplate:
        return PromptTemplate(
            thinker_prompt=self._load_yaml_file(
                os.path.join(prompt_path, "thinker_prompt.yaml"), ThinkerPrompt
            ),
            coder_prompt=self._load_yaml_file(
                os.path.join(prompt_path, "coder_prompt.yaml"), CoderPrompt
            ),
            writer_prompt=self._load_yaml_file(
                os.path.join(prompt_path, "writer_prompt.yaml"), WriterPrompt
            ),
            reviewer_prompt=self._load_yaml_file(
                os.path.join(prompt_path, "reviewer_prompt.yaml"),
                ReviewerPrompt,
            ),
        )

    def _load_yaml_file(self, file_path: str, model_class: Type[T]) -> T:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file '{file_path}' does not exist.")
        with open(file_path, "r") as f:
            return model_class(**yaml.safe_load(f))
