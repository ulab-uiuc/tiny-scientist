import os
from typing import Any, Dict, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel

from .data import CoderPrompt, ReviewerPrompt, ThinkerPrompt, WriterPrompt

T = TypeVar("T", bound=BaseModel)

class ParamConfig(BaseModel):
    """Configuration for parameter tuning."""
    pass


class PromptTemplate(BaseModel):
    """Configuration for prompts."""
    coder_prompt: CoderPrompt
    thinker_prompt: ThinkerPrompt
    reviewer_prompt: ReviewerPrompt
    writer_prompt: WriterPrompt


class Config(BaseModel):
    param: ParamConfig
    prompt_template: PromptTemplate

    def __init__(self, yaml_config_path: Optional[str] = None, **kwargs: Any) -> None:
        yaml_config_path = yaml_config_path or self._default_config_path()

        yaml_data = self._load_from_yaml(yaml_config_path)
        kwargs.update(yaml_data)
        super().__init__(**kwargs)

    def _default_config_path(self) -> str:
        this_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(this_dir, "..", "configs"))

    def _load_from_yaml(self, yaml_config_path: str) -> Dict[str, Any]:
        return {
            'param': self._load_yaml_file(os.path.join(yaml_config_path, 'param.yaml'), ParamConfig),
            'prompt_template': PromptTemplate(
                thinker_prompt=self._load_yaml_file(os.path.join(yaml_config_path, 'thinker_prompt.yaml'), ThinkerPrompt),
                coder_prompt=self._load_yaml_file(os.path.join(yaml_config_path, 'coder_prompt.yaml'), CoderPrompt),
                writer_prompt=self._load_yaml_file(os.path.join(yaml_config_path, 'writer_prompt.yaml'), WriterPrompt),
                reviewer_prompt=self._load_yaml_file(os.path.join(yaml_config_path, 'reviewer_prompt.yaml'), ReviewerPrompt),
            )
        }

    def _load_yaml_file(self, file_path: str, model_class: Type[T]) -> T:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file '{file_path}' does not exist.")
        with open(file_path, 'r') as f:
            return model_class(**yaml.safe_load(f))