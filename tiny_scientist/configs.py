import os
from typing import Any, Optional, Type, TypeVar

import toml
import yaml
from pydantic import BaseModel

from .data import (
    CoderPrompt,
    DrawerPrompt,
    MCPAgentPrompt,
    MCPConfig,
    ReviewerPrompt,
    ThinkerPrompt,
    WriterPrompt,
)

T = TypeVar("T", bound=BaseModel)


class PromptTemplate(BaseModel):
    """Configuration for prompts."""

    coder_prompt: CoderPrompt
    thinker_prompt: ThinkerPrompt
    reviewer_prompt: ReviewerPrompt
    writer_prompt: WriterPrompt
    drawer_prompt: DrawerPrompt
    mcp_agent_prompt: MCPAgentPrompt


class Config(BaseModel):
    prompt_template: PromptTemplate
    mcp: MCPConfig

    def __init__(
        self,
        prompt_path: Optional[str] = None,
        config_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        if not prompt_path:
            prompt_path = self._default_config_path()

        if not config_file_path:
            config_file_path = self._default_config_file_path()

        # Load MCP configuration from TOML
        mcp_config = self._load_mcp_config(config_file_path)

        yaml_data = {
            "prompt_template": self._load_from_yaml(prompt_path),
            "mcp": mcp_config,
        }
        kwargs.update(yaml_data)
        super().__init__(**kwargs)

    def _default_config_path(self) -> str:
        this_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(this_dir, "./", "prompts"))

    def _default_config_file_path(self) -> str:
        this_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(this_dir, "../", "config.toml"))

    def _load_mcp_config(self, config_file_path: str) -> MCPConfig:
        """Load MCP configuration from TOML file"""
        try:
            if os.path.exists(config_file_path):
                config_data = toml.load(config_file_path)
                mcp_data = config_data.get("mcp", {})
                return MCPConfig(**mcp_data)
            else:
                print(f"[Config] MCP config file not found: {config_file_path}")
                return MCPConfig()
        except Exception as e:
            print(f"[Config] Failed to load MCP config: {e}")
            return MCPConfig()

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
            drawer_prompt=self._load_yaml_file(
                os.path.join(prompt_path, "drawer_prompt.yaml"),
                DrawerPrompt,
            ),
            mcp_agent_prompt=self._load_yaml_file(
                os.path.join(prompt_path, "mcp_agent_prompt.yaml"),
                MCPAgentPrompt,
            ),
        )

    def _load_yaml_file(self, file_path: str, model_class: Type[T]) -> T:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file '{file_path}' does not exist.")
        with open(file_path, "r") as f:
            return model_class(**yaml.safe_load(f))
