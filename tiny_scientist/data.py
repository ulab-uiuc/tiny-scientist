from typing import Dict, List, Optional

from pydantic import BaseModel


class ReviewerPrompt(BaseModel):
    reviewer_system_prompt_base: str
    reviewer_system_prompt_neg: str
    reviewer_system_prompt_pos: str
    query_prompt: str
    template_instructions: str
    neurips_form: str
    meta_reviewer_system_prompt: str
    reviewer_reflection_prompt: str


class WriterPrompt(BaseModel):
    write_system_prompt: str
    write_system_prompt_related_work: str
    section_tips: Dict[str, str]
    error_list: str
    refinement_prompt: str
    second_refinement_prompt: str
    citation_system_prompt: str
    abstract_prompt: str
    section_prompt: Dict[str, str]
    citation_related_work_prompt: str
    add_citation_prompt: str
    embed_citation_prompt: str
    related_work_prompt: str
    title_refinement_prompt: str
    citation_aider_format: str


class CoderPrompt(BaseModel):
    experiment_keyword_prompt: str
    experiment_prompt: str
    experiment_success_prompt: str
    experiment_error_prompt: str
    experiment_timeout_prompt: str
    plot_initial_prompt: str
    plot_error_prompt: str
    plot_timeout_prompt: str
    notes_prompt: str


class ThinkerPrompt(BaseModel):
    idea_system_prompt: str
    evaluation_system_prompt: str
    idea_evaluation_prompt: str
    modify_idea_prompt: str
    merge_ideas_prompt: str
    query_prompt: str
    rethink_query_prompt: str
    novelty_query_prompt: str
    novelty_system_prompt: str
    idea_first_prompt: str
    idea_reflection_prompt: str
    novelty_prompt: str
    experiment_plan_prompt: str


class DrawerPrompt(BaseModel):
    diagram_system_prompt_base: str
    template_instructions: str
    few_shot_instructions: str
    error_list: str
    refinement_prompt: str


class MCPAgentPrompt(BaseModel):
    """Prompts for MCP Agent operations"""
    system_prompt: str
    planning_prompt: str
    tool_selection_prompt: str
    reflection_prompt: str
    error_handling_prompt: str
    goal_completion_prompt: str


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""
    type: str  # "stdio", "sse", "websocket"
    url: Optional[str] = None  # for sse/websocket connections
    command: Optional[str] = None  # for stdio connections (base command)
    args: Optional[List[str]] = None  # command arguments for stdio connections
    timeout: Optional[int] = 30  # connection timeout in seconds
    max_retries: Optional[int] = 3  # maximum retry attempts
    enabled: bool = True  # whether the server is enabled


class MCPModuleConfig(BaseModel):
    """MCP configuration for a specific module"""
    enabled: bool = False
    servers: List[str] = []  # list of server names to use
    capabilities: List[str] = Field(default_factory=list)  # list of required capabilities
    timeout: Optional[int] = 60  # operation timeout in seconds
    max_tool_calls: Optional[int] = 10  # maximum tool calls per session


class MCPConfig(BaseModel):
    """Global MCP configuration"""
    servers: Dict[str, MCPServerConfig] = {}
    agent: MCPModuleConfig = MCPModuleConfig()
