from typing import Dict

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
    non_experiment_plan_prompt: str


class DrawerPrompt(BaseModel):
    diagram_system_prompt: str
    section_prompt: Dict[str, str]
    error_list: str  # Remains global


class SafetyPrompt(BaseModel):
    risk_assessment_system_prompt: str
    attack_detection_system_prompt: str
    ethical_defense_system_prompt: str
    ethical_defense_prompt: str
