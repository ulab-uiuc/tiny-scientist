from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union


class ReviewerPrompt(BaseModel):
    reviewer_system_prompt_base: str
    reviewer_system_prompt_neg: str
    reviewer_system_prompt_pos: str
    query_prompt: str
    template_instructions: str
    neurips_form: str
    meta_reviewer_system_prompt: str
    reviewer_reflection_prompt: str


class SectionTips(BaseModel):
    Abstract: str
    Introduction: str
    Related_Work: str
    Method: str
    Experimental_Setup: str
    Results: str
    Discussion: str
    Conclusion: str

class Section_prompt(BaseModel):
    Introduction: str
    Method: str
    Experimental_Setup: str
    Results: str
    Discussion: str
    Conclusion: str


class WriterPrompt(BaseModel):
    write_system_prompt: str
    write_system_prompt_related_work: str
    section_tips: SectionTips
    error_list: str
    refinement_prompt: str
    second_refinement_prompt: str
    citation_system_prompt: str
    abstract_prompt: str
    section_prompt: Section_prompt
    citation_related_work_prompt: str
    add_citation_prompt: str
    embed_citation_prompt: str
    related_work_prompt: str
    title_refinement_prompt: str
    citation_aider_format: str

class CoderPrompt(BaseModel):
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
    novelty_system_prompt: str
    idea_first_prompt: str
    idea_reflection_prompt: str
    novelty_prompt: str
    experiment_plan_prompt: str
