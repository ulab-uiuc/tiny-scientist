import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .llm import get_response_from_llm
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.loader import load_paper


class Drawer:
    def __init__(self, model: Any, client: Any, config_dir: str, temperature: float = 0.75):
        """Initialize the Drawer with model configuration and prompt templates."""
        self.model = model
        self.client = client
        self.temperature = temperature
        
        # Load prompt templates
        with open(osp.join(config_dir, "diagram_prompt.yaml"), "r") as f:
            self.prompts = yaml.safe_load(f)
        
        # Process template instructions in diagram form
        if "template_instructions" in self.prompts and "few_shot_instructions" in self.prompts:
            self.prompts["few_shot_instructions"] = self.prompts["few_shot_instructions"].replace(
                "{{ template_instructions }}", self.prompts["template_instructions"]
            )
            
        # Set directory path
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def draw_diagram(
            self,
            text: str,
            example: Optional[str] = None,
            msg_history: Optional[List[Dict[str, Any]]] = None,
            return_msg_history: bool = False,
            drawer_system_prompt: Optional[str] = None,
    ) -> Any:
        """Generate a diagram for the given text with an optional few-shot example.
        
        Args:
            text: The text content of the paper
            example: Optional string of a serialized image to use as a few-shot example
            msg_history: Optional message history for conversation context
            return_msg_history: Whether to return the updated message history
            drawer_system_prompt: Optional custom system prompt
            
        Returns:
            Dict with diagram components or tuple with diagram and message history
        """
        # Use default system prompt if none provided
        if drawer_system_prompt is None:
            drawer_system_prompt = self.prompts.get("diagram_system_prompt_base")

        # Prepare prompt with the few-shot example
        base_prompt = self._prepare_diagram_prompt(text, example)

        # Generate diagram
        diagram, updated_msg_history = self._generate_diagram(
            base_prompt, drawer_system_prompt, msg_history
        )

        return (diagram, updated_msg_history) if return_msg_history else diagram

    def _prepare_diagram_prompt(self, text: str, example: Optional[str] = None) -> str:
        """Prepare the prompt with the few-shot example if provided."""
        if example:
            # Format the few-shot instructions with the example
            few_shot_prompt = self.prompts["few_shot_instructions"].format(example=example)
            # Combine the few-shot prompt with the paper text
            base_prompt = few_shot_prompt + f"\n\nHere is the paper you are asked to create a diagram for:\n```\n{text}\n```"
        else:
            # If no example is provided, use just the template instructions
            base_prompt = self.prompts["template_instructions"] + f"\n\nHere is the paper you are asked to create a diagram for:\n```\n{text}\n```"
        
        return str(base_prompt)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def _generate_diagram(
            self,
            base_prompt: str,
            drawer_system_prompt: str,
            msg_history: Optional[List[Dict[str, Any]]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate a diagram based on the paper content."""
        # Generate diagram
        llm_response, msg_history = get_response_from_llm(
            base_prompt,
            model=self.model,
            client=self.client,
            system_message=drawer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=self.temperature,
        )
        
        # Extract the diagram from the response
        diagram = self._extract_diagram(llm_response)
        
        return diagram, msg_history
    
    def _extract_diagram(self, response: str) -> Dict[str, Any]:
        """Extract the diagram SVG and summary from the LLM response."""
        result = {
            "summary": "",
            "svg": "",
            "full_response": response
        }
        
        # Extract the summary
        summary_start = response.find("SUMMARY:")
        if summary_start != -1:
            summary_end = response.find("DIAGRAM SVG:", summary_start)
            if summary_end != -1:
                result["summary"] = response[summary_start + 8:summary_end].strip()
        
        # Extract the SVG
        svg_start = response.find("```svg", summary_start if summary_start != -1 else 0)
        if svg_start == -1:
            # Try without language specifier
            svg_start = response.find("```", summary_start if summary_start != -1 else 0)
            if svg_start != -1:
                svg_start += 3  # Skip past ```
        else:
            svg_start += 6  # Skip past ```svg
        
        if svg_start != -1:
            svg_end = response.find("```", svg_start)
            if svg_end != -1:
                result["svg"] = response[svg_start:svg_end].strip()
        
        return result
