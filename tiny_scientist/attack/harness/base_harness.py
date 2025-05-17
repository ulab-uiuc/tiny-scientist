"""
Base harness class for prompt attacks.
"""
import dataclasses
from typing import Any, Dict, Optional

from ..intention.base_intention import Intention


@dataclasses.dataclass
class Harness:
    """Base class for all prompt attack harnesses."""
    
    name: str = ""
    site_url: str = ""
    application_document: str = ""
    
    def run_harness(self, prompt_injection: Intention) -> Dict[str, Any]:
        """
        Run the harness with the given prompt injection.
        
        Args:
            prompt_injection: The prompt injection intention to test
            
        Returns:
            Dictionary containing the results of the attack
        """
        raise NotImplementedError("Harness must implement run_harness method")
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate if the attack was successful based on the response.
        
        Args:
            response: The response from the LLM
            
        Returns:
            True if the attack was successful, False otherwise
        """
        raise NotImplementedError("Harness must implement validate_response method") 