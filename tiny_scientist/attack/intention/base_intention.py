"""
Base intention class for prompt attacks.
"""
import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class Intention:
    """Base class for all prompt attack intentions."""
    
    name: str = ""
    description: str = ""
    target_behavior: str = ""
    constraints: List[str] = dataclasses.field(default_factory=list)
    
    def get_attack_prompt(self) -> str:
        """
        Generate the attack prompt based on the intention.
        
        Returns:
            The generated attack prompt
        """
        raise NotImplementedError("Intention must implement get_attack_prompt method")
    
    def validate_attack(self, response: Dict[str, Any]) -> bool:
        """
        Validate if the attack was successful based on the response.
        
        Args:
            response: The response from the LLM
            
        Returns:
            True if the attack was successful, False otherwise
        """
        raise NotImplementedError("Intention must implement validate_attack method") 