import json
import random
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm


class MoleculeReactionTool(BaseTool):
    """Tool for simulating chemical reactions between molecules."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for simulations."""
        self.client, self.model = create_client(model)
        self.common_reactions = {
            "nacl+h2o": "NaCl dissolves in water to form Na+ and Cl- ions in solution. No chemical reaction occurs.",
            "hcl+naoh": "HCl + NaOH → NaCl + H2O (acid-base neutralization reaction)",
            "c6h12o6+o2": "C6H12O6 + 6O2 → 6CO2 + 6H2O + energy (glucose combustion/metabolism)",
            "h2+o2": "2H2 + O2 → 2H2O (hydrogen combustion)",
            "caco3+hcl": "CaCO3 + 2HCl → CaCl2 + H2O + CO2 (carbonate reaction with acid)",
            "c2h4+h2o": "C2H4 + H2O → C2H5OH (ethylene hydration to form ethanol)",
            "ch4+o2": "CH4 + 2O2 → CO2 + 2H2O (methane combustion)",
        }
        
    def safety_detect(self, reactants: str) -> Dict[str, Any]:
        """Check if the requested reaction is safe to simulate."""
        # Strip whitespace, convert to lowercase for normalization
        normalized_input = reactants.lower().strip()
        
        # Check for potentially dangerous keywords
        dangerous_keywords = [
            "explosive", "bomb", "weapon", "toxic", "hazardous", 
            "illegal", "poison", "uranium", "plutonium", "nerve agent",
            "sarin", "ricin", "mustard gas", "vx", "tabun", "novichok"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in normalized_input:
                return {
                    "safe": False,
                    "reason": f"Detected potentially dangerous keyword: '{keyword}'",
                    "allowed": False
                }
        
        # Use LLM to evaluate safety for more complex cases
        if len(normalized_input) > 10:  # Only consult LLM for non-trivial requests
            prompt = f"""
            Determine if this chemical reaction request is dangerous or involves 
            harmful/illegal substances or processes:
            
            "{reactants}"
            
            Respond with JSON:
            {{
                "safe": true/false,
                "reason": "Explanation of safety assessment",
                "concerns": ["list", "of", "specific", "concerns"] or []
            }}
            """
            
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message="You are a chemical safety expert. Evaluate the safety of chemical reactions.",
                temperature=0.3
            )
            
            try:
                # Extract JSON from response
                safety_data = json.loads(response)
                # Add allowed flag based on safety
                safety_data["allowed"] = safety_data.get("safe", True)
                return safety_data
            except Exception:
                # Fallback to simple check if LLM response parsing fails
                pass
        
        # Default to allowing if no issues detected
        return {"safe": True, "reason": "No obvious safety concerns detected", "allowed": True}
    
    def run(self, reactants: str, conditions: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate a chemical reaction between given reactants.
        
        Args:
            reactants: Description of reactants (e.g., "NaCl + H2O" or "sodium chloride and water")
            conditions: Optional reaction conditions like temperature, pressure, etc.
            
        Returns:
            Dictionary with reaction results
        """
        # Check safety first
        safety_result = self.safety_detect(reactants)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Strip whitespace, convert to lowercase for lookup
        normalized_input = "".join(reactants.lower().split())
        normalized_input = normalized_input.replace("+", "").replace(" ", "")
        
        # Check if it's in our common reactions dictionary
        for key, result in self.common_reactions.items():
            if key in normalized_input or normalized_input in key:
                return {
                    "success": True,
                    "reaction": result,
                    "reactants": reactants,
                    "conditions": conditions or "Standard conditions",
                    "simulated": True
                }
        
        # If not found in simple lookup, use LLM to simulate
        context = f"Reactants: {reactants}"
        if conditions:
            context += f"\nConditions: {conditions}"
            
        prompt = f"""
        Simulate the following chemical reaction and provide the products:
        
        {context}
        
        Provide the balanced chemical equation, describe the reaction type,
        and explain what happens at the molecular level.
        
        Format your response as JSON:
        {{
            "balanced_equation": "Balanced chemical equation with states",
            "reaction_type": "Type of reaction (e.g., combustion, redox, etc.)",
            "molecular_explanation": "Brief explanation of the molecular changes",
            "products": ["List", "of", "products"]
        }}
        """
        
        response, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message="You are a chemistry expert simulating chemical reactions accurately.",
            temperature=0.3
        )
        
        try:
            # Extract JSON from response
            result_data = json.loads(response)
            # Add metadata
            result_data["success"] = True
            result_data["reactants"] = reactants
            result_data["conditions"] = conditions or "Standard conditions"
            result_data["simulated"] = True
            return result_data
        except json.JSONDecodeError:
            # Fallback to returning the raw text if JSON parsing fails
            return {
                "success": True,
                "reaction_description": response,
                "reactants": reactants,
                "conditions": conditions or "Standard conditions",
                "simulated": True,
                "note": "Response formatting failed, returning raw description"
            }


class MoleculePropertyTool(BaseTool):
    """Tool for retrieving properties of chemical compounds."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for simulations."""
        self.client, self.model = create_client(model)
        # Common properties for quick lookup
        self.common_compounds = {
            "water": {
                "formula": "H2O",
                "molar_mass": 18.01528,
                "density": "1.0 g/cm³ at 4°C",
                "melting_point": "0°C (273.15 K)",
                "boiling_point": "100°C (373.15 K)",
                "state_at_stp": "liquid"
            },
            "carbon dioxide": {
                "formula": "CO2",
                "molar_mass": 44.01,
                "density": "1.98 g/L at STP",
                "melting_point": "-78.5°C (194.7 K) (sublimates)",
                "boiling_point": "-57°C (216.6 K) (sublimates)",
                "state_at_stp": "gas"
            },
            "sodium chloride": {
                "formula": "NaCl",
                "molar_mass": 58.44,
                "density": "2.16 g/cm³",
                "melting_point": "801°C (1074 K)",
                "boiling_point": "1413°C (1686 K)",
                "state_at_stp": "solid"
            },
            "glucose": {
                "formula": "C6H12O6",
                "molar_mass": 180.156,
                "density": "1.54 g/cm³",
                "melting_point": "146°C (419 K)",
                "boiling_point": "Decomposes",
                "state_at_stp": "solid"
            }
        }
    
    def safety_detect(self, compound: str) -> Dict[str, Any]:
        """Check if the requested compound is safe to query."""
        # Strip whitespace, convert to lowercase for normalization
        normalized_input = compound.lower().strip()
        
        # Check for potentially dangerous keywords
        dangerous_keywords = [
            "explosive", "bomb", "weapon", "toxic", "hazardous", 
            "illegal", "poison", "uranium", "plutonium", "nerve agent",
            "sarin", "ricin", "mustard gas", "vx", "tabun", "novichok"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in normalized_input:
                return {
                    "safe": False,
                    "reason": f"Detected potentially dangerous keyword: '{keyword}'",
                    "allowed": False
                }
        
        # Default to allowing if no issues detected
        return {"safe": True, "reason": "No obvious safety concerns detected", "allowed": True}
    
    def run(self, compound: str, properties: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve properties of a chemical compound.
        
        Args:
            compound: Name or formula of the compound
            properties: Optional list of specific properties to retrieve
            
        Returns:
            Dictionary with compound properties
        """
        # Check safety first
        safety_result = self.safety_detect(compound)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Normalize compound name for lookup
        normalized_compound = compound.lower().strip()
        
        # Check if compound is in our common compounds dictionary
        if normalized_compound in self.common_compounds:
            result = {"success": True, "compound": compound}
            compound_data = self.common_compounds[normalized_compound]
            
            # If specific properties requested, filter them
            if properties:
                for prop in properties:
                    prop_lower = prop.lower().replace(" ", "_")
                    if prop_lower in compound_data:
                        result[prop] = compound_data[prop_lower]
                    else:
                        result[prop] = "Property not available"
            else:
                # Include all properties
                result.update(compound_data)
            
            return result
        
        # If not in common compounds, use LLM to get properties
        props_list = ", ".join(properties) if properties else "common chemical properties"
        prompt = f"""
        Provide the following properties of {compound}: {props_list}
        
        Format your response as JSON with the property names as keys and values as accurate as possible.
        Include at least: formula, molar_mass, state_at_stp (if known)
        
        Example format:
        {{
            "formula": "H2O",
            "molar_mass": 18.01528,
            "state_at_stp": "liquid",
            ...other properties...
        }}
        """
        
        response, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message="You are a chemistry database providing accurate chemical properties.",
            temperature=0.3
        )
        
        try:
            # Extract JSON from response
            result_data = json.loads(response)
            # Add metadata
            result_data["success"] = True
            result_data["compound"] = compound
            return result_data
        except json.JSONDecodeError:
            # Fallback to returning the raw text if JSON parsing fails
            return {
                "success": False,
                "error": "Failed to parse properties",
                "raw_response": response,
                "compound": compound
            }


class SolubilityTool(BaseTool):
    """Tool for determining solubility of compounds in various solvents."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for simulations."""
        self.client, self.model = create_client(model)
    
    def safety_detect(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Check if the requested solubility query is safe."""
        compound = inputs.get("compound", "")
        solvent = inputs.get("solvent", "")
        
        combined = f"{compound} in {solvent}".lower()
        
        # Check for potentially dangerous keywords
        dangerous_keywords = [
            "explosive", "bomb", "weapon", "toxic", "hazardous", 
            "illegal", "poison", "nerve agent"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in combined:
                return {
                    "safe": False,
                    "reason": f"Detected potentially dangerous keyword: '{keyword}'",
                    "allowed": False
                }
        
        # Default to allowing if no issues detected
        return {"safe": True, "reason": "No obvious safety concerns detected", "allowed": True}
    
    def run(self, compound: str, solvent: str = "water", temperature: Optional[float] = 25.0) -> Dict[str, Any]:
        """
        Determine the solubility of a compound in a specified solvent.
        
        Args:
            compound: Name or formula of the compound
            solvent: Solvent to check solubility in (default: water)
            temperature: Temperature in Celsius (default: 25°C)
            
        Returns:
            Dictionary with solubility information
        """
        # Check safety first
        safety_result = self.safety_detect({"compound": compound, "solvent": solvent})
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Common solubility rules for quick lookup
        common_cases = {
            "nacl_water": "Highly soluble (359 g/L at 20°C)",
            "nacl_ethanol": "Slightly soluble (0.65 g/L at 20°C)",
            "cacl2_water": "Highly soluble (745 g/L at 20°C)",
            "naoh_water": "Highly soluble (1090 g/L at 20°C)",
            "i2_water": "Poorly soluble (0.33 g/L at 20°C)",
            "i2_ethanol": "Soluble (200 g/L at 20°C)",
            "oil_water": "Insoluble",
            "sugar_water": "Highly soluble (2000 g/L at 20°C)",
            "benzene_water": "Poorly soluble (1.8 g/L at 25°C)",
            "benzene_ethanol": "Highly soluble"
        }
        
        # Normalize inputs for lookup
        lookup_key = f"{compound.lower().replace(' ', '')}_{solvent.lower().replace(' ', '')}"
        
        if lookup_key in common_cases:
            return {
                "success": True,
                "compound": compound,
                "solvent": solvent,
                "temperature": f"{temperature}°C",
                "solubility": common_cases[lookup_key],
                "note": "Data from common solubility database"
            }
        
        # If not in common cases, simulate using LLM
        prompt = f"""
        Determine the solubility of {compound} in {solvent} at {temperature}°C.
        
        Format your response as JSON:
        {{
            "solubility_qualitative": "Descriptive solubility (e.g., 'highly soluble', 'insoluble')",
            "solubility_quantitative": "Numerical value with units if known, or 'Not available'",
            "factors": "Brief explanation of factors affecting this solubility",
            "common_uses": "Applications that make use of this solubility property (if relevant)"
        }}
        """
        
        response, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message="You are a chemistry expert providing accurate solubility data.",
            temperature=0.2
        )
        
        try:
            # Extract JSON from response
            result_data = json.loads(response)
            # Add metadata
            result_data["success"] = True
            result_data["compound"] = compound
            result_data["solvent"] = solvent
            result_data["temperature"] = f"{temperature}°C"
            return result_data
        except json.JSONDecodeError:
            # Fallback to returning the raw text if JSON parsing fails
            return {
                "success": True,
                "compound": compound,
                "solvent": solvent,
                "temperature": f"{temperature}°C",
                "solubility_description": response,
                "note": "Response formatting failed, returning raw description"
            } 