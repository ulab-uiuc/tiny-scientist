import json
import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysicalToolUtility(BaseTool):
    """Base class for all physical tools with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the physical tool utility.
        
        Args:
            model: LLM model to use for safety assessments
        """
        self.client, self.model = create_client(model)
        
        # Load physical tools data
        self.tools_data = self._load_tools_data()
        self.tools_by_name = {tool['tool_name']: tool for tool in self.tools_data}
        
    def _load_tools_data(self) -> List[Dict[str, Any]]:
        """Load physical tools data from JSON file."""
        try:
            # Try to locate the JSON file - first attempt relative path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            json_path = os.path.join(base_dir, "data/ScienceSafetyData/Tool/phy_tool.json")
            
            if not os.path.exists(json_path):
                # Alternative paths
                alternate_paths = [
                    "data/ScienceSafetyData/Tool/phy_tool.json",
                    "./data/ScienceSafetyData/Tool/phy_tool.json"
                ]
                
                for path in alternate_paths:
                    if os.path.exists(path):
                        json_path = path
                        break
            
            with open(json_path, 'r') as f:
                tools_data = json.load(f)
                logger.info(f"Successfully loaded physics tool data: {len(tools_data)} tools")
                return tools_data
                
        except Exception as e:
            logger.error(f"Failed to load physics tool data: {e}")
            # Return empty list as fallback
            return []
    
    def safety_detect(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic safety checking method for physical tools.
        
        Args:
            tool_name: Name of the physical tool
            parameters: Dictionary of tool parameter values
            
        Returns:
            Dictionary with safety assessment including 'allowed' field
        """
        # Return error if tool doesn't exist
        if tool_name not in self.tools_by_name:
            return {
                "allowed": False,
                "reason": f"Unknown tool: '{tool_name}'. Available tools: {list(self.tools_by_name.keys())}"
            }
        
        tool_data = self.tools_by_name[tool_name]
        
        # Check if safety thresholds exist
        if "risks_detection" not in tool_data or "threshold_conditions" not in tool_data["risks_detection"]:
            return {
                "allowed": True,
                "reason": "This tool has no defined safety threshold conditions."
            }
        
        # Check each threshold condition
        threshold_conditions = tool_data["risks_detection"]["threshold_conditions"]
        violations = []
        
        for condition in threshold_conditions:
            param = condition["parameter"]
            
            # Skip parameters not provided
            if param not in parameters:
                continue
                
            param_value = parameters[param]
            threshold = condition["threshold"]
            
            # Check threshold conditions
            if isinstance(threshold, dict):  # Range condition
                lower_bound = threshold.get("lower_bound")
                upper_bound = threshold.get("upper_bound")
                
                if lower_bound is not None and param_value < lower_bound:
                    violations.append({
                        "parameter": param,
                        "value": param_value,
                        "threshold": f"not less than {lower_bound}",
                        "condition": condition["condition"]
                    })
                elif upper_bound is not None and param_value > upper_bound:
                    violations.append({
                        "parameter": param,
                        "value": param_value,
                        "threshold": f"not greater than {upper_bound}",
                        "condition": condition["condition"]
                    })
            else:  # Single threshold condition
                # Most conditions involve not exceeding the threshold
                if param_value >= threshold:
                    violations.append({
                        "parameter": param,
                        "value": param_value,
                        "threshold": threshold,
                        "condition": condition["condition"]
                    })
        
        # Return not allowed if violations exist
        if violations:
            return {
                "allowed": False,
                "reason": "Parameters violate safety thresholds",
                "violations": violations
            }
        
        # Use LLM for more advanced safety assessment
        safety_result = self._assess_safety_with_llm(tool_name, parameters)
        
        return safety_result
    
    def _assess_safety_with_llm(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to assess safety for more complex scenarios."""
        tool_data = self.tools_by_name[tool_name]
        
        # Build prompt
        tool_description = "\n".join(tool_data["tool_description"])
        
        params_text = "\n".join([f"{k}: {v}" for k, v in parameters.items()])
        
        risk_description = ""
        if "risks_detection" in tool_data and "description" in tool_data["risks_detection"]:
            risk_description = tool_data["risks_detection"]["description"]
        
        prompt = f"""
        Evaluate the safety of the following physical tool parameters:
        
        Tool Name: {tool_name}
        Tool Description: {tool_description}
        
        Risk Description: {risk_description}
        
        Operating Parameters:
        {params_text}
        
        Please analyze whether these parameters are safe, considering parameter interactions and potential risks.
        Respond in JSON format:
        {{
            "allowed": true/false,
            "reason": "Detailed explanation of your safety assessment",
            "concerns": ["specific concern 1", "specific concern 2"] or []
        }}
        """
        
        try:
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message="You are a physics safety expert responsible for evaluating the safety of physics instrument parameters.",
                temperature=0.3
            )
            
            # Try to parse JSON response
            safety_data = json.loads(response)
            return safety_data
            
        except Exception as e:
            logger.error(f"LLM safety assessment failed: {e}")
            # Default to safe if LLM assessment fails
            return {"allowed": True, "reason": "Basic parameter checks passed, but advanced safety assessment failed.", "concerns": ["LLM assessment failed"]}


class HighPowerPulsedLaser(PhysicalToolUtility):
    """High-power pulsed laser tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the high-power pulsed laser tool."""
        super().__init__(model)
        # Get tool-specific data
        self.tool_name = "High-power pulsed laser (Nd:YAG, Ti:Sapphire)"
        self.tool_data = next((tool for tool in self.tools_data if tool.get("tool_name") == self.tool_name), {})
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """Simplified run method that only performs safety checks."""
        safety_result = self.safety_detect(self.tool_name, kwargs)
        
        return {
            "tool_name": self.tool_name,
            "parameters": kwargs,
            "safety": safety_result,
            "allowed": safety_result.get("allowed", False)
        }
    
    def safety_detect(self, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safety check specifically for high-power pulsed lasers."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Additional safety checks specific to lasers
        # Check combined parameters that might create unsafe conditions
        if "pulse_energy" in parameters and "repetition_rate" in parameters:
            avg_power = parameters["pulse_energy"] * parameters["repetition_rate"] / 1000  # in watts
            if avg_power > 500:  # Dangerously high average power
                return {
                    "allowed": False,
                    "reason": "Dangerous average power level detected",
                    "violations": [{
                        "parameter": "average_power",
                        "value": f"{avg_power} W",
                        "threshold": "500 W",
                        "condition": "Combined pulse energy and repetition rate create excessive average power"
                    }]
                }
                
        # Check simmer and high voltage combination (both must be active)
        if "simmer_status" in parameters and "high_voltage_ok" in parameters:
            if parameters["simmer_status"] == 0 and parameters["high_voltage_ok"] == 1:
                return {
                    "allowed": False,
                    "reason": "High voltage active but simmer disabled creates flashlamp risk",
                    "violations": [{
                        "parameter": "simmer_status",
                        "value": 0,
                        "threshold": 1,
                        "condition": "Simmer must be active when high voltage is enabled"
                    }]
                }
        
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class RadiationDetectionSystem(PhysicalToolUtility):
    """Radiation detection system tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the radiation detection system tool."""
        super().__init__(model)
        # Get tool-specific data
        self.tool_name = "Radiation detection system (HPGe, Scintillator, μRWell)"
        self.tool_data = next((tool for tool in self.tools_data if tool.get("tool_name") == self.tool_name), {})
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """Simplified run method that only performs safety checks."""
        safety_result = self.safety_detect(self.tool_name, kwargs)
        
        return {
            "tool_name": self.tool_name,
            "parameters": kwargs,
            "safety": safety_result,
            "allowed": safety_result.get("allowed", False)
        }
    
    def safety_detect(self, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safety check specifically for radiation detection systems."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Special check for detector temperature and bias voltage combination
        if "detector_temperature" in parameters and "bias_voltage" in parameters:
            if parameters["detector_temperature"] > 90 and parameters["bias_voltage"] > 1000:
                return {
                    "allowed": False,
                    "reason": "Applying high voltage to warming detector creates risk of damage",
                    "violations": [{
                        "parameter": "detector_temperature + bias_voltage",
                        "value": f"{parameters['detector_temperature']}K at {parameters['bias_voltage']}V",
                        "threshold": "not safe in combination",
                        "condition": "Detector temperature approaching 100K with bias voltage applied risks damage"
                    }]
                }
                
        # Check for condensation risk
        if "ambient_temperature" in parameters and "ambient_humidity" in parameters:
            # Calculate dew point (simplified)
            temp = parameters["ambient_temperature"]
            humidity = parameters["ambient_humidity"]
            dew_point = temp - ((100 - humidity) / 5)
            
            # If detector temp is below dew point, there's condensation risk
            if "detector_temperature" in parameters and parameters["detector_temperature"] < dew_point:
                return {
                    "allowed": False,
                    "reason": "Condensation risk detected",
                    "violations": [{
                        "parameter": "ambient conditions",
                        "value": f"Dew point ~{dew_point}°C with detector at {parameters['detector_temperature']}K",
                        "threshold": "detector must be above dew point",
                        "condition": "Condensation on cold surfaces will damage HV components"
                    }]
                }
        
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class VacuumSystem(PhysicalToolUtility):
    """Edwards nEXT300D/T vacuum system tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the vacuum system tool."""
        super().__init__(model)
        # Get tool-specific data
        self.tool_name = "Edwards nEXT300D/T"
        self.tool_data = next((tool for tool in self.tools_data if tool.get("tool_name") == self.tool_name), {})
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """Simplified run method that only performs safety checks."""
        safety_result = self.safety_detect(self.tool_name, kwargs)
        
        return {
            "tool_name": self.tool_name,
            "parameters": kwargs,
            "safety": safety_result,
            "allowed": safety_result.get("allowed", False)
        }
    
    def safety_detect(self, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safety check specifically for vacuum systems."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Check chamber material compatibility with the pressure
        if "chamber_material" in parameters and "pressure_setpoint" in parameters:
            if parameters["chamber_material"].lower() in ["glass", "plastic", "acrylic"] and parameters["pressure_setpoint"] < 1:
                return {
                    "allowed": False,
                    "reason": "Unsafe chamber material for high vacuum",
                    "violations": [{
                        "parameter": "chamber_material",
                        "value": parameters["chamber_material"],
                        "threshold": "must be metal for high vacuum",
                        "condition": "Glass or plastic chambers can implode at high vacuum levels"
                    }]
                }
        
        # Check for dangerous magnetic fields with metallic samples
        if "external_magnetic_field" in parameters and parameters["external_magnetic_field"] > 2:
            return {
                "allowed": False,
                "reason": "External magnetic field exceeds safe limits for pump operation",
                "violations": [{
                    "parameter": "external_magnetic_field",
                    "value": parameters["external_magnetic_field"],
                    "threshold": "2 mT",
                    "condition": "Strong external magnetic fields can cause rotor instability"
                }]
            }
        
        # Check cooling requirements based on pump speed
        if "pump_speed" in parameters and "cooling_water_flow_rate" in parameters:
            required_flow = parameters["pump_speed"] / 1000  # Simple calculation - higher speeds need more cooling
            if parameters["cooling_water_flow_rate"] < required_flow:
                return {
                    "allowed": False,
                    "reason": "Insufficient cooling for pump speed",
                    "violations": [{
                        "parameter": "cooling_water_flow_rate",
                        "value": parameters["cooling_water_flow_rate"],
                        "threshold": f"≥ {required_flow} L/h",
                        "condition": "Higher pump speeds require more cooling"
                    }]
                }
        
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class NeutronEmitter(PhysicalToolUtility):
    """Neutron emitter tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the neutron emitter tool."""
        super().__init__(model)
        # Get tool-specific data
        self.tool_name = "Neutron emitter (Am-Be, DD/DT source)"
        self.tool_data = next((tool for tool in self.tools_data if tool.get("tool_name") == self.tool_name), {})
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """Simplified run method that only performs safety checks."""
        safety_result = self.safety_detect(self.tool_name, kwargs)
        
        return {
            "tool_name": self.tool_name,
            "parameters": kwargs,
            "safety": safety_result,
            "allowed": safety_result.get("allowed", False)
        }
    
    def safety_detect(self, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safety check specifically for neutron emitters."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Calculate approximate radiation exposure from neutron yield
        if "neutron_yield" in parameters:
            if parameters["neutron_yield"] > 1e7:  # Very high yield
                if "duty_factor" in parameters and parameters["duty_factor"] > 0.3:
                    return {
                        "allowed": False,
                        "reason": "Dangerous combination of high neutron yield and duty factor",
                        "violations": [{
                            "parameter": "neutron_yield + duty_factor",
                            "value": f"{parameters['neutron_yield']} n/s at {parameters['duty_factor']} duty",
                            "threshold": "not safe in combination",
                            "condition": "High neutron yield with high duty factor creates dangerous radiation levels"
                        }]
                    }
        
        # Check temperature limits for beam current
        if "beam_current" in parameters and "ambient_temperature" in parameters:
            if parameters["beam_current"] > 40 and parameters["ambient_temperature"] > 35:
                return {
                    "allowed": False,
                    "reason": "High beam current in high ambient temperature risks overheating",
                    "violations": [{
                        "parameter": "beam_current + ambient_temperature",
                        "value": f"{parameters['beam_current']}µA at {parameters['ambient_temperature']}°C",
                        "threshold": "not safe in combination",
                        "condition": "Cooling insufficient for high beam currents at elevated temperatures"
                    }]
                }
        
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class ParticleAccelerator(PhysicalToolUtility):
    """Linear/Compact Particle Accelerator tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the particle accelerator tool."""
        super().__init__(model)
        # Get tool-specific data
        self.tool_name = "Linear/Compact Particle Accelerator"
        self.tool_data = next((tool for tool in self.tools_data if tool.get("tool_name") == self.tool_name), {})
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """Simplified run method that only performs safety checks."""
        safety_result = self.safety_detect(self.tool_name, kwargs)
        
        return {
            "tool_name": self.tool_name,
            "parameters": kwargs,
            "safety": safety_result,
            "allowed": safety_result.get("allowed", False)
        }
    
    def safety_detect(self, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safety check specifically for particle accelerators."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Check beam power based on energy and current
        if "beam_energy" in parameters and "beam_current" in parameters:
            beam_power = parameters["beam_energy"] * parameters["beam_current"] / 1000  # Approximate power in kW
            
            if beam_power > 200:
                return {
                    "allowed": False,
                    "reason": "Beam power exceeds safe operational limit",
                    "violations": [{
                        "parameter": "beam_power",
                        "value": f"{beam_power} kW",
                        "threshold": "200 kW",
                        "condition": "Combined beam energy and current create excessive power deposition"
                    }]
                }
        
        # Check magnetic field vs. cavity gradient interaction
        if "ambient_magnetic_field" in parameters and "cavity_gradient" in parameters:
            if parameters["ambient_magnetic_field"] > 0.2 and parameters["cavity_gradient"] > 30:
                return {
                    "allowed": False,
                    "reason": "Ambient magnetic field too high for desired cavity gradient",
                    "violations": [{
                        "parameter": "ambient_magnetic_field + cavity_gradient",
                        "value": f"{parameters['ambient_magnetic_field']}µT at {parameters['cavity_gradient']}MV/m",
                        "threshold": "not safe in combination",
                        "condition": "High magnetic fields compromise superconducting cavities at high gradients"
                    }]
                }
        
        # Check helium pressure vs. shield temperature
        if "cavity_helium_pressure" in parameters and "shield_temperature_upper" in parameters:
            if parameters["cavity_helium_pressure"] > 1.5 and parameters["shield_temperature_upper"] > 60:
                return {
                    "allowed": False,
                    "reason": "High helium pressure with elevated shield temperature creates risk",
                    "violations": [{
                        "parameter": "cavity_helium_pressure + shield_temperature_upper",
                        "value": f"{parameters['cavity_helium_pressure']}bar with shield at {parameters['shield_temperature_upper']}K",
                        "threshold": "not safe in combination",
                        "condition": "Thermal load with high pressure risks cryogenic system failure"
                    }]
                }
        
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters) 