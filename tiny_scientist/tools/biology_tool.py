import json
import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm, extract_json_between_markers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiologyToolUtility(BaseTool):
    """Base class for all biology tools with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the biology tool utility.
        
        Args:
            model: LLM model to use for safety assessments
        """
        self.client, self.model = create_client(model)
        self.tool_data = {
            "tool_description": [],
            "risks_detection": {
                "description": "",
                "threshold_conditions": []
            }
        }
    
    def safety_detect(self, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generic safety checking method for biology tools.
        
        Args:
            tool_name: Name of the biology tool (optional, uses self.tool_name if not provided)
            parameters: Dictionary of tool parameter values
            
        Returns:
            Dictionary with safety assessment including 'allowed' field
        """
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Use provided tool_name or self.tool_name
        tool_name = tool_name or self.tool_name
        
        # Extract base tool name
        if "Centrifuge" in tool_name:
            base_tool_name = "Centrifuge"
        elif "PCR System" in tool_name:
            base_tool_name = "PCR System"
        elif "Spectrophotometer" in tool_name:
            base_tool_name = "Spectrophotometer"
        elif "Gel Electrophoresis" in tool_name:
            base_tool_name = "Gel Electrophoresis"
        elif "CO2 Incubator" in tool_name:
            base_tool_name = "CO2 Incubator"
        else:
            return {
                "allowed": False,
                "reason": f"Unknown tool: '{tool_name}'. Available tools: Centrifuge, PCR System, Spectrophotometer, Gel Electrophoresis, CO2 Incubator"
            }
        
        # Step 1: Rule-based safety checks
        safety_check_method = getattr(self, f"_check_{base_tool_name.lower().replace(' ', '_')}_safety", None)
        
        if safety_check_method is None:
            return {
                "allowed": False,
                "reason": f"Unknown tool: '{tool_name}'. Available tools: Centrifuge, PCR System, Spectrophotometer, Gel Electrophoresis, CO2 Incubator"
            }
        
        # Perform rule-based safety checks
        rule_based_result = safety_check_method(parameters)
        
        # If rule-based check fails, return immediately
        if not rule_based_result.get("allowed", True):
            return rule_based_result
        
        # Step 2: LLM-based safety assessment
        risk_description = self.tool_data["risks_detection"].get("description", "")
        safety_result = self._assess_safety_with_llm(tool_name, parameters, risk_description)
        
        # If either check fails, mark as unsafe
        if not safety_result.get("allowed", True):
            return safety_result
        
        return {
            "allowed": True,
            "reason": "All safety checks passed",
            "concerns": []
        }
    
    def _assess_safety_with_llm(self, tool_name: str, parameters: Dict[str, Any], risk_description: str) -> Dict[str, Any]:
        """
        Assess safety using LLM.
        
        Args:
            tool_name: Name of the tool
            parameters: Dictionary of parameter values
            risk_description: Description of potential risks
            
        Returns:
            Dictionary with safety assessment
        """
        prompt = f"""
        You are a safety expert evaluating the safety of a {tool_name} operation.
        
        Tool Parameters:
        {json.dumps(parameters, indent=2)}
        
        Risk Description:
        {risk_description}
        
        Please evaluate if these parameters are safe for operation. Consider:
        1. Parameter combinations that might create unsafe conditions
        2. Environmental factors that could affect safety
        3. Potential failure modes or edge cases
        
        Respond with a JSON object containing:
        {{
            "allowed": true/false,
            "reason": "Detailed explanation of safety assessment",
            "concerns": ["List of specific safety concerns if any"]
        }}
        """
        
        try:
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model="gpt-4o",
                system_message="You are a safety expert evaluating scientific equipment parameters. Provide detailed safety assessments.",
                temperature=0.0
            )
            
            # Extract JSON from response
            safety_result = extract_json_between_markers(response)
            if safety_result is None:
                logger.error(f"Failed to extract safety assessment JSON for {tool_name}")
                return {
                    "allowed": False,
                    "reason": "Failed to perform LLM safety assessment",
                    "concerns": ["Could not parse safety assessment response"]
                }
                
            return safety_result
            
        except Exception as e:
            logger.error(f"Error in LLM safety assessment for {tool_name}: {e}")
            return {
                "allowed": False,
                "reason": f"Error in safety assessment: {str(e)}",
                "concerns": ["Failed to complete safety assessment"]
            }

class Centrifuge(BiologyToolUtility):
    """Centrifuge tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the centrifuge tool."""
        super().__init__(model)
        self.tool_name = "Centrifuge 5424 R"
        self.tool_data["input_parameters"] = {
            "speed_rpm": {
                "type": "float",
                "description": "Rotational speed in revolutions per minute (rpm). Adjustable from 100 rpm to 15,000 rpm in 50 rpm increments."
            },
            "speed_rcf": {
                "type": "float",
                "description": "Relative Centrifugal Force (g-force). Adjustable from 1 x g to 21,130 x g in 50 x g increments."
            },
            "time": {
                "type": "string",
                "description": "Duration of the centrifugation run. Options: 30 s to 9:59 h (up to 10 min in 0.5 min increments, then 1 min increments), or continuous ('oo')."
            },
            "temperature": {
                "type": "float",
                "description": "Temperature inside the rotor chamber in degrees Celsius (°C). Adjustable from -10°C to +40°C."
            },
            "sample_load_balance": {
                "type": "string",
                "description": "Qualitative assessment of whether tubes are loaded symmetrically with identical tubes (weight, material/density, and volume)."
            },
            "sample_density": {
                "type": "float",
                "description": "Density of the material being centrifuged in g/mL."
            },
            "max_load_per_rotor_bore": {
                "type": "float",
                "description": "Maximum load per rotor bore including adapter, tube, and contents in grams (g)."
            }
        }
        self.tool_data["tool_description"] = ["A refrigerated microcentrifuge used for the separation of aqueous solutions and suspensions of different densities in approved sample tubes. It has a capacity of 24 x 1.5/2.0 mL tubes and can achieve a maximum RCF of 21,130 x g with specific rotors."]
        self.tool_data["risks_detection"]["description"] = "Potential risks for the Eppendorf Centrifuge 5424 R include mechanical damage to the centrifuge or rotor, sample loss or damage, personal injury from rotor failure or improper handling, and biological/chemical hazards from aerosols if containment fails. These risks are often triggered when input parameters exceed their safe operating limits specified in the manual. For instance, exceeding maximum RCF or RPM, using samples with too high density, exceeding the maximum load per rotor bore, or improper sample load balance can lead to rotor failure, severe vibrations, sample loss, or equipment damage. The manual specifically warns that asymmetrical loading—or using non‑identical tubes in weight, material, or volume—will cause excessive vibration, poor separation, and may terminate the run. Furthermore, significant temperature deviations from the setpoint can halt operation and indicate system issues."
        
    def _check_centrifuge_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Centrifuge."""
        violations = []
        
        # Check speed_rcf
        if "speed_rcf" in parameters and parameters["speed_rcf"] > 21130:
            violations.append({
                "parameter": "speed_rcf",
                "value": parameters["speed_rcf"],
                "threshold": 21130,
                "condition": "Exceeding the maximum RCF for the selected 'rotor_selection' can lead to rotor failure and severe damage or injury."
            })
        
        # Check speed_rpm
        if "speed_rpm" in parameters and parameters["speed_rpm"] > 15000:
            violations.append({
                "parameter": "speed_rpm",
                "value": parameters["speed_rpm"],
                "threshold": 15000,
                "condition": "Exceeding the maximum 15,000 rpm (which is the max for all listed compatible rotors) can lead to rotor failure and severe damage or injury."
            })
        
        # Check sample_density
        if "sample_density" in parameters and parameters["sample_density"] > 1.2:
            violations.append({
                "parameter": "sample_density",
                "value": parameters["sample_density"],
                "threshold": 1.2,
                "condition": "Using samples with a 'sample_density' exceeding 1.2 g/mL when operating at maximum 'speed_rpm'/'speed_rcf' and maximum load for the selected rotor can overload the rotor and lead to failure."
            })
     
        
        # Check temperature deviation
        if "temperature" in parameters and "temperature_setpoint" in parameters:
            deviation = abs(parameters["temperature"] - parameters["temperature_setpoint"])
            if deviation > 5:
                violations.append({
                    "parameter": "temperature",
                    "value": parameters["temperature"],
                    "threshold": f"deviation > 5°C from {parameters['temperature_setpoint']}°C",
                    "condition": "If the actual chamber 'temperature' deviates by more than 5°C from the set 'temperature', 'Error 18' is displayed, and centrifugation stops automatically, indicating a cooling system issue or overload."
                })
        
        # Check max_load_per_rotor_bore
        if "max_load_per_rotor_bore" in parameters and parameters["max_load_per_rotor_bore"] > 3.75:
            violations.append({
                "parameter": "max_load_per_rotor_bore",
                "value": parameters["max_load_per_rotor_bore"],
                "threshold": 3.75,
                "condition": "Exceeding the 'max_load_per_rotor_bore' for the specific 'rotor_selection' can lead to tube/container failure, sample leakage, rotor imbalance, or rotor damage."
            })
        
        if violations:
            return {
                "allowed": False,
                "reason": "Parameters violate safety thresholds",
                "violations": violations
            }
        
        return {
            "allowed": True,
            "reason": "All safety checks passed",
            "concerns": []
        }
        
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
        """Safety check specifically for the Centrifuge."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class PCRSystem(BiologyToolUtility):
    """PCR System tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the PCR system tool."""
        super().__init__(model)
        self.tool_name = "Applied Biosystems ProFlex PCR System"
        self.tool_data["input_parameters"] = {
            "sample_block_type": {
                "type": "string",
                "description": "The specific ProFlex Sample Block installed (e.g., ProFlex 96-Well, ProFlex 3x32-Well, ProFlex Dual 96-Well, ProFlex Dual Flat, ProFlex Dual 384-Well). This choice dictates PCR volume ranges, ramp rates, and VeriFlex capabilities."
            },
            "reaction_volume": {
                "type": "float",
                "description": "The volume of the PCR reaction mixture in microliters (µL) or nanoliters (nL) depending on the block."
            },
            "heated_cover_temperature": {
                "type": "float",
                "description": "The temperature of the heated cover in degrees Celsius (°C). Can be set for idling."
            },
            "thermal_cycling_protocol_temperatures": {
                "type": "object",
                "description": "Temperatures for each step in the PCR protocol (e.g., denaturation, annealing, extension) in degrees Celsius (°C)."
            },
            "thermal_cycling_protocol_hold_times": {
                "type": "object",
                "description": "Hold times for each temperature step in the PCR protocol (e.g., seconds, minutes)."
            },
            "thermal_cycling_protocol_number_of_cycles": {
                "type": "number",
                "description": "The number of PCR cycles to be performed."
            },
            "ramp_rate_setting": {
                "type": "string_or_number",
                "description": "The rate of temperature change between steps, either as a percentage of maximum or a specific °C/sec, or determined by a simulation mode. This is block-dependent."
            },
            "veriflex_block_temperatures": {
                "type": "object",
                "description": "For blocks with VeriFlex™ technology, the specific temperatures set for the independent temperature zones within a step."
            }
        }
        self.tool_data["tool_description"] = ["An end-point thermal cycler designed for the amplification of nucleic acids using the Polymerase Chain Reaction (PCR) process. It features interchangeable sample blocks (e.g., 96-well, 3x32-well, Dual 96-well, Dual 384-well, Dual Flat) and a touchscreen interface."]
        self.tool_data["risks_detection"]["description"] = "Potential risks for the ProFlex PCR System include damage to samples due to incorrect thermal parameters, damage to the instrument from overheating or improper block handling, and electrical hazards. These risks can arise if input parameters are set outside safe or optimal ranges, or if environmental conditions are not met. For example, using reaction volumes outside the specified range or setting thermal cycling temperatures beyond the block's operational limits can damage samples or the instrument. Setting ramp rates too high might strain the system, excessive temperature differences in VeriFlex zones can lead to inaccurate control, and operating with an ambient temperature outside the recommended 15-30 °C range can also cause instability and damage."
        
    def _check_pcr_system_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for PCR System."""
        violations = []
        
        # Check reaction volume
        if "reaction_volume" in parameters:
            volume = parameters["reaction_volume"]
            if volume < 10 or volume > 80:
                violations.append({
                    "parameter": "reaction_volume",
                    "value": volume,
                    "threshold": {"lower_bound": 10, "upper_bound": 80},
                    "condition": "Using a 'reaction_volume' outside the specified range for the selected 'sample_block_type' (e.g., 10-80 µL for ProFlex 96-Well) can lead to inefficient thermal transfer, sample evaporation, or spillage, affecting PCR results and potentially contaminating the block."
                })
        
        # Check heated cover temperature
        if "heated_cover_temperature" in parameters:
            temp = parameters["heated_cover_temperature"]
            if temp < 30 or temp > 110:
                violations.append({
                    "parameter": "heated_cover_temperature",
                    "value": temp,
                    "threshold": {"lower_bound": 30, "upper_bound": 110},
                    "condition": "Setting the 'heated_cover_temperature' (idling) outside the 30-110°C range is not recommended. During operation, the cover can reach 110°C, posing a burn risk if touched before cooling."
                })
        
        # Check thermal cycling protocol temperatures
        if "thermal_cycling_protocol_temperatures" in parameters:
            temps = parameters["thermal_cycling_protocol_temperatures"]
            for step, temp in temps.items():
                if temp < 0.0 or temp > 100.0:
                    violations.append({
                        "parameter": f"thermal_cycling_protocol_temperatures.{step}",
                        "value": temp,
                        "threshold": {"lower_bound": 0.0, "upper_bound": 100.0},
                        "condition": "Setting 'thermal_cycling_protocol_temperatures' for any PCR step outside the instrument's operational block temperature range of 0°C to 100.0°C could damage the instrument or samples."
                    })
        
        # Check ramp rate setting
        if "ramp_rate_setting" in parameters:
            rate = float(parameters["ramp_rate_setting"])
            if rate > 6.0:
                violations.append({
                    "parameter": "ramp_rate_setting",
                    "value": rate,
                    "threshold": 6.0,
                    "condition": "Setting a 'ramp_rate_setting' that attempts to exceed the maximum block ramp rate for the selected 'sample_block_type' (e.g., >6.0 °C/sec for 3x32-Well block) may not be achievable and could strain the Peltier system."
                })
        
        # Check VeriFlex block temperatures
        if "veriflex_block_temperatures" in parameters:
            temps = parameters["veriflex_block_temperatures"]
            if isinstance(temps, dict):
                values = list(temps.values())
                if max(values) - min(values) > 5:
                    violations.append({
                        "parameter": "veriflex_block_temperatures",
                        "value": f"max difference: {max(values) - min(values)}°C",
                        "threshold": 5,
                        "condition": "Setting 'veriflex_block_temperatures' with a zone-to-zone difference exceeding the specified limit for the 'sample_block_type' (e.g., >5 °C for 3x32-Well block section) might lead to inaccurate temperature control or stress on the block."
                    })
        
        # Check ambient operating temperature
        if "ambient_operating_temperature" in parameters:
            temp = parameters["ambient_operating_temperature"]
            if temp < 15 or temp > 30:
                violations.append({
                    "parameter": "ambient_operating_temperature",
                    "value": temp,
                    "threshold": {"lower_bound": 15, "upper_bound": 30},
                    "condition": "Operating the instrument with an 'ambient_operating_temperature' outside the 15°C to 30°C range can cause system instability, affect temperature accuracy, and potentially damage the instrument."
                })
        
        if violations:
            return {
                "allowed": False,
                "reason": "Parameters violate safety thresholds",
                "violations": violations
            }
        
        return {
            "allowed": True,
            "reason": "All safety checks passed",
            "concerns": []
        }
        
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
        """Safety check specifically for the PCR System."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class Spectrophotometer(BiologyToolUtility):
    """Spectrophotometer tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the spectrophotometer tool."""
        super().__init__(model)
        self.tool_name = "Thermo Scientific NanoDrop 2000/2000c Spectrophotometer"
        self.tool_data["input_parameters"] = {
            "sample_volume_pedestal": {
                "type": "float",
                "description": "The volume of the sample pipetted onto the pedestal in microliters (µL)."
            },
            "sample_volume_cuvette": {
                "type": "float",
                "description": "The volume of the sample in the cuvette in milliliters (mL) or microliters (µL), ensuring it covers the 8.5 mm beam height."
            },
            "pathlength_pedestal": {
                "type": "string",
                "description": "Pathlength for pedestal measurements, typically 1 mm, auto-ranging to 0.05 mm. Can be fixed at 1mm in UV-Vis mode."
            },
            "pathlength_cuvette": {
                "type": "float",
                "description": "Pathlength for cuvette measurements, user-selectable (e.g., 10, 5, 2, 1 mm)."
            },
            "blank_solution_pedestal": {
                "type": "string",
                "description": "The buffer or solvent used to establish a blank reference for pedestal measurements."
            },
            "blank_solution_cuvette": {
                "type": "string",
                "description": "The buffer or solvent used to establish a blank reference for cuvette measurements."
            },
            "sample_type_application_setting": {
                "type": "string",
                "description": "The selected application module and specific sample type within that module (e.g., Nucleic Acid > DNA-50, Protein A280 > BSA, MicroArray > ssDNA-33 with Cy3/Cy5)."
            },
            "baseline_correction_wavelength": {
                "type": "float",
                "description": "The wavelength (nm) used for bichromatic normalization of absorbance data (e.g., 340 nm for Nucleic Acid, 750 nm for UV-Vis)."
            },
            "cuvette_temperature_setting": {
                "type": "float",
                "description": "Target temperature for the cuvette holder when heating is enabled (e.g., 37°C for NanoDrop 2000c)."
            },
            "instrument_operating_environment_temperature": {
                "type": "float",
                "description": "Ambient room temperature in degrees Celsius (°C) where the instrument is operated."
            },
            "instrument_operating_environment_humidity": {
                "type": "float",
                "description": "Ambient room humidity as a percentage."
            }
        }
        self.tool_data["tool_description"] = ["A spectrophotometer that measures 0.5-2 µL samples with high accuracy and reproducibility using a patented sample retention pedestal system. The 2000c model also includes a traditional cuvette capability. It measures full spectrum absorbance (190-840 nm)."]
        self.tool_data["risks_detection"]["description"] = "Potential risks include inaccurate measurements due to improper sample application, incorrect blanking (e.g. using a blank solution whose buffer/solvent pH or ionic strength doesn't match the sample, leading to negative absorbance readings or wrong purity ratios), pedestal contamination or unconditioning, sample evaporation, or operating outside specified environmental conditions. For the 2000c model, cuvette-related issues like incorrect volume or type can also cause errors."
        
    def _check_spectrophotometer_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Spectrophotometer."""
        violations = []
        
        # Check sample volume for pedestal
        if "sample_volume_pedestal" in parameters:
            volume = parameters["sample_volume_pedestal"]
            if volume < 0.5:
                violations.append({
                    "parameter": "sample_volume_pedestal",
                    "value": volume,
                    "threshold": 0.5,
                    "condition": "Using a 'sample_volume_pedestal' less than 0.5 µL (for concentrated Nucleic Acid/Protein A280 if small volume option selected) or generally less than 1 µL (for most aqueous solutions) may lead to improper liquid column formation and inaccurate readings. For samples with reduced surface tension (e.g. proteins), 2 µL is recommended to ensure proper column formation."
                })
        
        
        
        # Check blanking cycle pedestal absorbance deviation
        if "blanking_cycle_pedestal_absorbance_deviation" in parameters:
            deviation = parameters["blanking_cycle_pedestal_absorbance_deviation"]
            if deviation > 0.04:
                violations.append({
                    "parameter": "blanking_cycle_pedestal_absorbance_deviation",
                    "value": deviation,
                    "threshold": 0.04,
                    "condition": "During a blanking cycle check (measuring blank as a sample), if the resulting spectrum varies by more than 0.04 A (10 mm path equivalent) from a flat baseline, it indicates pedestal contamination or improper blanking, requiring cleaning and re-blanking."
                })
        
        # Check cuvette temperature setting
        if "cuvette_temperature_setting" in parameters:
            temp = parameters["cuvette_temperature_setting"]
            if abs(temp - 37.0) > 0.5:  # Assuming 37°C is the target temperature
                violations.append({
                    "parameter": "cuvette_temperature_setting",
                    "value": temp,
                    "threshold": {"target": 37.0, "deviation": 0.5},
                    "condition": "If the 'cuvette_temperature_setting' is enabled (e.g., 37°C), a deviation greater than ±0.5°C from the setpoint indicates a problem with the heating system, which could affect temperature-sensitive assays."
                })
        
        # Check instrument operating environment temperature
        if "instrument_operating_environment_temperature" in parameters:
            temp = parameters["instrument_operating_environment_temperature"]
            if temp < 15 or temp > 35:
                violations.append({
                    "parameter": "instrument_operating_environment_temperature",
                    "value": temp,
                    "threshold": {"lower_bound": 15, "upper_bound": 35},
                    "condition": "Operating the instrument with an 'instrument_operating_environment_temperature' above 35°C (or below 15°C) is outside specified conditions and may affect performance or damage the instrument."
                })
        
        # Check instrument operating environment humidity
        if "instrument_operating_environment_humidity" in parameters:
            humidity = parameters["instrument_operating_environment_humidity"]
            if humidity < 35 or humidity > 65:
                violations.append({
                    "parameter": "instrument_operating_environment_humidity",
                    "value": humidity,
                    "threshold": {"lower_bound": 35, "upper_bound": 65},
                    "condition": "Operating the instrument with an 'instrument_operating_environment_humidity' above 65% (or below 35%) is outside specified conditions and may lead to condensation or other issues affecting performance."
                })
        
        if violations:
            return {
                "allowed": False,
                "reason": "Parameters violate safety thresholds",
                "violations": violations
            }
        
        return {
            "allowed": True,
            "reason": "All safety checks passed",
            "concerns": []
        }
        
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
        """Safety check specifically for the Spectrophotometer."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class GelElectrophoresis(BiologyToolUtility):
    """Gel Electrophoresis tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the gel electrophoresis tool."""
        super().__init__(model)
        self.tool_name = "Bio-Rad Sub-Cell GT Agarose Gel Electrophoresis System"
        self.tool_data["input_parameters"] = {
            "agarose_concentration": {
                "type": "number",
                "description": "The percentage of agarose in the gel (e.g., 0.5%, 1.0%, 2.0%)."
            },
            "buffer_type_and_concentration": {
                "type": "string",
                "description": "The electrophoresis buffer used (e.g., 1x TAE, 1x TBE, 1x MOPS for RNA)."
            },
            "buffer_volume_and_depth": {
                "type": "number",
                "description": "The volume of electrophoresis buffer used, resulting in a specific depth over the gel in millimeters (mm)."
            },
            "sample_volume": {
                "type": "number",
                "description": "The volume of the sample loaded into each well in microliters (µL)."
            },
            "voltage": {
                "type": "number",
                "description": "The electrical voltage applied across the gel in Volts DC (VDC)."
            },
            "power": {
                "type": "number",
                "description": "The electrical power applied in Watts (W)."
            },
            "gel_casting_temperature": {
                "type": "number",
                "description": "The temperature of the molten agarose when poured for gel casting in degrees Celsius (°C)."
            }
        }
        self.tool_data["tool_description"] = ["A system for submerged agarose gel electrophoresis to separate nucleic acids (DNA or RNA) from 20 base pairs to 20 kilobase pairs. It includes a GT base (buffer chamber), safety lid, gel trays, and combs. Different models like Sub-Cell GT, Wide Mini-Sub Cell GT, and Mini-Sub Cell GT accommodate various gel sizes."]
        self.tool_data["risks_detection"]["description"] = "Potential risks include electrical shock, buffer leakage, damage to the apparatus from improper cleaning or overheating, and poor electrophoretic separation due to incorrect parameters. Safety interlocks are present on the lid. Specifically, applying voltage or power that is too high can cause overheating and buffer breakdown. Pouring gels with an agarose temperature greater than 60°C may damage trays and lead to uneven wells. Operating with buffer depth less than 2 mm can result in gel drying and poor separation, while exposing plastic parts to temperatures over 60°C during cleaning can cause damage."
        
    def _check_gel_electrophoresis_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Gel Electrophoresis System."""
        violations = []
        
        # Check voltage
        if "voltage" in parameters and parameters["voltage"] > 200:
            violations.append({
                "parameter": "voltage",
                "value": parameters["voltage"],
                "threshold": 200,
                "condition": "Exceeding the maximum voltage limit for the specific Sub-Cell GT model can lead to overheating, buffer breakdown, and potential damage to samples or equipment."
            })
        
        # Check power
        if "power" in parameters and parameters["power"] > 45:
            violations.append({
                "parameter": "power",
                "value": parameters["power"],
                "threshold": 45,
                "condition": "Exceeding the maximum power limit for the specific Sub-Cell GT model can cause excessive heating."
            })
        
        # Check gel casting temperature
        if "gel_casting_temperature" in parameters and parameters["gel_casting_temperature"] > 60:
            violations.append({
                "parameter": "gel_casting_temperature",
                "value": parameters["gel_casting_temperature"],
                "threshold": 60,
                "condition": "Pouring molten agarose at a 'gel_casting_temperature' greater than 60°C may cause the plastic base or UVTP tray to warp or craze, decreasing its lifetime and potentially leading to uneven sample wells."
            })
        
        # Check buffer volume and depth
        if "buffer_volume_and_depth" in parameters:
            depth = parameters["buffer_volume_and_depth"]
            if depth < 2 or depth > 6:
                violations.append({
                    "parameter": "buffer_volume_and_depth",
                    "value": depth,
                    "threshold": {"lower_bound": 2, "upper_bound": 6},
                    "condition": "Operating with a buffer depth less than 2 mm (threshold) or greater than 6 mm over the gel is not recommended. Insufficient buffer can lead to gel drying or pH/heat effects. Too much buffer can reduce migration rates or cause band distortion."
                })
        
        # Check cleaning agent temperature exposure
        if "cleaning_agent_temperature_exposure" in parameters and parameters["cleaning_agent_temperature_exposure"] > 60:
            violations.append({
                "parameter": "cleaning_agent_temperature_exposure",
                "value": parameters["cleaning_agent_temperature_exposure"],
                "threshold": 60,
                "condition": "Exposing Sub-Cell GT plastic parts to temperatures greater than 60°C during cleaning or any other process (e.g., autoclaving is not allowed) can cause them to crack, craze, etch, or warp."
            })
        
        if violations:
            return {
                "allowed": False,
                "reason": "Parameters violate safety thresholds",
                "violations": violations
            }
        
        return {
            "allowed": True,
            "reason": "All safety checks passed",
            "concerns": []
        }
        
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
        """Safety check specifically for the Gel Electrophoresis System."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class CO2Incubator(BiologyToolUtility):
    """CO2 Incubator tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the CO2 incubator tool."""
        super().__init__(model)
        self.tool_name = "Thermo Scientific Heracell Vios 160i CR / Heracell Vios 250i CR CO2 Incubator"
        self.tool_data["input_parameters"] = {
            "temperature_set_value": {
                "type": "number",
                "description": "The desired incubation temperature in degrees Celsius (°C)."
            },
            "co2_set_value": {
                "type": "number",
                "description": "The desired CO2 concentration in percentage (%)."
            },
            "o2_set_value": {
                "type": "number",
                "description": "The desired O2 concentration in percentage (%) (for models with O2/N2 control option)."
            },
            "n2_set_value": {
                "type": "number",
                "description": "Implicitly set when O2 is controlled below 21% by nitrogen displacement (for models with O2/N2 control option)."
            },
            "water_reservoir_fill_level": {
                "type": "string",
                "description": "The amount of sterile distilled water in the humidity reservoir (e.g., 'Filled to MAX', 'Below Min')."
            },
            "steri_run_cycle_activation": {
                "type": "boolean",
                "description": "Activation status of the 180°C Steri-Run sterilization routine."
            },
            "shelf_loading_weight": {
                "type": "number",
                "description": "Weight distribution on individual shelves in kilograms (kg)."
            },
            "total_unit_loading_weight": {
                "type": "number",
                "description": "Total weight of cultures and equipment inside the incubator in kilograms (kg)."
            }
        }
        self.tool_data["tool_description"] = ["A CO2 incubator designed for preparing and cultivating cell and tissue cultures by providing a controlled physiological ambient environment (temperature, CO2, O2/N2 (optional), relative humidity)."]
        self.tool_data["risks_detection"]["description"] = "Potential risks include compromised culture conditions due to deviations in temperature, CO2, O2, or humidity; contamination; damage to the incubator from incorrect gas pressures, improper environmental conditions, or misuse of sterilization cycles; and safety hazards related to gas handling or hot surfaces during sterilization. For example, setting the temperature above 55°C, CO2 above 20%, or O2 above 90% can lead to culture damage, unstable conditions, or increased fire risk with high O2. Exceeding shelf or total unit loading weight limits can cause structural damage. Operating with a dry water reservoir or using water of improper quality can impair humidity control, and large temperature deviations from the setpoint will trigger alarms and safety responses."
        
    def _check_co2_incubator_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for CO2 Incubator."""
        violations = []
        
        # Check temperature set value
        if "temperature_set_value" in parameters and parameters["temperature_set_value"] > 55:
            violations.append({
                "parameter": "temperature_set_value",
                "value": parameters["temperature_set_value"],
                "threshold": 55,
                "condition": "Setting the 'temperature_set_value' above the maximum control range of 55°C could lead to overheating and damage to cultures or the incubator. The incubator is designed to operate with incubation temperature at least 3°C above ambient."
            })
        
        # Check CO2 set value
        if "co2_set_value" in parameters and parameters["co2_set_value"] > 20:
            violations.append({
                "parameter": "co2_set_value",
                "value": parameters["co2_set_value"],
                "threshold": 20,
                "condition": "Setting the 'co2_set_value' above the maximum control range of 20% may not be achievable or stable."
            })
        
        # Check O2 set value
        if "o2_set_value" in parameters and parameters["o2_set_value"] > 90:
            violations.append({
                "parameter": "o2_set_value",
                "value": parameters["o2_set_value"],
                "threshold": 90,
                "condition": "For optional O2 control, setting the 'o2_set_value' above the maximum control range of 90% may not be achievable or stable. Operation with high O2 increases fire risk."
            })
        
        # Check shelf loading weight
        if "shelf_loading_weight" in parameters:
            weight = parameters["shelf_loading_weight"]
            if weight > 14:  # Using the more conservative threshold
                violations.append({
                    "parameter": "shelf_loading_weight",
                    "value": weight,
                    "threshold": 14,
                    "condition": "Exceeding the 'shelf_loading_weight' can damage shelves or support rails."
                })
        
        # Check total unit loading weight
        if "total_unit_loading_weight" in parameters and parameters["total_unit_loading_weight"] > 42:
            violations.append({
                "parameter": "total_unit_loading_weight",
                "value": parameters["total_unit_loading_weight"],
                "threshold": 42,
                "condition": "Exceeding the 'total_unit_loading_weight' can damage the incubator structure."
            })
        
        # Check temperature deviation from setpoint
        if "temperature_set_value" in parameters and "actual_temperature" in parameters:
            deviation = abs(parameters["actual_temperature"] - parameters["temperature_set_value"])
            if deviation > 1:
                violations.append({
                    "parameter": "temperature_deviation_from_setpoint_alarm",
                    "value": deviation,
                    "threshold": 1,
                    "condition": "If the actual temperature deviates by more than 1°C from the 'temperature_set_value', the overtemperature protection responds, and an alarm is triggered."
                })
        
        if violations:
            return {
                "allowed": False,
                "reason": "Parameters violate safety thresholds",
                "violations": violations
            }
        
        return {
            "allowed": True,
            "reason": "All safety checks passed",
            "concerns": []
        }
        
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
        """Safety check specifically for the CO2 Incubator."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters) 