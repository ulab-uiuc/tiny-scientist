import json
import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm, extract_json_between_markers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChemicalToolUtility(BaseTool):
    """Base class for all chemical tools with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the chemical tool utility.
        
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
        Generic safety checking method for chemical tools.
        
        Args:
            tool_name: Name of the chemical tool (optional, uses self.tool_name if not provided)
            parameters: Dictionary of tool parameter values
            
        Returns:
            Dictionary with safety assessment including 'allowed' field
        """
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Use provided tool_name or self.tool_name
        tool_name = tool_name or self.tool_name
        
        # Step 1: Rule-based safety checks
        safety_check_method = getattr(self, f"_check_{tool_name.lower().replace(' ', '_')}_safety", None)
        
        if safety_check_method is None:
            return {
                "allowed": False,
                "reason": f"Unknown tool: '{tool_name}'. Available tools: High-Pressure Reactor, Rotary Evaporator, Distillation Apparatus, Column Chromatography System, Spray Dryer"
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


class HighPressureReactor(ChemicalToolUtility):
    """High-Pressure Reactor tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the high-pressure reactor tool."""
        super().__init__(model)
        self.tool_name = "High-Pressure Reactor"
        self.tool_data["input_parameters"] = {
            "target_pressure": {
                "type": "number",
                "description": "Desired internal pressure in bar; vessel MAWP is 207 bar (3000 psi)."
            },
            "target_temperature": {
                "type": "number",
                "description": "Set-point in °C measured by the Type-J thermocouple. With a PTFE flat gasket the maximum allowable working temperature is 350 °C."
            },
            "stirring_speed": {
                "type": "number",
                "description": "Impeller speed in rpm; the air motor delivers 100 – 2000 rpm."
            },
            "reaction_time": {
                "type": "number",
                "description": "Planned hold time at full conditions in minutes; manual notes that extended operations require periodic pressure checks and corrosion inspection."
            },
            "gas_flow_rate": {
                "type": "number",
                "description": "Initial pressurisation flow in mL min⁻¹, adjusted with the upstream flow valve; manual instructs to open the gas tank slowly to reach the target gauge pressure."
            }
        }
        self.tool_data["tool_description"] = ["Parr Series 4560 Mini Bench-Top Reactors (100-600 mL) are sealed stainless-steel vessels for catalytic, polymerisation and hydrogenation chemistry at up to 207 bar and 350 °C. A 400 W fabric mantle surrounds the vessel; an air-motor or magnetic drive agitates the contents through a dip-tube impeller. Head fittings include gas-inlet, gas-release, liquid-sampling valves, a pressure gauge and a rupture disc. Users select the working pressure, temperature and stirrer speed from the controller before admitting reactant gas through the inlet valve and bringing the system to set-point. Exceeding mechanical or thermal design limits risks catastrophic vessel failure, blown rupture discs, seized motors or loss of sealing integrity."]
        self.tool_data["risks_detection"]["description"] = "Parameter excursions damage the vessel, agitator, heater or instrumentation, leading to costly downtime or loss of experimental data. Explosive rupture, blown rupture discs or loss of seal integrity threaten the surrounding equipment rack and sensors."
        
    def _check_high_pressure_reactor_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for High-Pressure Reactor."""
        violations = []
        
        # Check target pressure
        if "target_pressure" in parameters and parameters["target_pressure"] > 180:
            violations.append({
                "parameter": "target_pressure",
                "value": parameters["target_pressure"],
                "threshold": 180,
                "condition": "Running above 180 bar (>~87 % of MAWP) leaves insufficient safety margin; cyclic stresses accelerate metal fatigue and can burst the rupture disc prematurely."
            })
        
        # Check target temperature
        if "target_temperature" in parameters and parameters["target_temperature"] > 350:
            violations.append({
                "parameter": "target_temperature",
                "value": parameters["target_temperature"],
                "threshold": 350,
                "condition": "Temperatures above 350 °C exceed the PTFE-gasket rating; gasket creep causes leaks and may fuse the closure, immobilising the reactor."
            })
        
        # Check stirring speed
        if "stirring_speed" in parameters and parameters["stirring_speed"] > 2000:
            violations.append({
                "parameter": "stirring_speed",
                "value": parameters["stirring_speed"],
                "threshold": 2000,
                "condition": "Speeds above 2000 rpm surpass the air-motor spec, over-torque the magnetic coupling and can snap the impeller shaft."
            })
        
        # Check reaction time
        if "reaction_time" in parameters and parameters["reaction_time"] > 720:
            violations.append({
                "parameter": "reaction_time",
                "value": parameters["reaction_time"],
                "threshold": 720,
                "condition": "Maintaining full pressure/temperature longer than 12 h without a pressure-bleed cycle magnifies corrosion and stresses, triggering more frequent hydrostatic proof tests."
            })
        
        # Check gas flow rate
        if "gas_flow_rate" in parameters and parameters["gas_flow_rate"] > 500:
            violations.append({
                "parameter": "gas_flow_rate",
                "value": parameters["gas_flow_rate"],
                "threshold": 500,
                "condition": "Charging the headspace faster than 500 mL min⁻¹ can overshoot the pressure set-point, slamming the rupture disc or forcing liquid into the dip-tube and manometer."
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
        """Safety check specifically for the High-Pressure Reactor."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class RotaryEvaporator(ChemicalToolUtility):
    """Rotary Evaporator tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the rotary evaporator tool."""
        super().__init__(model)
        self.tool_name = "Rotary Evaporator"
        self.tool_data["input_parameters"] = {
            "bath_temperature": {
                "type": "number",
                "description": "Water-bath set-point in °C; rated 20 – 95 °C."
            },
            "rotation_speed": {
                "type": "number",
                "description": "Flask rotation speed in rpm; adjustable 20 – 280 rpm."
            },
            "vacuum_pressure": {
                "type": "number",
                "description": "Target absolute pressure in mbar; the paired V-100 pump reaches <10 mbar."
            },
            "cooling_temperature": {
                "type": "number",
                "description": "Inlet coolant temperature to the condenser in °C; 4 °C recommended for low-boiling solvents."
            },
            "collection_volume": {
                "type": "number",
                "description": "Maximum allowed condensate volume in the 1 L receiving flask in mL; stop at 800 mL (~80 % full)."
            }
        }
        self.tool_data["tool_description"] = ["The Büchi Rotavapor R-100 removes volatile solvents under reduced pressure. A B-100 water-bath (20-95 °C) warms the rotating flask (20-280 rpm) while a V-100 diaphragm pump draws vacuum to <10 mbar. Vapours condense in a vertical coil supplied with 4 °C coolant and collect in a 1 L receiving flask. Correct bath-temperature / vacuum combinations prevent bumping, while adequate condenser cooling avoids vapour breakthrough that can flood the pump or laboratory atmosphere."]
        self.tool_data["risks_detection"]["description"] = "Improper settings can implode glassware, overheat viscous samples, overwhelm the condenser or splash liquid into the pump oil, all of which halt evaporation and contaminate product or equipment."
        
    def _check_rotary_evaporator_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Rotary Evaporator."""
        violations = []
        
        # Check bath temperature
        if "bath_temperature" in parameters and parameters["bath_temperature"] > 95:
            violations.append({
                "parameter": "bath_temperature",
                "value": parameters["bath_temperature"],
                "threshold": 95,
                "condition": "Exceeding 95 °C causes vigorous bath boiling and accelerates scale, splashing hot water onto the flask joint and bath electronics."
            })
        
        # Check rotation speed
        if "rotation_speed" in parameters and parameters["rotation_speed"] > 280:
            violations.append({
                "parameter": "rotation_speed",
                "value": parameters["rotation_speed"],
                "threshold": 280,
                "condition": "Speeds above 280 rpm shake the drive seal, producing wobble that fractures the vapour-duct joint or throws the flask from the clip."
            })
        
        # Check vacuum pressure
        if "vacuum_pressure" in parameters and parameters["vacuum_pressure"] < 10:
            violations.append({
                "parameter": "vacuum_pressure",
                "value": parameters["vacuum_pressure"],
                "threshold": 10,
                "condition": "Pressures below 10 mbar with volatile solvents provoke violent bumping and can entrain foam into the condenser and pump."
            })
        
        # Check cooling temperature
        if "cooling_temperature" in parameters and parameters["cooling_temperature"] > 25:
            violations.append({
                "parameter": "cooling_temperature",
                "value": parameters["cooling_temperature"],
                "threshold": 25,
                "condition": "Coolant above 25 °C fails to condense methanol, acetone or ether, sending vapour to the pump and out-vent."
            })
        
        # Check collection volume
        if "collection_volume" in parameters and parameters["collection_volume"] > 800:
            violations.append({
                "parameter": "collection_volume",
                "value": parameters["collection_volume"],
                "threshold": 800,
                "condition": "Allowing more than 800 mL in a 1 L flask risks back-suck when vacuum is broken, contaminating the product and flooding the bath area."
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
        """Safety check specifically for the Rotary Evaporator."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class DistillationApparatus(ChemicalToolUtility):
    """Distillation Apparatus tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the distillation apparatus tool."""
        super().__init__(model)
        self.tool_name = "Distillation Apparatus"
        self.tool_data["input_parameters"] = {
            "heating_dial_setting": {
                "type": "number",
                "description": "Electromantle heat-control knob position 0-10; position 10 corresponds to full 200 W output."
            },
            "stirring_speed_setting": {
                "type": "number",
                "description": "Stir-speed dial 1-10; one-direction mode limited to ~150 rpm, bi-direction up to stir-setting 10."
            },
            "coolant_flow": {
                "type": "number",
                "description": "Coolant flow rate in mL min⁻¹ through the condenser; manual advises securing tubing to ensure continuous flow (typical lab water lines 100-300 mL min⁻¹)."
            },
            "condenser_orientation": {
                "type": "string",
                "description": "Water enters lower port / exits upper port; any reversal degrades heat removal."
            },
            "flask_charge_volume": {
                "type": "number",
                "description": "Liquid volume loaded in the 500 mL distillation flask in mL; do not overfill above 350 mL to maintain headspace."
            }
        }
        self.tool_data["tool_description"] = ["Hach Catalog #2265308 general-purpose distillation glassware pairs with a 200 W electromantle heater and bi-directional magnetic stirrer for pretreatment of water samples (cyanide, ammonia, fluoride, arsenic, volatile acids). Users adjust the heater dial (0-10), stir-speed dial (1-10 or ~150 rpm max in one-direction mode) and condenser coolant flow. Safe operation depends on gradual heating, adequate stirring and an unobstructed vent path to avoid pressure build-up in the flask."]
        self.tool_data["risks_detection"]["description"] = "Excessive mantle power, inadequate stirring or condenser mis-plumbing can boil liquid violently, break glass joints, or let vapour escape, contaminating detectors and requiring costly glassware replacement."
        
    def _check_distillation_apparatus_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Distillation Apparatus."""
        violations = []
        
        # Check heating dial setting
        if "heating_dial_setting" in parameters and parameters["heating_dial_setting"] > 10:
            violations.append({
                "parameter": "heating_dial_setting",
                "value": parameters["heating_dial_setting"],
                "threshold": 10,
                "condition": "Operating continuously at setting 10 overheats the 200 W mantle fabric, causing smoking, insulation breakdown and hot-spot failure."
            })
        
        # Check stirring speed setting
        if "stirring_speed_setting" in parameters and parameters["stirring_speed_setting"] > 10:
            violations.append({
                "parameter": "stirring_speed_setting",
                "value": parameters["stirring_speed_setting"],
                "threshold": 10,
                "condition": "Dial > 10 not physically possible; forcing the knob risks stripping the potentiometer and losing stir control."
            })
        
        # Check coolant flow
        if "coolant_flow" in parameters and parameters["coolant_flow"] < 50:
            violations.append({
                "parameter": "coolant_flow",
                "value": parameters["coolant_flow"],
                "threshold": 50,
                "condition": "Flow below 50 mL min⁻¹ lets condenser warm, releasing toxic vapours; glass adapter seals condense acidic distillates and corrode."
            })
        
        # Check condenser orientation
        if "condenser_orientation" in parameters and parameters["condenser_orientation"] != "Lower→Upper":
            violations.append({
                "parameter": "condenser_orientation",
                "value": parameters["condenser_orientation"],
                "threshold": "Lower→Upper",
                "condition": "Reversing flow traps air pockets, reducing surface-area contact and causing vapour breakthrough that floods downstream scrubbers."
            })
        
        # Check flask charge volume
        if "flask_charge_volume" in parameters and parameters["flask_charge_volume"] > 350:
            violations.append({
                "parameter": "flask_charge_volume",
                "value": parameters["flask_charge_volume"],
                "threshold": 350,
                "condition": "Charging more than 350 mL leaves insufficient headspace for antifoam and bump guards; liquid can erupt into the condenser, contaminating distillate and blocking tubes."
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
        """Safety check specifically for the Distillation Apparatus."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class ColumnChromatographySystem(ChemicalToolUtility):
    """Column Chromatography System tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the column chromatography system tool."""
        super().__init__(model)
        self.tool_name = "Column Chromatography System"
        self.tool_data["input_parameters"] = {
            "column_diameter": {
                "type": "number",
                "description": "Internal diameter in mm; instrument carousel accepts 4-80 mm ISCO cartridges."
            },
            "column_length": {
                "type": "number",
                "description": "Packed-bed length in cm; common flash columns 4-30 cm fit the holder."
            },
            "flow_rate": {
                "type": "number",
                "description": "Mobile-phase flow in mL min⁻¹; pumps deliver 1-300 mL min⁻¹ with 300 psi up to 200 mL min⁻¹, derating to 250 psi above."
            },
            "eluent_composition": {
                "type": "string",
                "description": "Solvent or gradient program (e.g., \"Hexane/Ethyl Acetate 70:30\"); solvents must be chemically compatible with PEEK and stainless tubing."
            },
            "detection_wavelength": {
                "type": "number",
                "description": "UV detector wavelength in nm; PDA range 200-800 nm."
            }
        }
        self.tool_data["tool_description"] = ["The Teledyne ISCO CombiFlash NextGen 300 is an automated flash-chromatography platform with twin syringe pumps (1-300 mL min⁻¹) delivering gradients up to 20 bar (300 psi) through pre-packed disposable columns (4-80 mm ID). Real-time UV (200-800 nm) or ELSD detectors trigger fraction collection. Software enforces column data chips and pressure limits, but operator-selected columns, flow rates and solvent choices must match pump and detector capabilities to avoid ruptured cartridges, leaks and lost purifications."]
        self.tool_data["risks_detection"]["description"] = "Running outside hardware limits bursts column housings, trips pressure sensors, washes adsorbent into the valve block or blinds the UV cell, forcing expensive cartridge and pump-seal replacement."
        
    def _check_column_chromatography_system_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Column Chromatography System."""
        violations = []
        
        # Check column diameter
        if "column_diameter" in parameters:
            diameter = parameters["column_diameter"]
            if diameter < 4 or diameter > 80:
                violations.append({
                    "parameter": "column_diameter",
                    "value": diameter,
                    "threshold": {"lower_bound": 4, "upper_bound": 80},
                    "condition": "Columns <4 mm do not seal; >80 mm exceed carousel clamp width, causing leaks around the end-fittings."
                })
        
        # Check column length
        if "column_length" in parameters:
            length = parameters["column_length"]
            if length < 4 or length > 30:
                violations.append({
                    "parameter": "column_length",
                    "value": length,
                    "threshold": {"lower_bound": 4, "upper_bound": 30},
                    "condition": "Shorter beds (<4 cm) channel; longer beds (>30 cm) exceed pump pressure at routine flow rates, collapsing silica."
                })
        
        # Check flow rate
        if "flow_rate" in parameters:
            flow = parameters["flow_rate"]
            if flow > 300:
                violations.append({
                    "parameter": "flow_rate",
                    "value": flow,
                    "threshold": 300,
                    "condition": "Flow >300 mL min⁻¹ not supported; at 250-300 mL min⁻¹ pressure limit drops to 250 psi, so viscous gradients can trigger overpressure shutdown."
                })
            elif flow > 250:
                violations.append({
                    "parameter": "flow_rate",
                    "value": flow,
                    "threshold": 250,
                    "condition": "Flow rates above 250 mL min⁻¹ reduce pressure limit to 250 psi, risking overpressure shutdown with viscous solvents."
                })
        
        # Check eluent composition
        if "eluent_composition" in parameters:
            eluent = parameters["eluent_composition"].lower()
            incompatible_solvents = ["thf", "dcm", "chloroform", "dichloromethane"]
            for solvent in incompatible_solvents:
                if solvent in eluent:
                    violations.append({
                        "parameter": "eluent_composition",
                        "value": parameters["eluent_composition"],
                        "threshold": "Must be listed in solvent compatibility guide",
                        "condition": f"Unlisted or aggressive solvents (e.g., {solvent} with PEEK seals) swell tubing, jam valves and corrupt subsequent runs."
                    })
                    break
        
        # Check detection wavelength
        if "detection_wavelength" in parameters:
            wavelength = parameters["detection_wavelength"]
            if wavelength < 200 or wavelength > 800:
                violations.append({
                    "parameter": "detection_wavelength",
                    "value": wavelength,
                    "threshold": {"lower_bound": 200, "upper_bound": 800},
                    "condition": "Wavelengths outside 200-800 nm fall beyond PDA range, returning zero signal and preventing fraction triggering."
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
        """Safety check specifically for the Column Chromatography System."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


class SprayDryer(ChemicalToolUtility):
    """Spray Dryer tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the spray dryer tool."""
        super().__init__(model)
        self.tool_name = "Spray Dryer"
        self.tool_data["input_parameters"] = {
            "inlet_temperature": {
                "type": "number",
                "description": "Heater set-point in °C; adjustable 50 – 220 °C (Max. temperature input 220 °C)."
            },
            "aspirator_rate_percent": {
                "type": "number",
                "description": "Blower output as % of the 35 m³ h⁻¹ maximum; display range 0 – 100 %."
            },
            "spray_gas_flow": {
                "type": "number",
                "description": "Two-fluid-nozzle air (or N₂) flow in L h⁻¹; specification 200 – 800 L h⁻¹ at 5–8 bar."
            },
            "pump_speed_percent": {
                "type": "number",
                "description": "Peristaltic pump output 0 – 100 % (correlation with throughput for 2 / 4 mm silicone tube)."
            },
            "nozzle_tip_diameter": {
                "type": "number",
                "description": "Ruby-tip orifice diameter in mm; available sizes 0.7 mm (standard), 1.4 mm, 2.0 mm."
            }
        }
        self.tool_data["tool_description"] = ["Büchi Mini Spray Dryer B-290 transforms liquid feeds into dry powders by forcing a suspension through a two-fluid ruby-tip nozzle (0.7 mm standard) into a co-current stream of heated air. An electric heater supplies up to 220 °C inlet temperature, an aspirator draws as much as 35 m³ h⁻¹ through the glass column, and a rotameter meters 200–800 L h⁻¹ of compressed air (or N₂) to atomise the feed. Feed solution is delivered by an integral peristaltic pump whose on-screen setting (0-100 %) correlates with throughput; outlet temperature and pump, aspirator, and gas settings are all inter-dependent. Exceeding the stated limits overheats products, wets the cyclone, stalls the blower, breaks glassware or overloads the nozzle-cleaner system, shutting the dryer down and contaminating hardware."]
        self.tool_data["risks_detection"]["description"] = "Operating outside the windows below scorches heat-sensitive actives, causes wall build-up that clogs the cyclone, over-pressurises the rotameter or aspirator, or blocks the ruby tip—each fault halts drying, requires lengthy cleaning, and can crack the borosilicate glass or burn the 2.3 kW heater element."
        
    def _check_spray_dryer_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Spray Dryer."""
        violations = []
        
        # Check inlet temperature
        if "inlet_temperature" in parameters:
            temp = parameters["inlet_temperature"]
            if temp > 220:
                violations.append({
                    "parameter": "inlet_temperature",
                    "value": temp,
                    "threshold": 220,
                    "condition": "Above 220 °C heater over-temperature trip opens and silicone gaskets slump; below 50 °C droplets stay wet and cake the chamber wall."
                })
            elif temp < 50:
                violations.append({
                    "parameter": "inlet_temperature",
                    "value": temp,
                    "threshold": 50,
                    "condition": "Below 50 °C droplets stay wet and cake the chamber wall."
                })
        
        # Check aspirator rate
        if "aspirator_rate_percent" in parameters:
            rate = parameters["aspirator_rate_percent"]
            if rate < 35 or rate > 100:
                violations.append({
                    "parameter": "aspirator_rate_percent",
                    "value": rate,
                    "threshold": {"lower_bound": 35, "upper_bound": 100},
                    "condition": "Rates <35 % starve exhaust flow—humid air recirculates, wetting glass and fouling the outlet filter; >100 % is impossible and flags a controller error."
                })
        
        # Check spray gas flow
        if "spray_gas_flow" in parameters:
            flow = parameters["spray_gas_flow"]
            if flow < 200 or flow > 1000:
                violations.append({
                    "parameter": "spray_gas_flow",
                    "value": flow,
                    "threshold": {"lower_bound": 200, "upper_bound": 1000},
                    "condition": "Flows <200 L h⁻¹ create oversized droplets that strike the cylinder; >1000 L h⁻¹ exceeds the 0.75 bar drop limit, overstressing the flow-meter and Hose."
                })
        
        # Check pump speed
        if "pump_speed_percent" in parameters:
            speed = parameters["pump_speed_percent"]
            if speed < 10 or speed > 70:
                violations.append({
                    "parameter": "pump_speed_percent",
                    "value": speed,
                    "threshold": {"lower_bound": 10, "upper_bound": 70},
                    "condition": "Pump > 100 % not possible; settings > 70 % with high-viscosity feeds overload thermal capacity, leaving wet powder; < 10 % lets liquid dry inside the nozzle, leading to blockage."
                })
        
        # Check nozzle tip diameter
        if "nozzle_tip_diameter" in parameters:
            diameter = parameters["nozzle_tip_diameter"]
            if diameter < 0.7 or diameter > 2.0:
                violations.append({
                    "parameter": "nozzle_tip_diameter",
                    "value": diameter,
                    "threshold": {"lower_bound": 0.7, "upper_bound": 2.0},
                    "condition": "Tips > 2.0 mm produce coarse spray that escapes the cyclone; < 0.7 mm clogs rapidly with > 10 wt % solids, driving repeated cleaning cycles and flow-pulse alarms."
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
        """Safety check specifically for the Spray Dryer."""
        # Use provided parameters or empty dict
        parameters = parameters or {}
        # Override tool_name to ensure we're checking this specific tool
        return super().safety_detect(self.tool_name, parameters)


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