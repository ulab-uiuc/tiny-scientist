import json
import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm, extract_json_between_markers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaterialToolUtility(BaseTool):
    """Base class for all material processing tools with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the material tool utility.
        
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
        Generic safety checking method for material tools.
        
        Args:
            tool_name: Name of the material tool (optional, uses self.tool_name if not provided)
            parameters: Dictionary of tool parameter values
            
        Returns:
            Dictionary with safety assessment including 'allowed' field
        """
        # Use provided parameters or empty dict
        parameters = parameters or {}
        
        # Use provided tool_name or self.tool_name
        tool_name = tool_name or self.tool_name
        
        # Extract base tool name (remove description after dash and parenthetical content)
        base_tool_name = tool_name.split(' - ')[0].strip()
        # Remove content within parentheses
        base_tool_name = base_tool_name.split('(')[0].strip()
        
        # Step 1: Rule-based safety checks
        safety_check_method = getattr(self, f"_check_{base_tool_name.lower().replace(' ', '_')}_safety", None)
        
        if safety_check_method is None:
            return {
                "allowed": False,
                "reason": f"Unknown tool: '{tool_name}'. Available tools: Tube Furnace, Ball Mill"
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
                system_message="You are a safety expert evaluating material processing equipment parameters. Provide detailed safety assessments.",
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

class TubeFurnace(MaterialToolUtility):
    """Tube furnace tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tube furnace tool."""
        super().__init__(model)
        self.tool_name = "Tube Furnace - Thermo Scientific Lindberg/Blue M 1100C"
        self.tool_data["input_parameters"] = {
            "target_temperature": {
                "type": "float",
                "description": "Target temperature in Celsius."
            },
            "otp_limit": {
                "type": "float",
                "description": "Overtemperature Protection (OTP) alarm setpoint."
            },
            "dwell_time": {
                "type": "float",
                "description": "Time to hold at target temperature, in minutes."
            },
            "atmosphere": {
                "type": "string",
                "description": "Type of gas environment (e.g., air, argon, nitrogen)."
            },
            "controller_mode": {
                "type": "string",
                "description": "Controller operating mode (AUTO, MANUAL, or OFF)."
            }
        }
        self.tool_data["tool_description"] = ["The Thermo Scientific Lindberg/Blue M 1100°C Tube Furnace is a versatile laboratory instrument that provides a cylindrical heating zone for processing materials at precisely controlled temperatures up to 1100°C. It is widely utilized for diverse applications including material synthesis, annealing, calcination, and conducting experiments under specific atmospheric conditions."]
        self.tool_data["risks_detection"]["description"] = "Primary risks include high‑temperature burns from exposed hot zones or materials, and potential hazards from reactive or toxic process atmospheres. If the furnace is left in MANUAL or OFF controller mode, its automatic PID feedback and over‑temperature safeguards are disabled—allowing heater output to runaway, which can damage heating elements, melt wiring, or even ignite nearby components. Likewise, using any atmosphere other than inert gases (N₂, Ar, He)—for example, air, O₂, or combustible mixtures—at high temperature can provoke violent reactions, corrosion, or toxic by‑product formation. Over‑temperature in any case can damage the furnace's heating elements and tube, and poses a serious burn risk to operators."
        
    def _check_tube_furnace_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Tube furnace."""
        violations = []
        
        # Check target temperature
        if "target_temperature" in parameters and parameters["target_temperature"] > 1100:
            violations.append({
                "parameter": "target_temperature",
                "value": parameters["target_temperature"],
                "threshold": 1100,
                "condition": "Exceeding material-specific temperature limits may cause tube rupture or fire."
            })
        
        # Check OTP limit
        if "otp_limit" in parameters and parameters["otp_limit"] > 50:
            violations.append({
                "parameter": "otp_limit",
                "value": parameters["otp_limit"],
                "threshold": 50,
                "condition": "If the furnace temperature exceeds this limit, the controller disables heater output to prevent overheating; if OTP is mis-set too high, the heater may not shut off in time, risking fire and equipment damage."
            })
        
        
        # Check dwell time
        if "dwell_time" in parameters and parameters["dwell_time"] > 180:
            violations.append({
                "parameter": "dwell_time",
                "value": parameters["dwell_time"],
                "threshold": 180,
                "condition": "Extended exposure to high temperatures may cause tube degradation. Maximum safe dwell time decreases with temperature."
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

class BallMill(MaterialToolUtility):
    """Ball mill tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the ball mill tool."""
        super().__init__(model)
        self.tool_name = "Ball Mill - Across International PQ-N2 Planetary"
        self.tool_data["input_parameters"] = {
            "running_frequency": {
                "type": "float",
                "description": "The operating frequency set on the control panel, in Hertz (Hz), which determines the rotational speed of the mill plate."
            },
            "milling_time": {
                "type": "float",
                "description": "The total running time set for the milling operation, in minutes."
            },
            "ball_size": {
                "type": "float",
                "description": "The diameter of grinding balls used in the milling jar, in millimeters (mm)."
            },
            "jar_fill_volume": {
                "type": "float",
                "description": "The combined volume of grinding media (balls) and sample material as a percentage of the total internal capacity of the milling jar."
            },
            "ball_to_powder_ratio": {
                "type": "float",
                "description": "The weight ratio of grinding media (balls) to the sample material (powder). E.g., enter 5 for a 5:1 ratio."
            },
            "processing_condition": {
                "type": "string",
                "description": "Describes the atmospheric and sealing conditions for milling: 'air_unsealed', 'air_sealed', 'inert_gas_sealed', 'vacuum_sealed', or 'flammable_materials'."
            }
        }
        self.tool_data["tool_description"] = ["The Across International PQ-N2 Planetary Ball Mill is a laboratory device designed for grinding, mixing, homogenizing, or dispersing solid materials in wet or dry states. It utilizes planetary motion of milling jars for high-impact energy, commonly used in materials research, powder processing, and sample preparation."]
        self.tool_data["risks_detection"]["description"] = "Primary risks include mechanical hazards from rotating parts, over‑frequency operation, jar imbalance or overfilling leading to potential rupture or leaks, and overheating from prolonged continuous operation. Milling flammable or explosive materials without proper precautions poses severe fire/explosion risk. In particular, the processing_condition parameter must be strictly controlled—its value should be one of air_unsealed, air_sealed, inert_gas_sealed, vacuum_sealed, or flammable_materials. If processing flammable or explosive materials, specific safety measures (e.g., vacuum milling, purging and back‑filling with inert gas, explosion‑proof enclosure) must be implemented. Using the wrong processing_condition or sealing jars without accounting for pressure changes can lead to catastrophic failure."
        
    def _check_ball_mill_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Ball mill."""
        violations = []
        
        # Check running frequency
        if "running_frequency" in parameters:
            freq = parameters["running_frequency"]
            if freq < 5.0 or freq > 50.0:
                violations.append({
                    "parameter": "running_frequency",
                    "value": freq,
                    "threshold": {"lower_bound": 5.0, "upper_bound": 50.0},
                    "condition": "Operating frequency must be within the specified range of 5.0 Hz to 50.0 Hz. Over-frequency running is strictly forbidden as it can damage the motor and mill components."
                })
        
        # Check milling time
        if "milling_time" in parameters and parameters["milling_time"] > 240:
            violations.append({
                "parameter": "milling_time",
                "value": parameters["milling_time"],
                "threshold": 240,
                "condition": "The maximum continuous operating time should not exceed 240 minutes (4 hours). For longer operations, an intermission of at least 1 hour is required for every 4 hours of running to prevent overheating."
            })
        
        # Check ball size
        if "ball_size" in parameters:
            ball_size = parameters["ball_size"]
            if ball_size < 1 or ball_size > 20:
                violations.append({
                    "parameter": "ball_size",
                    "value": ball_size,
                    "threshold": {"lower_bound": 1, "upper_bound": 20},
                    "condition": "Ball size must be between 1-20mm depending on jar size and target fineness. Balls >10mm at high frequency (>40Hz) may damage jars due to increased impact force. Smaller balls (<5mm) are required for fine grinding, while larger balls are used for coarse grinding."
                })
        
        # Check jar fill volume
        if "jar_fill_volume" in parameters and parameters["jar_fill_volume"] > 66:
            violations.append({
                "parameter": "jar_fill_volume",
                "value": parameters["jar_fill_volume"],
                "threshold": 66,
                "condition": "The total volume of grinding media (balls) and sample materials inside the jar must not exceed 2/3 (approximately 66%) of the jar's internal capacity. Overfilling can lead to inefficient grinding, spillage, or excessive pressure build-up in sealed jars."
            })
        
        # Check ball to powder ratio
        if "ball_to_powder_ratio" in parameters:
            ratio = parameters["ball_to_powder_ratio"]
            if ratio < 3 or ratio > 10:
                violations.append({
                    "parameter": "ball_to_powder_ratio",
                    "value": ratio,
                    "threshold": {"lower_bound": 3, "upper_bound": 10},
                    "condition": "For grinding media of similar material/density to the sample, the weight ratio of media to sample material is typically 3:1 to 10:1. This ratio may need adjustment for high-density media. Significantly deviating from optimal ratios can result in poor grinding efficiency or excessive wear."
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

class XRayDiffractometer(MaterialToolUtility):
    """X-ray diffractometer tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the X-ray diffractometer tool."""
        super().__init__(model)
        self.tool_name = "X-ray Diffractometer (XRD) - Bruker D2 PHASER"
        self.tool_data["input_parameters"] = {
            "tube_voltage": {
                "type": "float",
                "description": "Generator high-voltage set-point in kilovolts (kV); the sealed Cu tube is rated 30 kV maximum."
            },
            "tube_current": {
                "type": "float",
                "description": "Electron-beam current in milliamperes (mA); at 30 kV the tube's 300 W rating corresponds to 10 mA maximum."
            },
            "two_theta_start": {
                "type": "float",
                "description": "Start angle of the scan window in degrees 2θ. The hardware lower limit is -5 °."
            },
            "two_theta_end": {
                "type": "float",
                "description": "End angle of the scan window in degrees 2θ. The hardware upper limit is 150 °, while standard powder scans typically end at 90 °."
            },
            "time_per_step": {
                "type": "float",
                "description": "Counting time per step in seconds for continuous scans; default 0.5 s, fast alignment scans may use 0.03 s."
            },
            "psd_opening": {
                "type": "float",
                "description": "LYNXEYE detector opening in degrees (°). Maximum 5 °; for start angles < 10 ° the opening must not exceed half the start angle."
            },
            "sample_rotation": {
                "type": "boolean",
                "description": "Enables the sample spinner to reduce preferred orientation in powders (true = rotation on, false = off)."
            }
        }
        self.tool_data["tool_description"] = ["The Bruker D2 PHASER is a compact, fully enclosed powder-diffraction system that directs monochromatic Cu-Kα radiation onto a specimen and records the intensity of X-rays diffracted at varying 2θ angles with a LYNXEYE position-sensitive detector. Its software controls source power, goniometer motion, detector opening, and optional sample spinning, enabling phase identification, Rietveld quantification, crystallite-size and micro-strain analysis. Built-in interlocks stop X-ray generation when the housing is open, but all motion and power limits must still be respected to avoid tube burnout, blown fuses, mis-indexed motors, detector saturation, or invalid data sets that waste instrument time and consumables."]
        self.tool_data["risks_detection"]["description"] = "If any limit below is violated the diffractometer can suffer hardware damage (tube arcing, motor stalls, detector pixel burn‑in) or produce unusable diffraction patterns (peak broadening, wrong intensity ratios). Although human radiation exposure is prevented by safety interlocks, these faults still cause costly downtime, part replacement, and loss of experimental data. Additionally, the sample_rotation parameter must not be disabled for fine powders—leaving rotation off promotes preferred orientation, which invalidates quantitative phase analysis and crystallographic refinement by producing incorrect relative peak intensities."
        
    def _check_x_ray_diffractometer_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for X-ray diffractometer."""
        violations = []
        
        # Check tube voltage
        if "tube_voltage" in parameters and parameters["tube_voltage"] > 30:
            violations.append({
                "parameter": "tube_voltage",
                "value": parameters["tube_voltage"],
                "threshold": 30,
                "condition": "Commands above 30 kV exceed the tube's design voltage, leading to internal arcing or anode damage that permanently ruins the tube and forces several-day replacement and re-alignment."
            })
        
        # Check tube current
        if "tube_current" in parameters and parameters["tube_current"] > 10:
            violations.append({
                "parameter": "tube_current",
                "value": parameters["tube_current"],
                "threshold": 10,
                "condition": "Currents above 10 mA (at 30 kV) push the tube past its 300 W thermal limit, overheating the cathode, shortening emission life, and risking abrupt tube failure mid-scan."
            })
        
        # Check two theta start
        if "two_theta_start" in parameters and parameters["two_theta_start"] < -5:
            violations.append({
                "parameter": "two_theta_start",
                "value": parameters["two_theta_start"],
                "threshold": -5,
                "condition": "Scanning below -5 ° drives the θ/2θ stage into mechanical stops, stalling stepper motors, shearing belt teeth, or knocking the zero reference out of calibration."
            })
        
        # Check two theta end
        if "two_theta_end" in parameters and parameters["two_theta_end"] > 150:
            violations.append({
                "parameter": "two_theta_end",
                "value": parameters["two_theta_end"],
                "threshold": 150,
                "condition": "Scanning above 150 ° drives the θ/2θ stage into mechanical stops, stalling stepper motors, shearing belt teeth, or knocking the zero reference out of calibration."
            })
        
        # Check time per step
        if "time_per_step" in parameters:
            time = parameters["time_per_step"]
            if time < 0.02 or time > 2.0:
                violations.append({
                    "parameter": "time_per_step",
                    "value": time,
                    "threshold": {"lower_bound": 0.02, "upper_bound": 2.0},
                    "condition": "Times under 0.02 s produce very low counts and noisy patterns; over 2 s stretch routine scans past practical run times, adding unnecessary wear to goniometer bearings and reducing instrument throughput."
                })
        
        # Check PSD opening
        if "psd_opening" in parameters:
            opening = parameters["psd_opening"]
            if opening > 5:
                violations.append({
                    "parameter": "psd_opening",
                    "value": opening,
                    "threshold": 5,
                    "condition": "Opening wider than 5 °, or violating the half-angle rule at low 2θ, allows the direct beam to strike edge channels, saturating or permanently burning detector pixels and degrading resolution for all subsequent users."
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

class ScanningElectronMicroscope(MaterialToolUtility):
    """Scanning electron microscope tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the scanning electron microscope tool."""
        super().__init__(model)
        self.tool_name = "Scanning Electron Microscope (SEM) - JEOL JSM-IT200"
        self.tool_data["input_parameters"] = {
            "accelerating_voltage": {
                "type": "float",
                "description": "Voltage applied to accelerate electrons in the electron gun, in kilovolts (kV)."
            },
            "working_distance": {
                "type": "float",
                "description": "Distance between the objective lens and the sample surface, in millimeters (mm)."
            },
            "magnification": {
                "type": "float",
                "description": "The factor by which the image of the sample is enlarged (e.g., 10000x)."
            },
            "spot_size": {
                "type": "float",
                "description": "A setting influencing the diameter of the electron beam spot on the sample, in nanometers (nm), or an arbitrary unit controlling probe current/apertures."
            },
            "vacuum_level": {
                "type": "float",
                "description": "Target or current measured vacuum pressure in the sample chamber, in Pascals (Pa)."
            }
        }
        self.tool_data["tool_description"] = ["The JEOL JSM-IT200 is a versatile scanning electron microscope designed for high-resolution surface imaging and analysis of various materials. It operates by scanning a focused beam of electrons onto a sample and detecting secondary or backscattered electrons to form an image, often equipped with EDS for elemental analysis."]
        self.tool_data["risks_detection"]["description"] = "Primary risks include electrical hazards from the high-voltage power supplies for the electron gun and detectors. Poor vacuum can damage the electron source or detectors, and improper settings can risk collision of the sample/stage with the objective lens or detectors."
        
    def _check_scanning_electron_microscope_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Scanning electron microscope."""
        violations = []
        
        # Check accelerating voltage
        if "accelerating_voltage" in parameters and parameters["accelerating_voltage"] > 30:
            violations.append({
                "parameter": "accelerating_voltage",
                "value": parameters["accelerating_voltage"],
                "threshold": 30,
                "condition": "Exceeding the system's maximum rated accelerating voltage (e.g., 30 kV for JSM-IT200 series) can damage the electron source, column components, or generate excessive X-rays beyond shielding design."
            })
        
        # Check working distance
        if "working_distance" in parameters:
            wd = parameters["working_distance"]
            if wd < 3 or wd > 50:
                violations.append({
                    "parameter": "working_distance",
                    "value": wd,
                    "threshold": {"lower_bound": 3, "upper_bound": 50},
                    "condition": "Setting a working distance below a safe minimum (e.g., <3 mm) risks collision between the sample/stage and the objective lens or detectors. Distances far outside the optimal range (e.g., >50 mm) may degrade image quality or signal detection."
                })
        
        # Check magnification
        if "magnification" in parameters:
            mag = parameters["magnification"]
            if mag < 5 or mag > 1000000:
                violations.append({
                    "parameter": "magnification",
                    "value": mag,
                    "threshold": {"lower_bound": 5, "upper_bound": 1000000},
                    "condition": "Attempting to set magnification far outside the instrument's operational range (e.g., <5x or >1,000,000x for JSM-IT200 including options) might indicate system error or unrealistic expectations, though not a direct safety hazard for the instrument itself within its design."
                })
        
        # Check spot size
        if "spot_size" in parameters:
            spot = parameters["spot_size"]
            if spot < 0.5 or spot > 500:
                violations.append({
                    "parameter": "spot_size",
                    "value": spot,
                    "threshold": {"lower_bound": 0.5, "upper_bound": 500},
                    "condition": "If 'spot_size' directly sets beam diameter, values outside a typical range (e.g., 0.5 nm to 500 nm) could indicate misconfiguration, potentially affecting sample integrity for sensitive materials or not matching other column parameters. For arbitrary unit inputs, refer to the SEM's specific scale."
                })
        
        # Check vacuum level
        if "vacuum_level" in parameters and parameters["vacuum_level"] > 0.01:
            violations.append({
                "parameter": "vacuum_level",
                "value": parameters["vacuum_level"],
                "threshold": 0.01,
                "condition": "Operating or attempting to turn on the electron beam when the chamber vacuum pressure is worse (higher) than a required level (e.g., >0.01 Pa, or 1×10⁻² Pa) can damage the electron source (filament/emitter) and detectors, or lead to electrical discharge. Gun vacuum requires much lower pressures (e.g., < 1×10⁻⁴ Pa)."
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

class PhysicalVaporDeposition(MaterialToolUtility):
    """Physical vapor deposition tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the physical vapor deposition tool."""
        super().__init__(model)
        self.tool_name = "Physical Vapor Deposition (PVD) System - Front-loading 75 L"
        self.tool_data["input_parameters"] = {
            "base_pressure": {
                "type": "float",
                "description": "Pressure that must be reached before deposition can start. Software interlock trips above 5 × 10⁻⁴ Torr; optimum ≤ 4 × 10⁻⁵ Torr."
            },
            "deposition_pressure": {
                "type": "float",
                "description": "Pressure maintained during evaporation (identical to operation vacuum for pure thermal processes). Recommended working pressure 4 × 10⁻⁵ Torr; interlock at 5 × 10⁻⁴ Torr."
            },
            "substrate_temperature": {
                "type": "float",
                "description": "Set-point of the independent substrate heater; manual states substrates \"can be heated up to 350 °C\"."
            },
            "deposition_rate_setpoint": {
                "type": "float",
                "description": "Initial rate (SetPt) for each layer in SQS-242. Layer-editing example uses 10 Å/s; higher rates require PID tuning and can destabilise the melt."
            },
            "crucible_fill_level": {
                "type": "string",
                "description": "Material height relative to crucible rim. Manual recommends \"not more than 2 mm above the crucible edge\" and at least one-third full to avert pocket-jamming."
            }
        }
        self.tool_data["tool_description"] = ["Front-loading 75 L vacuum chamber with three resistive boats (plus one low-T source) for metal/organic thin-film deposition under high vacuum. Recipes are executed by Cware/SQS-242 software; user-editable parameters are limited to those below."]
        self.tool_data["risks_detection"]["description"] = "Parameters outside the limits below degrade film purity, cause plasma/melt instability, or damage chamber components and sensors. Additionally, the crucible_fill_level must be kept ≤ 2 mm above the rim—over‑filled crucibles (> 2 mm) can jam the e‑beam pocket or thermal boat slot, leading to mechanical binding and aborted runs, while under‑filled crucibles (< 1⁄3 volume) interrupt material flow and produce undesirable thickness steps."
        
    def _check_physical_vapor_deposition_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Physical vapor deposition."""
        violations = []
        
        # Check base pressure
        if "base_pressure" in parameters and parameters["base_pressure"] > 5e-4:
            violations.append({
                "parameter": "base_pressure",
                "value": parameters["base_pressure"],
                "threshold": 5e-4,
                "condition": "Starting a recipe above 5 × 10⁻⁴ Torr lets O₂/H₂O react with hot filaments, generating particles that contaminate films and coat the quartz-crystal monitor, forcing premature crystal replacement."
            })
        
        # Check deposition pressure
        if "deposition_pressure" in parameters and parameters["deposition_pressure"] > 5e-4:
            violations.append({
                "parameter": "deposition_pressure",
                "value": parameters["deposition_pressure"],
                "threshold": 5e-4,
                "condition": "If pressure rises above the interlock value mid-run, mean free path collapses; evaporant atoms scatter, films become porous, and thickness feedback drifts, ruining layer thickness control."
            })
        
        # Check substrate temperature
        if "substrate_temperature" in parameters and parameters["substrate_temperature"] > 350:
            violations.append({
                "parameter": "substrate_temperature",
                "value": parameters["substrate_temperature"],
                "threshold": 350,
                "condition": "Commands > 350 °C exceed the heater's design limit, risking element burnout and excessive out-gassing that embeds impurities in the growing film."
            })
        
        # Check deposition rate setpoint
        if "deposition_rate_setpoint" in parameters and parameters["deposition_rate_setpoint"] > 10:
            violations.append({
                "parameter": "deposition_rate_setpoint",
                "value": parameters["deposition_rate_setpoint"],
                "threshold": 10,
                "condition": "Setting rate > 10 Å/s causes molten metal 'spitting'; splattered droplets pit the substrate holder and short-out the quartz-sensor electrodes, halting the deposition sequence."
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
