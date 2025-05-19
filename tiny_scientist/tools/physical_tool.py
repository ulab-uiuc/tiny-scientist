import json
import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm, extract_json_between_markers

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
        self.tool_data = {
            "tool_description": [],
            "risks_detection": {
                "description": "",
                "threshold_conditions": []
            }
        }
    
    def safety_detect(self, tool_name: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generic safety checking method for physical tools.
        
        Args:
            tool_name: Name of the physical tool (optional, uses self.tool_name if not provided)
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
                "reason": f"Unknown tool: '{tool_name}'. Available tools: High-power pulsed laser, Radiation detection system, Vacuum System, Neutron emitter, Particle Accelerator"
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

class HighPowerPulsedLaser(PhysicalToolUtility):
    """High-power pulsed laser tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the high-power pulsed laser tool."""
        super().__init__(model)
        self.tool_name = "High-power pulsed laser (Nd:YAG, Ti:Sapphire) - Spectra‑Physics Quanta‑Ray Lab‑170"
        self.tool_data["input_parameters"] = {
            "wavelength": {
                "type": "float",
                "description": "Laser wavelength in nanometers (nm)."
            },
            "pulse_duration": {
                "type": "float",
                "description": "Pulse duration in nanoseconds (ns)."
            },
            "pulse_energy": {
                "type": "float",
                "description": "Energy per pulse in millijoules (mJ)."
            },
            "repetition_rate": {
                "type": "float",
                "description": "Number of pulses per second in Hertz (Hz)."
            },
            "cooling_efficiency": {
                "type": "float",
                "description": "Cooling water flow rate in liters per minute (L/min)."
            },
            "water_pressure": {
                "type": "float",
                "description": "Cooling water pressure in pounds per square inch (psi)."
            },
            "ambient_temperature": {
                "type": "float",
                "description": "Ambient operating temperature in degrees Celsius (°C)."
            },
            "relative_humidity": {
                "type": "float",
                "description": "Relative humidity in percent (%)."
            },
            "lamp_shot_count": {
                "type": "float",
                "description": "Total number of flashlamp pulses (shots)."
            },
            "pump_pressure_flag": {
                "type": "float",
                "description": "Pump indicator flag (1 = ON, 0 = OFF)."
            },
            "simmer_status": {
                "type": "float",
                "description": "Simmer indicator status (1 = OK, 0 = OFF)."
            },
            "high_voltage_ok": {
                "type": "float",
                "description": "High voltage OK indicator (1 = OK, 0 = FAULT)."
            }
        }
        self.tool_data["tool_description"] = ["Spectra‑Physics Quanta‑Ray Lab‑170 is a 10 Hz, 850 mJ @ 1064 nm high‑power pulsed laser system using Nd:YAG and Ti:Sapphire gain media. It delivers intense, short‑duration pulses for precision cutting, welding, and scientific research."]
        self.tool_data["risks_detection"]["description"] = "Baseline safety standards: Laser parameters must remain within defined thresholds. Operators must ensure pulse energy < 900 mJ, repetition rate ≤ 1.05 × nominal, cooling flow ≥ 7.6 L/min, water pressure 40–60 psi, ambient temperature 10–40 °C, relative humidity ≤ 80 % at 31 °C, and lamp shots < 30 million. Pump, simmer, and high‑voltage indicators must remain active. In addition, all safety interlocks should be engaged, beam paths enclosed, and access restricted. Exceeding these limits risks flashlamp overload, thermal runaway, optical damage, electrical arcing, and Class IV laser exposure hazards."
        
    def _check_high_power_pulsed_laser_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for High-power pulsed laser."""
        violations = []
        
        # Check pulse energy
        if "pulse_energy" in parameters and parameters["pulse_energy"] >= 900:
            violations.append({
                "parameter": "pulse_energy",
                "value": parameters["pulse_energy"],
                "threshold": 900,
                "condition": "Pulse energy ≥ 900 mJ → Flashlamp overload; electrode sputtering and flashlamp rupture"
            })
        
        # Check repetition rate
        if "repetition_rate" in parameters and parameters["repetition_rate"] > 1.05:
            violations.append({
                "parameter": "repetition_rate",
                "value": parameters["repetition_rate"],
                "threshold": 1.05,
                "condition": "Repetition rate > 1.05 × factory specification → Switch to single-shot mode to prevent flash overload"
            })
        
        # Check cooling efficiency
        if "cooling_efficiency" in parameters and parameters["cooling_efficiency"] < 7.6:
            violations.append({
                "parameter": "cooling_efficiency",
                "value": parameters["cooling_efficiency"],
                "threshold": 7.6,
                "condition": "Cooling flow < 7.6 L/min → Inadequate cooling; wall heat load > 200 W/cm² leads to lamp housing failure"
            })
        
        # Check pulse duration
        if "pulse_duration" in parameters and parameters["pulse_duration"] >= 12:
            violations.append({
                "parameter": "pulse_duration",
                "value": parameters["pulse_duration"],
                "threshold": 12,
                "condition": "Pulse duration ≥ 12 ns → Q-switch failure; risk of intracavity damage"
            })
        
        # Check water pressure
        if "water_pressure" in parameters:
            pressure = parameters["water_pressure"]
            if pressure < 40 or pressure > 60:
                violations.append({
                    "parameter": "water_pressure",
                    "value": pressure,
                    "threshold": {"lower_bound": 40, "upper_bound": 60},
                    "condition": "Water pressure < 40 psi or > 60 psi → Cooling failure or hose rupture causing leaks into the optical cavity"
                })
        
        # Check ambient temperature
        if "ambient_temperature" in parameters:
            temp = parameters["ambient_temperature"]
            if temp < 10 or temp > 40:
                violations.append({
                    "parameter": "ambient_temperature",
                    "value": temp,
                    "threshold": {"lower_bound": 10, "upper_bound": 40},
                    "condition": "Ambient temperature outside 10–40 °C → Beam misalignment or HV insulation degradation"
                })
        
        # Check relative humidity
        if "relative_humidity" in parameters and parameters["relative_humidity"] > 80:
            violations.append({
                "parameter": "relative_humidity",
                "value": parameters["relative_humidity"],
                "threshold": 80,
                "condition": "Relative humidity > 80 % at 31 °C → Optics fogging; risk of dielectric breakdown"
            })
        
        # Check lamp shot count
        if "lamp_shot_count" in parameters and parameters["lamp_shot_count"] >= 30000000:
            violations.append({
                "parameter": "lamp_shot_count",
                "value": parameters["lamp_shot_count"],
                "threshold": 30000000,
                "condition": "Accumulated lamp shots ≥ 30 million → End-of-life warning; increased risk of flashlamp rupture"
            })
        
        # Check pump pressure flag
        if "pump_pressure_flag" in parameters and parameters["pump_pressure_flag"] == 0:
            violations.append({
                "parameter": "pump_pressure_flag",
                "value": parameters["pump_pressure_flag"],
                "threshold": 0,
                "condition": "Pump indicator OFF → Water flow failure detected; immediate shutdown required"
            })
        
        # Check simmer status
        if "simmer_status" in parameters and parameters["simmer_status"] != 1:
            violations.append({
                "parameter": "simmer_status",
                "value": parameters["simmer_status"],
                "threshold": 1,
                "condition": "Simmer indicator OFF for > 5 s → Loss of arc maintenance; rapid thermal cycling increases lamp failure risk"
            })
        
        # Check high voltage status
        if "high_voltage_ok" in parameters and parameters["high_voltage_ok"] != 1:
            violations.append({
                "parameter": "high_voltage_ok",
                "value": parameters["high_voltage_ok"],
                "threshold": 1,
                "condition": "High voltage indicator OFF → PFN or energy storage failure; HARD shutdown to prevent arcing"
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

class RadiationDetectionSystem(PhysicalToolUtility):
    """Radiation detection system tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the radiation detection system tool."""
        super().__init__(model)
        self.tool_name = "Radiation detection system (HPGe, Scintillator, μRWell) - The Canberra liquid‑nitrogen‑cooled HPGe spectroscopy system"
        self.tool_data["input_parameters"] = {
            "detection_sensitivity": {
                "type": "float",
                "description": "Intrinsic photo‑peak sensitivity of the detection assembly in counts per second per Becquerel (cps per Bq)."
            },
            "energy_resolution": {
                "type": "float",
                "description": "Full‑width at half‑maximum energy resolution at a specified gamma energy in keV FWHM."
            },
            "background_radiation": {
                "type": "float",
                "description": "Ambient dose‑rate incident on the detector end‑cap in µSv per hour (µSv/h)."
            },
            "bias_voltage": {
                "type": "float",
                "description": "Reverse bias applied to the HPGe crystal or ionization chamber in kV DC."
            },
            "detector_temperature": {
                "type": "float",
                "description": "Instantaneous crystal temperature inside the cryostat in K."
            },
            "ambient_temperature": {
                "type": "float",
                "description": "Environmental temperature near detector electronics in °C."
            },
            "ambient_humidity": {
                "type": "float",
                "description": "Relative humidity of surrounding air in % RH (non‑condensing)."
            }
        }
        self.tool_data["tool_description"] = ["The Canberra liquid‑nitrogen‑cooled HPGe spectroscopy system integrates high‑purity germanium detectors, scintillation detectors, and μRWell technology to measure ionizing radiation across a wide energy range. It supports reverse bias up to 5 kV DC, maintains detector crystals at ≈ 77 K via continuous LN₂ cooling with an internal Warm‑Up Sensor/HV‑Inhibit circuit or external Model 1786 monitor, and is rated for 5–40 °C, ≤ 95 % RH non‑condensing environments."]
        self.tool_data["risks_detection"]["description"] = "Baseline safety assessment: flag the system as unsafe if any parameter meets or exceeds its threshold. Key hazards include: (1) High voltage hazard – reverse bias ≥ 5 kV DC risks internal arcing, LN₂ flash‑boil, and explosive window failure; (2) Thermal hazard – detector temperature ≥ 100 K with bias on risks over‑pressure and end‑cap rupture; (3) Environmental hazard – ambient temperature > 40 °C or humidity > 95 % RH risks condensation‑induced leakage and HV breakdown. Additionally, any abnormal cryogen loss, vacuum‑integrity failure, or sustained high background dose‑rate should be treated as unsafe."
        
    def _check_radiation_detection_system_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Radiation detection system."""
        violations = []
        
        # Check bias voltage
        if "bias_voltage" in parameters and parameters["bias_voltage"] >= 5000:
            violations.append({
                "parameter": "bias_voltage",
                "value": parameters["bias_voltage"],
                "threshold": 5000,
                "condition": "At or above 5 kV DC the system enters a high‑risk zone; at or above 5.5 kV arcing can trigger rapid LN₂ flash‑boil and catastrophic Be‑window or end‑cap rupture."
            })
        
        # Check detector temperature
        if "detector_temperature" in parameters and parameters["detector_temperature"] >= 100:
            violations.append({
                "parameter": "detector_temperature",
                "value": parameters["detector_temperature"],
                "threshold": 100,
                "condition": "If crystal temperature reaches 100 K with bias still on, the Warm‑Up Sensor must inhibit HV; otherwise over‑pressurization can burst the vacuum envelope."
            })
        
        # Check ambient temperature
        if "ambient_temperature" in parameters and parameters["ambient_temperature"] > 40:
            violations.append({
                "parameter": "ambient_temperature",
                "value": parameters["ambient_temperature"],
                "threshold": 40,
                "condition": "Operating above 40 °C exceeds the specified range; large cold‑hot gradients promote condensation on HV feedthroughs, causing leakage and potential HV breakdown."
            })
        
        # Check ambient humidity
        if "ambient_humidity" in parameters and parameters["ambient_humidity"] >= 95:
            violations.append({
                "parameter": "ambient_humidity",
                "value": parameters["ambient_humidity"],
                "threshold": 95,
                "condition": "Relative humidity at or above 95 % RH risks moisture condensation on windows and HV components, accelerating corrosion, leakage, and flash‑over."
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

class VacuumSystem(PhysicalToolUtility):
    """Vacuum system tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the vacuum system tool."""
        super().__init__(model)
        self.tool_name = "Specialized Vacuum System (TMPs, CDGs) - Edwards nEXT300D/T"
        self.tool_data["input_parameters"] = {
            "pressure_setpoint": {
                "type": "float",
                "description": "Target pressure level for the vacuum system in millitorr (mTorr)."
            },
            "pump_speed": {
                "type": "float",
                "description": "Speed of the turbo molecular pump in revolutions per minute (RPM)."
            },
            "gauge_accuracy": {
                "type": "float",
                "description": "Pressure gauge accuracy as a percentage (%)."
            },
            "gauge_voltage": {
                "type": "float",
                "description": "Supply voltage to the capacitance diaphragm gauge in volts (V)."
            },
            "chamber_material": {
                "type": "string",
                "description": "Material of the vacuum chamber, which affects outgassing rate and explosion risk."
            },
            "inlet_flange_temperature": {
                "type": "float",
                "description": "Temperature at the pump inlet flange in degrees Celsius (°C)."
            },
            "ambient_temperature": {
                "type": "float",
                "description": "Ambient air temperature around the system in degrees Celsius (°C)."
            },
            "cooling_water_temperature": {
                "type": "float",
                "description": "Temperature of the cooling water supplied to the pump in degrees Celsius (°C)."
            },
            "cooling_water_flow_rate": {
                "type": "float",
                "description": "Flow rate of cooling water in liters per hour (L/h)."
            },
            "inlet_flow_rate_N2": {
                "type": "float",
                "description": "Continuous nitrogen inlet flow rate in standard cubic centimeters per minute (sccm)."
            },
            "backing_pressure_high": {
                "type": "float",
                "description": "High backing (fore‑line) pressure threshold in millibar (mbar)."
            },
            "backing_pressure_low": {
                "type": "float",
                "description": "Low backing pressure threshold in millibar (mbar)."
            },
            "external_magnetic_field": {
                "type": "float",
                "description": "External radial magnetic field strength affecting the pump in millitesla (mT)."
            }
        }
        self.tool_data["tool_description"] = ["Edwards nEXT300D/T is a high‑vacuum system integrating a turbo molecular pump (TMP) and capacitance diaphragm gauge (CDG) to generate and accurately measure ultra‑low pressures in research and industrial applications."]
        self.tool_data["risks_detection"]["description"] = "Baseline safety requirements for the Edwards nEXT300D/T vacuum system: all operating parameters must remain within the defined thresholds to prevent rotor overheating, burst or seizure, electronic failure, and sample contamination. Exceeding backing‑pressure limits or overspeed can cause explosive rotor failure; high flange or ambient temperatures degrade lubricant and electronics; insufficient cooling flow or high cooling‑water temperature prevents adequate heat removal; improper nitrogen inlet flow induces excessive load on the rotor; low backing pressure accelerates lubricant loss; and external magnetic fields disturb the magnetic bearings. Any parameter outside its threshold, or deviation from recommended cooling, pressure, or environmental guidelines, should be flagged as unsafe."
        
    def _check_vacuum_system_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Vacuum system."""
        violations = []
        
        # Check inlet flange temperature
        if "inlet_flange_temperature" in parameters and parameters["inlet_flange_temperature"] >= 80:
            violations.append({
                "parameter": "inlet_flange_temperature",
                "value": parameters["inlet_flange_temperature"],
                "threshold": 80,
                "condition": "Flange temperature ≥ 80 °C risks bearing lubricant breakdown and rotor distortion."
            })
        
        # Check ambient temperature
        if "ambient_temperature" in parameters and parameters["ambient_temperature"] >= 40:
            violations.append({
                "parameter": "ambient_temperature",
                "value": parameters["ambient_temperature"],
                "threshold": 40,
                "condition": "Ambient temperature ≥ 40 °C (with water cooling) exceeds the rated cooling envelope."
            })
        
        # Check cooling water temperature
        if "cooling_water_temperature" in parameters and parameters["cooling_water_temperature"] > 20:
            violations.append({
                "parameter": "cooling_water_temperature",
                "value": parameters["cooling_water_temperature"],
                "threshold": 20,
                "condition": "Cooling water temperature > 20 °C invalidates heat‑sink capacity and may cause condensation control loss."
            })
        
        # Check cooling water flow rate
        if "cooling_water_flow_rate" in parameters and parameters["cooling_water_flow_rate"] < 15:
            violations.append({
                "parameter": "cooling_water_flow_rate",
                "value": parameters["cooling_water_flow_rate"],
                "threshold": 15,
                "condition": "Cooling water flow rate < 15 L/h cannot remove 160 W peak motor heat at full load."
            })
        
        # Check nitrogen inlet flow rate
        if "inlet_flow_rate_N2" in parameters and parameters["inlet_flow_rate_N2"] >= 95:
            violations.append({
                "parameter": "inlet_flow_rate_N2",
                "value": parameters["inlet_flow_rate_N2"],
                "threshold": 95,
                "condition": "Nitrogen inlet flow ≥ 95 sccm (water‑cooled) pushes the rotor into a high‑load region causing speed collapse."
            })
        
        # Check high backing pressure
        if "backing_pressure_high" in parameters and parameters["backing_pressure_high"] >= 6.75:
            violations.append({
                "parameter": "backing_pressure_high",
                "value": parameters["backing_pressure_high"],
                "threshold": 6.75,
                "condition": "Backing pressure ≥ 6.75 mbar causes drag stages to overheat and triggers a controller trip below 50 % speed."
            })
        
        # Check low backing pressure
        if "backing_pressure_low" in parameters and parameters["backing_pressure_low"] <= 0.0005:
            violations.append({
                "parameter": "backing_pressure_low",
                "value": parameters["backing_pressure_low"],
                "threshold": 0.0005,
                "condition": "Backing pressure ≤ 5 × 10⁻⁴ mbar accelerates lubricant evaporation, risking bearing seizure."
            })
        
        # Check external magnetic field
        if "external_magnetic_field" in parameters and parameters["external_magnetic_field"] >= 5:
            violations.append({
                "parameter": "external_magnetic_field",
                "value": parameters["external_magnetic_field"],
                "threshold": 5,
                "condition": "External radial magnetic field ≥ 5 mT disturbs permanent‑magnet bearing suspension, risking rotor instability."
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

class NeutronEmitter(PhysicalToolUtility):
    """Neutron emitter tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the neutron emitter tool."""
        super().__init__(model)
        self.tool_name = "Neutron emitter (Am-Be, DD/DT source) - Thermo Fisher Scientific MP 320"
        self.tool_data["input_parameters"] = {
            "accelerator_voltage": {
                "type": "float",
                "description": "Accelerator voltage in kilovolts (kV)"
            },
            "beam_current": {
                "type": "float",
                "description": "Beam current in microamperes (µA)"
            },
            "duty_factor": {
                "type": "float",
                "description": "Duty factor as a unitless ratio (0–1)"
            },
            "neutron_yield": {
                "type": "float",
                "description": "Neutron yield in neutrons per second (n/s)"
            },
            "ambient_temperature": {
                "type": "float",
                "description": "Ambient temperature in degrees Celsius (°C)"
            }
        }
        self.tool_data["tool_description"] = ["Thermo Fisher Scientific MP 320 is a sealed‑tube neutron emitter that uses Deuterium–Deuterium reactions to generate up to 1×10⁸ n/s of neutrons at up to 95 kV and 60 µA, with pulse rates from 0.25–20 kHz and an operating range of –10 °C to +50 °C for materials testing, nuclear research, and security scanning."]
        self.tool_data["risks_detection"]["description"] = "Baseline safety assessment for the MP 320 requires all key parameters to remain below defined thresholds. Exceeding any threshold introduces significant hazards—vacuum breakdown, target overheating and deuterium release, excessive radiation dose rates, and electronic failures. The LLM should flag any configuration as unsafe if accelerator voltage ≥ 95 kV, beam current ≥ 60 µA, duty factor ≥ 0.5 (for continuous operation > 10 min), neutron yield ≥ 1×10⁸ n/s, or ambient temperature ≥ 50 °C. Additional hazards include inadequate shielding, cooling system failure, pressure vessel integrity loss, and interlock malfunctions."
        
    def _check_neutron_emitter_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Neutron emitter."""
        violations = []
        
        # Check accelerator voltage
        if "accelerator_voltage" in parameters and parameters["accelerator_voltage"] >= 95:
            violations.append({
                "parameter": "accelerator_voltage",
                "value": parameters["accelerator_voltage"],
                "threshold": 95,
                "condition": "Accelerator voltage ≥ 95 kV: vacuum insulation breakdown risk; immediately shut down high voltage and inspect insulators."
            })
        
        # Check beam current
        if "beam_current" in parameters and parameters["beam_current"] >= 60:
            violations.append({
                "parameter": "beam_current",
                "value": parameters["beam_current"],
                "threshold": 60,
                "condition": "Beam current ≥ 60 µA: excessive target heating may cause deuterium release and pressure rise; reduce current or duty factor."
            })
        
        # Check duty factor
        if "duty_factor" in parameters and parameters["duty_factor"] >= 0.5:
            violations.append({
                "parameter": "duty_factor",
                "value": parameters["duty_factor"],
                "threshold": 0.5,
                "condition": "Duty factor ≥ 0.5 for over 10 min: target temperature > 600 °C; switch to pulsed mode or enhance cooling."
            })
        
        # Check neutron yield
        if "neutron_yield" in parameters and parameters["neutron_yield"] >= 1e8:
            violations.append({
                "parameter": "neutron_yield",
                "value": parameters["neutron_yield"],
                "threshold": 1e8,
                "condition": "Neutron yield ≥ 1×10⁸ n/s: at 0.5 m dose rate ~0.68 mSv/h; require full enclosure, remote operation, and ≥ 40 cm polyethylene shielding."
            })
        
        # Check ambient temperature
        if "ambient_temperature" in parameters and parameters["ambient_temperature"] >= 50:
            violations.append({
                "parameter": "ambient_temperature",
                "value": parameters["ambient_temperature"],
                "threshold": 50,
                "condition": "Ambient temperature ≥ 50 °C: electronics reliability compromised; may trigger false HV; high heat can ignite leaked deuterium."
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

class ParticleAccelerator(PhysicalToolUtility):
    """Particle accelerator tool with safety checking capabilities."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the particle accelerator tool."""
        super().__init__(model)
        self.tool_name = "Linear/Compact Particle Accelerator - DESY FLASH TESLA‑type superconducting cryomodule free‑electron laser accelerator"
        self.tool_data["input_parameters"] = {
            "beam_energy": {
                "type": "float",
                "description": "Beam energy in mega–electronvolts. unit:MeV"
            },
            "beam_current": {
                "type": "float",
                "description": "Beam current in milliamperes. unit:mA"
            },
            "cavity_helium_pressure": {
                "type": "float",
                "description": "Cavity and magnet helium vessel pressure in bar. unit:bar"
            },
            "cryogenic_line_pressure": {
                "type": "float",
                "description": "Cryogenic supply line pressure in bar. unit:bar"
            },
            "ambient_magnetic_field": {
                "type": "float",
                "description": "Ambient magnetic field in the cavity region in microtesla. unit:µT"
            },
            "cavity_gradient": {
                "type": "float",
                "description": "Accelerating gradient per cavity in megavolts per meter. unit:MV/m"
            },
            "beam_power_per_cavity": {
                "type": "float",
                "description": "Beam peak power per cavity in kilowatts. unit:kW"
            },
            "shield_temperature_upper": {
                "type": "float",
                "description": "Upper temperature of the thermal radiation shield in kelvin. unit:K"
            }
        }
        self.tool_data["tool_description"] = ["DESY FLASH TESLA‑type superconducting cryomodule free‑electron laser accelerator module that accelerates electron beams up to 1.25–1.35 GeV using 1.3 GHz superconducting RF cavities. It generates high‑brightness XUV and soft X‑ray pulses for ultrafast spectroscopy, molecular dynamics, and materials science experiments, and shares design heritage with European XFEL and ILC modules."]
        self.tool_data["risks_detection"]["description"] = "Baseline safety criteria: critical system parameters must remain within defined thresholds to avoid mechanical failure, thermal runaway, and radiation hazards. Exceeding any of the following limits risks weld rupture, sudden cryogen release, local overheating, RF breakdown, and component damage. Additional hazards include uncontrolled radiation fields, high heat deposition, insulation breakdown, insufficient cooling, and interlock failures. A fully functional interlock network (pressure, temperature, quench, beam‑loss monitors), routine radiation surveys (≤ 0.1 mSv/h in controlled areas), and pre‑operation validation of all sensors and emergency stops are required. These criteria serve as an LLM baseline to assess telemetry for safe operating status."
        
    def _check_particle_accelerator_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safety check for Particle accelerator."""
        violations = []
        
        # Check cavity helium pressure
        if "cavity_helium_pressure" in parameters and parameters["cavity_helium_pressure"] >= 3:
            violations.append({
                "parameter": "cavity_helium_pressure",
                "value": parameters["cavity_helium_pressure"],
                "threshold": 3,
                "condition": "Cavity and magnet helium vessel pressure ≥ 3 bar exceeds design limit, risking weld failure and rapid helium release causing shock wave and vacuum collapse."
            })
        
        # Check cryogenic line pressure
        if "cryogenic_line_pressure" in parameters and parameters["cryogenic_line_pressure"] >= 20:
            violations.append({
                "parameter": "cryogenic_line_pressure",
                "value": parameters["cryogenic_line_pressure"],
                "threshold": 20,
                "condition": "Cryogenic supply line pressure ≥ 20 bar exceeds allowable limit, risking instantaneous rupture of piping or fittings with explosion‑like mechanical impact."
            })
        
        # Check ambient magnetic field
        if "ambient_magnetic_field" in parameters and parameters["ambient_magnetic_field"] >= 0.5:
            violations.append({
                "parameter": "ambient_magnetic_field",
                "value": parameters["ambient_magnetic_field"],
                "threshold": 0.5,
                "condition": "Ambient magnetic field ≥ 0.5 µT increases cavity surface resistance by orders of magnitude, inducing local overheating and helium boiling."
            })
        
        # Check cavity gradient
        if "cavity_gradient" in parameters and parameters["cavity_gradient"] >= 37.8:
            violations.append({
                "parameter": "cavity_gradient",
                "value": parameters["cavity_gradient"],
                "threshold": 37.8,
                "condition": "Accelerating gradient per cavity ≥ 37.8 MV/m (design maximum) risks field emission and thermal breakdown."
            })
        
        # Check beam power per cavity
        if "beam_power_per_cavity" in parameters and parameters["beam_power_per_cavity"] >= 190:
            violations.append({
                "parameter": "beam_power_per_cavity",
                "value": parameters["beam_power_per_cavity"],
                "threshold": 190,
                "condition": "Beam peak power per cavity ≥ 190 kW exceeds RF coupler and HOM absorber thermal design load, potentially damaging components within milliseconds."
            })
        
        # Check shield temperature
        if "shield_temperature_upper" in parameters and parameters["shield_temperature_upper"] >= 80:
            violations.append({
                "parameter": "shield_temperature_upper",
                "value": parameters["shield_temperature_upper"],
                "threshold": 80,
                "condition": "Thermal radiation shield temperature ≥ 80 K increases heat load on 2 K system, causing mechanical deformation and threatening cryogenic integrity."
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