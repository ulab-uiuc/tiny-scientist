import json
import math
from typing import Any, Dict, List, Optional, Union

from ..tool import BaseTool
from ..utils.llm import create_client, get_response_from_llm


class KinematicsTool(BaseTool):
    """Tool for calculating kinematic quantities in classical mechanics."""
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the requested calculation is safe to perform."""
        # Extract relevant values to check
        velocity = inputs.get("velocity", 0)
        acceleration = inputs.get("acceleration", 0)
        time = inputs.get("time", 0)
        mass = inputs.get("mass", 0)
        
        # Check for unreasonable values
        if isinstance(velocity, (int, float)) and abs(velocity) > 3e8:  # Speed of light
            return {
                "safe": False,
                "reason": f"Velocity {velocity} exceeds the speed of light (3e8 m/s)",
                "allowed": False
            }
            
        if isinstance(time, (int, float)) and time < 0:
            return {
                "safe": False,
                "reason": f"Negative time value {time} is not physically meaningful",
                "allowed": False
            }
            
        if isinstance(mass, (int, float)) and mass < 0:
            return {
                "safe": False,
                "reason": f"Negative mass value {mass} is not physically meaningful",
                "allowed": False
            }
        
        # Check for extremely large values that might cause numerical issues
        for name, value in inputs.items():
            if isinstance(value, (int, float)) and abs(value) > 1e20:
                return {
                    "safe": False,
                    "reason": f"Value for {name} ({value}) is unreasonably large",
                    "allowed": False
                }
        
        # Default to allowing if no issues detected
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self, 
        calculation_type: str,
        initial_velocity: Optional[float] = None,
        final_velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        time: Optional[float] = None,
        displacement: Optional[float] = None,
        mass: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate kinematic quantities based on the provided parameters.
        
        Args:
            calculation_type: Type of calculation to perform
                ("motion", "force", "energy", "momentum", etc.)
            initial_velocity: Initial velocity in m/s
            final_velocity: Final velocity in m/s
            acceleration: Acceleration in m/s²
            time: Time in seconds
            displacement: Displacement in meters
            mass: Mass in kg (for force, energy, momentum calculations)
            
        Returns:
            Dictionary with calculated results
        """
        # Check safety first
        inputs = {
            "velocity": max(abs(initial_velocity or 0), abs(final_velocity or 0)),
            "acceleration": acceleration or 0,
            "time": time or 0,
            "displacement": displacement or 0,
            "mass": mass or 0
        }
        safety_result = self.safety_detect(inputs)
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        result = {"success": True, "calculation_type": calculation_type}
        
        # Uniform motion calculations
        if calculation_type.lower() in ["motion", "uniform motion", "kinematics"]:
            # Kinematic equations:
            # v = u + at
            # s = ut + 0.5at²
            # v² = u² + 2as
            # s = 0.5(u+v)t
            
            # Calculate any missing value based on available inputs
            if initial_velocity is not None and acceleration is not None and time is not None:
                final_vel = initial_velocity + acceleration * time
                result["final_velocity"] = final_vel
                
                displ = initial_velocity * time + 0.5 * acceleration * time**2
                result["displacement"] = displ
                
            elif initial_velocity is not None and final_velocity is not None and time is not None:
                accel = (final_velocity - initial_velocity) / time
                result["acceleration"] = accel
                
                displ = 0.5 * (initial_velocity + final_velocity) * time
                result["displacement"] = displ
                
            elif initial_velocity is not None and final_velocity is not None and acceleration is not None:
                if acceleration != 0:
                    time_val = (final_velocity - initial_velocity) / acceleration
                    result["time"] = time_val
                    
                    displ = (final_velocity**2 - initial_velocity**2) / (2 * acceleration)
                    result["displacement"] = displ
                else:
                    # Constant velocity case
                    result["time"] = "Infinite (constant velocity with zero acceleration)"
                    result["note"] = "Cannot determine displacement without time"
                    
            elif initial_velocity is not None and acceleration is not None and displacement is not None:
                discriminant = initial_velocity**2 + 2 * acceleration * displacement
                if discriminant >= 0:
                    final_vel = math.sqrt(discriminant)
                    result["final_velocity"] = final_vel
                    
                    time_val = (final_vel - initial_velocity) / acceleration if acceleration != 0 else displacement / initial_velocity
                    result["time"] = time_val
                else:
                    result["success"] = False
                    result["error"] = "Invalid parameters: displacement not achievable with given initial velocity and acceleration"
                    
            else:
                result["success"] = False
                result["error"] = "Insufficient parameters provided. Need at least 3 of: initial_velocity, final_velocity, acceleration, time, displacement"
        
        # Force calculations (F = ma)
        elif calculation_type.lower() in ["force", "newton's second law"]:
            if mass is not None and acceleration is not None:
                force = mass * acceleration
                result["force"] = force
                result["unit"] = "newtons (N)"
            elif mass is not None and force is not None:
                acceleration_val = force / mass
                result["acceleration"] = acceleration_val
                result["unit"] = "m/s²"
            elif force is not None and acceleration is not None:
                mass_val = force / acceleration
                result["mass"] = mass_val
                result["unit"] = "kg"
            else:
                result["success"] = False
                result["error"] = "Insufficient parameters for force calculation. Need at least 2 of: mass, force, acceleration"
        
        # Energy calculations
        elif calculation_type.lower() in ["energy", "kinetic energy", "potential energy"]:
            if "kinetic" in calculation_type.lower() and mass is not None:
                if final_velocity is not None:
                    kinetic_energy = 0.5 * mass * final_velocity**2
                    result["kinetic_energy"] = kinetic_energy
                    result["unit"] = "joules (J)"
                elif initial_velocity is not None:
                    kinetic_energy = 0.5 * mass * initial_velocity**2
                    result["kinetic_energy"] = kinetic_energy
                    result["unit"] = "joules (J)"
                    result["note"] = "Calculated using initial velocity as final velocity was not provided"
                else:
                    result["success"] = False
                    result["error"] = "Velocity is required for kinetic energy calculation"
            
            elif "potential" in calculation_type.lower() and mass is not None:
                if displacement is not None:  # Treating displacement as height for potential energy
                    gravity = 9.8  # m/s², acceleration due to gravity on Earth
                    potential_energy = mass * gravity * displacement
                    result["potential_energy"] = potential_energy
                    result["unit"] = "joules (J)"
                else:
                    result["success"] = False
                    result["error"] = "Height (displacement) is required for potential energy calculation"
            else:
                result["success"] = False
                result["error"] = "Insufficient parameters for energy calculation"
        
        # Momentum calculations (p = mv)
        elif calculation_type.lower() in ["momentum", "linear momentum"]:
            if mass is not None:
                if final_velocity is not None:
                    momentum = mass * final_velocity
                    result["momentum"] = momentum
                    result["unit"] = "kg·m/s"
                elif initial_velocity is not None:
                    momentum = mass * initial_velocity
                    result["momentum"] = momentum
                    result["unit"] = "kg·m/s"
                    result["note"] = "Calculated using initial velocity as final velocity was not provided"
                else:
                    result["success"] = False
                    result["error"] = "Velocity is required for momentum calculation"
            else:
                result["success"] = False
                result["error"] = "Mass is required for momentum calculation"
        
        else:
            result["success"] = False
            result["error"] = f"Unknown calculation type: {calculation_type}"
        
        # Add input parameters to the result
        result["parameters"] = {
            "initial_velocity": initial_velocity,
            "final_velocity": final_velocity,
            "acceleration": acceleration,
            "time": time,
            "displacement": displacement,
            "mass": mass
        }
        
        return result


class ElectricalTool(BaseTool):
    """Tool for calculating electrical quantities using Ohm's Law and related formulas."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for simulations."""
        self.client, self.model = create_client(model)
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the requested electrical calculation is safe."""
        # Extract relevant values to check
        voltage = inputs.get("voltage", 0)
        current = inputs.get("current", 0)
        
        # Check for dangerous levels of electrical parameters
        if isinstance(voltage, (int, float)) and voltage > 50:  # Generally, voltage > 50V is considered dangerous
            return {
                "safe": True,  # Still safe for calculation, just add a warning
                "reason": f"WARNING: Voltage of {voltage}V is potentially dangerous to humans",
                "warning": "High voltage",
                "allowed": True
            }
            
        if isinstance(current, (int, float)) and current > 0.1:  # Current > 100mA can be lethal
            return {
                "safe": True,  # Still safe for calculation, just add a warning
                "reason": f"WARNING: Current of {current}A is potentially dangerous to humans",
                "warning": "High current",
                "allowed": True
            }
        
        # Check for negative resistance which is not physically meaningful
        resistance = inputs.get("resistance", 0)
        if isinstance(resistance, (int, float)) and resistance < 0:
            return {
                "safe": False,
                "reason": f"Negative resistance value {resistance} is not physically meaningful",
                "allowed": False
            }
        
        # Default to allowing if no issues detected
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self, 
        calculation_type: str,
        voltage: Optional[float] = None,
        current: Optional[float] = None,
        resistance: Optional[float] = None,
        power: Optional[float] = None,
        time: Optional[float] = None,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate electrical quantities based on the provided parameters.
        
        Args:
            calculation_type: Type of calculation to perform
                ("ohm", "power", "energy", "circuit", etc.)
            voltage: Voltage in volts (V)
            current: Current in amperes (A)
            resistance: Resistance in ohms (Ω)
            power: Power in watts (W)
            time: Time in seconds (for energy calculations)
            component: Description of electrical component (for additional context)
            
        Returns:
            Dictionary with calculated results
        """
        # Check safety first
        inputs = {
            "voltage": voltage or 0,
            "current": current or 0,
            "resistance": resistance or 0,
            "power": power or 0,
            "time": time or 0
        }
        safety_result = self.safety_detect(inputs)
        
        result = {"success": True, "calculation_type": calculation_type}
        
        # Add safety warnings if present
        if safety_result.get("warning"):
            result["warning"] = safety_result.get("warning")
            result["safety_note"] = safety_result.get("reason")
        
        # Check if calculation is allowed to proceed
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Ohm's Law calculations (V = IR)
        if calculation_type.lower() in ["ohm", "ohm's law", "resistance"]:
            if voltage is not None and current is not None and current != 0:
                resistance_val = voltage / current
                result["resistance"] = resistance_val
                result["unit"] = "ohms (Ω)"
                
            elif voltage is not None and resistance is not None and resistance != 0:
                current_val = voltage / resistance
                result["current"] = current_val
                result["unit"] = "amperes (A)"
                
            elif current is not None and resistance is not None:
                voltage_val = current * resistance
                result["voltage"] = voltage_val
                result["unit"] = "volts (V)"
                
            else:
                result["success"] = False
                result["error"] = "Insufficient parameters for Ohm's Law calculation. Need at least 2 of: voltage, current, resistance"
        
        # Power calculations (P = VI, P = I²R, P = V²/R)
        elif calculation_type.lower() in ["power", "electrical power"]:
            if voltage is not None and current is not None:
                power_val = voltage * current
                result["power"] = power_val
                result["unit"] = "watts (W)"
                
            elif current is not None and resistance is not None:
                power_val = current**2 * resistance
                result["power"] = power_val
                result["unit"] = "watts (W)"
                
            elif voltage is not None and resistance is not None and resistance != 0:
                power_val = voltage**2 / resistance
                result["power"] = power_val
                result["unit"] = "watts (W)"
                
            elif power is not None:
                if voltage is not None and voltage != 0:
                    current_val = power / voltage
                    result["current"] = current_val
                    result["unit_current"] = "amperes (A)"
                    
                    resistance_val = voltage**2 / power
                    result["resistance"] = resistance_val
                    result["unit_resistance"] = "ohms (Ω)"
                    
                elif current is not None and current != 0:
                    voltage_val = power / current
                    result["voltage"] = voltage_val
                    result["unit_voltage"] = "volts (V)"
                    
                    resistance_val = power / current**2
                    result["resistance"] = resistance_val
                    result["unit_resistance"] = "ohms (Ω)"
                    
                elif resistance is not None:
                    current_val = math.sqrt(power / resistance)
                    result["current"] = current_val
                    result["unit_current"] = "amperes (A)"
                    
                    voltage_val = math.sqrt(power * resistance)
                    result["voltage"] = voltage_val
                    result["unit_voltage"] = "volts (V)"
                    
                else:
                    result["success"] = False
                    result["error"] = "Insufficient parameters for power calculation. Need power and one of: voltage, current, resistance"
            else:
                result["success"] = False
                result["error"] = "Insufficient parameters for power calculation. Need at least 2 of: voltage, current, resistance, or power and 1 other parameter"
        
        # Energy calculations (E = Pt)
        elif calculation_type.lower() in ["energy", "electrical energy"]:
            if power is not None and time is not None:
                energy = power * time
                result["energy"] = energy
                result["unit"] = "joules (J)"
                
                # Convert to kilowatt-hours if appropriate
                if energy > 1000:
                    energy_kwh = energy / 3600000  # 1 kWh = 3,600,000 J
                    result["energy_kwh"] = energy_kwh
                    result["unit_kwh"] = "kilowatt-hours (kWh)"
            else:
                result["success"] = False
                result["error"] = "Both power and time are required for energy calculation"
        
        # Circuit analysis (use LLM for more complex cases)
        elif calculation_type.lower() in ["circuit", "circuit analysis"]:
            if component:
                prompt = f"""
                Perform circuit analysis for the following component/circuit:
                Component/Circuit: {component}
                
                Provided parameters:
                - Voltage: {voltage if voltage is not None else 'Not provided'}
                - Current: {current if current is not None else 'Not provided'}
                - Resistance: {resistance if resistance is not None else 'Not provided'}
                - Power: {power if power is not None else 'Not provided'}
                - Time: {time if time is not None else 'Not provided'}
                
                Analyze the circuit and provide calculations for unknown parameters.
                Format your response as JSON with the calculated values and explanations.
                """
                
                response, _ = get_response_from_llm(
                    msg=prompt,
                    client=self.client,
                    model=self.model,
                    system_message="You are an electrical engineering expert performing accurate circuit analysis.",
                    temperature=0.2
                )
                
                try:
                    # Extract JSON from response
                    circuit_analysis = json.loads(response)
                    # Add metadata
                    circuit_analysis["success"] = True
                    circuit_analysis["component"] = component
                    return circuit_analysis
                except json.JSONDecodeError:
                    # Fallback to returning the raw text if JSON parsing fails
                    return {
                        "success": True,
                        "component": component,
                        "analysis": response,
                        "parameters": inputs,
                        "note": "Response formatting failed, returning raw analysis"
                    }
            else:
                result["success"] = False
                result["error"] = "Component description is required for circuit analysis"
        
        else:
            result["success"] = False
            result["error"] = f"Unknown calculation type: {calculation_type}"
        
        # Add input parameters to the result
        result["parameters"] = {
            "voltage": voltage,
            "current": current,
            "resistance": resistance,
            "power": power,
            "time": time,
            "component": component
        }
        
        return result


class ThermodynamicsTool(BaseTool):
    """Tool for thermodynamic calculations and simulations."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the tool with a language model for simulations."""
        self.client, self.model = create_client(model)
    
    def safety_detect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the requested thermodynamic calculation is safe."""
        # Extract relevant values to check
        temperature = inputs.get("temperature", 0)
        pressure = inputs.get("pressure", 0)
        
        # Check for extreme temperature values
        if isinstance(temperature, (int, float)):
            # Check for absolute zero violation
            if temperature < 0 and inputs.get("temperature_unit", "K").upper() == "K":
                return {
                    "safe": False,
                    "reason": f"Temperature {temperature}K is below absolute zero, which is physically impossible",
                    "allowed": False
                }
            
            # Check for extremely high temperatures
            if temperature > 1e8:  # Arbitrarily high temperature
                return {
                    "safe": True,  # Still calculate, but warn
                    "reason": f"Temperature {temperature} is extremely high",
                    "warning": "Extreme temperature",
                    "allowed": True
                }
        
        # Check for negative pressure (vacuum can only go to 0 Pa absolute)
        if isinstance(pressure, (int, float)) and pressure < 0:
            return {
                "safe": False,
                "reason": f"Negative pressure {pressure} is not physically meaningful",
                "allowed": False
            }
        
        # Default to allowing if no issues detected
        return {"safe": True, "reason": "No safety concerns detected", "allowed": True}
    
    def run(
        self, 
        calculation_type: str,
        temperature: Optional[float] = None,
        temperature_unit: str = "K",  # K, C, or F
        pressure: Optional[float] = None,
        pressure_unit: str = "Pa",  # Pa, bar, atm
        volume: Optional[float] = None,
        volume_unit: str = "m3",  # m3, L
        mass: Optional[float] = None,
        substance: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        final_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate thermodynamic quantities based on the provided parameters.
        
        Args:
            calculation_type: Type of calculation to perform
                ("gas law", "heat", "entropy", "process", etc.)
            temperature: Temperature value
            temperature_unit: Unit of temperature (K, C, F)
            pressure: Pressure value
            pressure_unit: Unit of pressure (Pa, bar, atm)
            volume: Volume value
            volume_unit: Unit of volume (m3, L)
            mass: Mass in kg
            substance: Name of substance for specific calculations
            initial_state: Dictionary of initial state parameters
            final_state: Dictionary of final state parameters
            
        Returns:
            Dictionary with calculated results
        """
        # Standardize units for calculations
        temp_k = temperature
        if temperature is not None:
            if temperature_unit.upper() == "C":
                temp_k = temperature + 273.15
            elif temperature_unit.upper() == "F":
                temp_k = (temperature - 32) * 5/9 + 273.15
        
        pres_pa = pressure
        if pressure is not None:
            if pressure_unit.lower() == "bar":
                pres_pa = pressure * 100000
            elif pressure_unit.lower() == "atm":
                pres_pa = pressure * 101325
        
        vol_m3 = volume
        if volume is not None:
            if volume_unit.lower() == "l":
                vol_m3 = volume / 1000
        
        # Check safety first with standardized units
        inputs = {
            "temperature": temp_k,
            "temperature_unit": "K",
            "pressure": pres_pa,
            "pressure_unit": "Pa",
            "volume": vol_m3,
            "volume_unit": "m3",
            "mass": mass or 0,
            "substance": substance or ""
        }
        safety_result = self.safety_detect(inputs)
        
        result = {"success": True, "calculation_type": calculation_type}
        
        # Add safety warnings if present
        if safety_result.get("warning"):
            result["warning"] = safety_result.get("warning")
            result["safety_note"] = safety_result.get("reason")
        
        # Check if calculation is allowed to proceed
        if not safety_result.get("allowed", True):
            return {
                "success": False,
                "error": "Safety check failed",
                "details": safety_result
            }
        
        # Ideal Gas Law calculations (PV = nRT)
        if calculation_type.lower() in ["gas law", "ideal gas law"]:
            # Universal gas constant
            R = 8.3145  # J/(mol·K)
            
            # If mass and substance are provided, we can calculate moles
            moles = None
            if mass is not None and substance is not None:
                # Use LLM to get molecular weight if needed
                prompt = f"What is the molecular weight of {substance} in g/mol? Just return the numerical value."
                response, _ = get_response_from_llm(
                    msg=prompt,
                    client=self.client,
                    model=self.model,
                    system_message="You are a chemistry expert providing accurate molecular weights.",
                    temperature=0.1
                )
                
                try:
                    # Try to extract a number from the response
                    import re
                    molecular_weight_match = re.search(r"(\d+\.?\d*|\.\d+)", response)
                    if molecular_weight_match:
                        molecular_weight = float(molecular_weight_match.group(1))
                        moles = mass * 1000 / molecular_weight  # Convert mass from kg to g
                        result["moles"] = moles
                        result["molecular_weight"] = molecular_weight
                except Exception:
                    result["note"] = "Could not calculate moles from mass and substance"
            
            # Calculations based on available parameters
            if pres_pa is not None and vol_m3 is not None and temp_k is not None:
                if moles is None:
                    # Calculate moles using PV = nRT
                    moles = (pres_pa * vol_m3) / (R * temp_k)
                    result["moles"] = moles
                
                # Also provide the value of PV/T as a check
                pv_t = pres_pa * vol_m3 / temp_k
                result["PV/T"] = pv_t
                result["nR"] = moles * R
                
            elif pres_pa is not None and vol_m3 is not None and moles is not None:
                # Calculate temperature
                temp_k = (pres_pa * vol_m3) / (moles * R)
                result["temperature"] = temp_k
                result["temperature_celsius"] = temp_k - 273.15
                
            elif pres_pa is not None and temp_k is not None and moles is not None:
                # Calculate volume
                vol_m3 = (moles * R * temp_k) / pres_pa
                result["volume"] = vol_m3
                result["volume_liters"] = vol_m3 * 1000
                
            elif vol_m3 is not None and temp_k is not None and moles is not None:
                # Calculate pressure
                pres_pa = (moles * R * temp_k) / vol_m3
                result["pressure"] = pres_pa
                result["pressure_bar"] = pres_pa / 100000
                result["pressure_atm"] = pres_pa / 101325
                
            else:
                result["success"] = False
                result["error"] = "Insufficient parameters for gas law calculation. Need at least 3 of: pressure, volume, temperature, and moles (or mass and substance)"
        
        # Heat and work calculations
        elif calculation_type.lower() in ["heat", "work", "energy transfer"]:
            if initial_state and final_state:
                # For more complex cases or when full states are provided, use LLM
                prompt = f"""
                Calculate the heat and work for a thermodynamic process with:
                
                Initial state: {json.dumps(initial_state)}
                Final state: {json.dumps(final_state)}
                Substance: {substance or 'Unknown'}
                
                Format your response as JSON with heat, work, and explanation fields.
                Use standard SI units, and include the sign convention (+ for system gaining heat/work).
                """
                
                response, _ = get_response_from_llm(
                    msg=prompt,
                    client=self.client,
                    model=self.model,
                    system_message="You are a thermodynamics expert calculating accurate heat and work values.",
                    temperature=0.2
                )
                
                try:
                    # Extract JSON from response
                    calculation_result = json.loads(response)
                    # Add metadata
                    calculation_result["success"] = True
                    calculation_result["substance"] = substance
                    calculation_result["initial_state"] = initial_state
                    calculation_result["final_state"] = final_state
                    return calculation_result
                except json.JSONDecodeError:
                    # Fallback to returning the raw text if JSON parsing fails
                    return {
                        "success": True,
                        "substance": substance,
                        "analysis": response,
                        "initial_state": initial_state,
                        "final_state": final_state,
                        "note": "Response formatting failed, returning raw analysis"
                    }
            else:
                # Handle simpler calculations or provide guidance
                result["success"] = False
                result["error"] = "Insufficient parameters for heat/work calculation. Need initial and final states."
                result["suggestion"] = "For a heat or work calculation, provide initial_state and final_state dictionaries with temperature, pressure, and volume information."
        
        # For other calculation types, use LLM
        else:
            # Prepare a prompt based on the calculation type and available parameters
            param_description = ", ".join([f"{k}: {v}" for k, v in inputs.items() if v])
            prompt = f"""
            Perform a {calculation_type} calculation with the following parameters:
            {param_description}
            
            Format your response as JSON with calculated values and explanations.
            Use proper SI units and provide unit conversions where helpful.
            """
            
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message="You are a thermodynamics expert providing accurate calculations.",
                temperature=0.2
            )
            
            try:
                # Extract JSON from response
                calculation_result = json.loads(response)
                # Add metadata
                calculation_result["success"] = True
                calculation_result["calculation_type"] = calculation_type
                calculation_result["parameters"] = inputs
                return calculation_result
            except json.JSONDecodeError:
                # Fallback to returning the raw text if JSON parsing fails
                return {
                    "success": True,
                    "calculation_type": calculation_type,
                    "analysis": response,
                    "parameters": inputs,
                    "note": "Response formatting failed, returning raw analysis"
                }
        
        # Add input parameters to the result (with original units)
        result["parameters"] = {
            "temperature": temperature,
            "temperature_unit": temperature_unit,
            "pressure": pressure,
            "pressure_unit": pressure_unit,
            "volume": volume,
            "volume_unit": volume_unit,
            "mass": mass,
            "substance": substance
        }
        
        return result 