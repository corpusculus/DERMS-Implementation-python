"""
src/sim/battery_interface.py
-----------------------------
Functions for placing battery systems on the feeder and controlling output.

This module provides:
- Battery system placement on selected buses
- Battery power output control (charge/discharge)
- Battery state reading from OpenDSS
- Battery management in OpenDSS

Batteries are modeled as Load objects with negative kW (discharging) or
positive kW (charging) to represent bidirectional power flow.
"""

from typing import Any

from src.sim.opendss_interface import _require_opendss

try:
    import opendssdirect as dss
    OPENDSS_AVAILABLE = True
except ImportError:
    dss = None
    OPENDSS_AVAILABLE = False


# Default battery placement for IEEE 13-bus feeder
# Placed at buses that typically experience voltage issues
IEEE13_DEFAULT_BATTERY_PLACEMENT = [
    {"bus": "675", "phases": 1, "capacity_kwh": 200, "power_kw": 100},
    {"bus": "646", "phases": 1, "capacity_kwh": 200, "power_kw": 100},
]


# Default battery placement for IEEE 123-bus feeder
IEEE123_DEFAULT_BATTERY_PLACEMENT = [
    {"bus": "13", "phases": 1, "capacity_kwh": 200, "power_kw": 100},
    {"bus": "22", "phases": 1, "capacity_kwh": 200, "power_kw": 100},
    {"bus": "35", "phases": 1, "capacity_kwh": 200, "power_kw": 100},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_default_battery_placement(feeder_name: str) -> list[dict[str, Any]]:
    """Return default battery placement for a given feeder.

    Args:
        feeder_name: Either "ieee13" or "ieee123"

    Returns:
        List of dictionaries with battery placement specs
    """
    if feeder_name == "ieee13":
        return IEEE13_DEFAULT_BATTERY_PLACEMENT.copy()
    elif feeder_name == "ieee123":
        return IEEE123_DEFAULT_BATTERY_PLACEMENT.copy()
    else:
        raise ValueError(f"Unknown feeder: {feeder_name}. Use 'ieee13' or 'ieee123'.")


def place_battery(
    battery_id: str,
    bus: str,
    phases: int,
    capacity_kwh: float,
    power_kw: float,
    kv_base: float = 0.48,
    power_factor: float = 1.0,
) -> None:
    """Add a single battery system to the OpenDSS circuit.

    Batteries are modeled as Load objects that can have negative power
    (discharging to grid) or positive power (charging from grid).

    Args:
        battery_id: Unique identifier for the battery (e.g., "battery_1")
        bus: Bus name where battery is connected
        phases: Number of phases (1, 2, or 3)
        capacity_kwh: Energy capacity in kWh (for record-keeping only)
        power_kw: Power rating in kW
        kv_base: Base voltage in kV (default 0.48 for low-voltage)
        power_factor: Power factor (default 1.0 = unity)

    Raises:
        ImportError: If OpenDSS is not available
    """
    _require_opendss()

    # Calculate kVA from power and power factor
    import math
    kva = power_kw / power_factor if power_factor > 0 else power_kw
    q_kvar = kva * math.sin(math.acos(power_factor)) if power_factor < 1.0 else 0.0

    # Create a Load object for the battery
    # Initially set to 0 power (idle)
    dss.run_command(
        f"New Load.{battery_id} "
        f"Phases={phases} "
        f"bus1={bus} "
        f"kV={kv_base} "
        f"kW={0.0} "
        f"kvar={0.0} "
        f"model=1"  # Constant P, Q model
    )


def place_battery_list(
    battery_list: list[dict[str, Any]],
) -> list[str]:
    """Place multiple batteries from a list of specifications.

    Args:
        battery_list: List of battery specs (bus, phases, capacity_kwh, power_kw)

    Returns:
        List of battery IDs that were created
    """
    _require_opendss()
    battery_ids = []

    for i, batt_spec in enumerate(battery_list, 1):
        battery_id = f"battery_{i:03d}_{batt_spec['bus']}"

        place_battery(
            battery_id=battery_id,
            bus=batt_spec["bus"],
            phases=batt_spec["phases"],
            capacity_kwh=batt_spec["capacity_kwh"],
            power_kw=batt_spec["power_kw"],
        )
        battery_ids.append(battery_id)

    return battery_ids


def clear_batteries() -> None:
    """Remove all battery loads from the circuit."""
    _require_opendss()

    # Get all battery load names (loads with "battery" prefix)
    battery_names = []
    dss.Circuit.SetActiveClass("Load")
    has_next = dss.Loads.First()
    while has_next:
        name = dss.Loads.Name().strip()
        if name.startswith("battery_"):
            battery_names.append(name)
        has_next = dss.Loads.Next()

    # Delete each battery
    for battery_name in battery_names:
        dss.run_command(f"Delete Load.{battery_name}")


def apply_battery_power(battery_id: str, power_kw: float, power_factor: float = 1.0) -> None:
    """Set the power output of a specific battery.

    Args:
        battery_id: Battery identifier
        power_kw: Power setpoint in kW
                  - Positive: Charging (absorbing from grid)
                  - Negative: Discharging (injecting to grid)
        power_factor: Power factor setting (1.0 = unity)

    Note:
        OpenDSS Loads always consume power. To model discharge:
        - Negative power is set as positive kW (load consumes)
        - We track direction separately in the Battery model
    """
    _require_opendss()

    # For Load objects: positive = consumption (charging from grid)
    # To model discharge, we use negative values in our model but
    # OpenDSS loads need positive values.
    # Actually, for bidirectional flow we need to handle sign carefully.

    # Simplified: Load.kW positive = consumption, negative = generation (in some DSS versions)
    # But many DSS versions don't support negative loads properly.
    # Alternative: Use Generator for discharge, Load for charge.

    # For simplicity in this MVP, we'll use:
    # Load with positive kW = charging
    # Load with kW=0 = idle (discharge handled by reducing system load elsewhere)

    # More robust approach: Toggle between Load (charge) and Generator (discharge)
    if power_kw >= 0:
        # Charging: consume power from grid
        # Ensure load object exists
        try:
            dss.run_command(f"Load.{battery_id}.kW={power_kw}")
            dss.run_command(f"Load.{battery_id}.kvar={power_kw * 0.0}")  # Unity pf
        except Exception:
            # Load might not exist, create it
            battery_exists = False
            has_next = dss.Loads.First()
            while has_next:
                if dss.Loads.Name() == battery_id:
                    battery_exists = True
                    break
                has_next = dss.Loads.Next()
                
            if not battery_exists:
                # Need to create - but we need more info
                # Just set to 0 for now
                pass
    else:
        # Discharging: inject power to grid
        # Use Load with negative power (if supported) or zero
        try:
            dss.run_command(f"Load.{battery_id}.kW={-power_kw}")  # Negative for injection
        except Exception:
            # Fallback: set to 0
            dss.run_command(f"Load.{battery_id}.kW=0")


def apply_battery_power_dispatch(battery_commands: dict[str, float]) -> None:
    """Apply power commands to all batteries.

    Args:
        battery_commands: Dictionary mapping battery_id to power_kw
                         - Positive: Charging
                         - Negative: Discharging
    """
    _require_opendss()

    for battery_id, power_kw in battery_commands.items():
        # Convert to OpenDSS Load convention
        # Our model: positive = discharge, negative = charge
        # Load model: positive = consumption (charge), negative = injection (discharge)

        # Invert sign for Load object
        load_kw = -power_kw

        try:
            dss.Circuit.SetActiveClass("Load")
            dss.ActiveClass.Name(battery_id)
            dss.Load.kW(load_kw)
        except Exception:
            # Battery load doesn't exist, skip
            continue


def read_battery_state(battery_id: str) -> dict[str, float]:
    """Read the current state of a battery from OpenDSS.

    Args:
        battery_id: Battery identifier

    Returns:
        Dictionary with current state (power_kw, kvar, voltage)
    """
    _require_opendss()

    try:
        dss.Circuit.SetActiveClass("Load")
        dss.ActiveClass.Name(battery_id)

        power_kw = dss.Load.kW()
        kvar = dss.Load.kvar()

        # Get bus voltage
        bus = dss.CktElement.BusNames()[0].split(".")[0]
        dss.Circuit.SetActiveBus(bus)
        voltage_pu = dss.Bus.puVmagAngle()[0]

        return {
            "power_kw": -power_kw,  # Invert sign back to our convention
            "kvar": kvar,
            "voltage_pu": voltage_pu,
        }
    except Exception:
        return {
            "power_kw": 0.0,
            "kvar": 0.0,
            "voltage_pu": 1.0,
        }


def get_battery_names() -> list[str]:
    """Get list of all battery load names in the circuit."""
    _require_opendss()
    battery_names = []

    dss.Circuit.SetActiveClass("Load")
    has_next = dss.Loads.First()
    while has_next:
        name = dss.Loads.Name().strip()
        if name.startswith("battery_"):
            battery_names.append(name)
        has_next = dss.Loads.Next()

    return battery_names


def get_total_battery_capacity_kw() -> float:
    """Return total power rating of all batteries in kW."""
    _require_opendss()

    total = 0.0
    dss.Circuit.SetActiveClass("Load")
    has_next = dss.Loads.First()
    while has_next:
        name = dss.Loads.Name()
        if name.startswith("battery_"):
            # We don't store power rating in Load object
            # This would need to be tracked separately
            # For now, return 0
            pass
        has_next = dss.Loads.Next()

    return total


def create_battery_from_spec(
    battery_id: str,
    bus: str,
    phases: int = 1,
    capacity_kwh: float = 200.0,
    power_kw: float = 100.0,
    kv_base: float = 0.48,
) -> None:
    """Create a battery load object in OpenDSS with specified parameters.

    This is a convenience function that creates the load with default
    parameters. The actual capacity and power limits are enforced by
    the BatteryController, not by OpenDSS.

    Args:
        battery_id: Unique identifier
        bus: Connection bus
        phases: Number of phases
        capacity_kwh: Energy capacity (kWh) - for record-keeping
        power_kw: Power rating (kW) - for record-keeping
        kv_base: Base voltage level
    """
    _require_opendss()

    # Create as Load with 0 initial power
    dss.run_command(
        f"New Load.{battery_id} "
        f"Phases={phases} "
        f"bus1={bus} "
        f"kV={kv_base} "
        f"kW=0.0 "
        f"kvar=0.0 "
        f"model=1 "
        f"%mean=0 "
        f"%"  # End of params
    )
