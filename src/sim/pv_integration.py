"""
src/sim/pv_integration.py
-------------------------
Functions for placing PV systems on the feeder and scaling PV capacity.

This module provides:
- PV system placement on selected buses
- PV scaling/sweep functionality to find overvoltage threshold
- PV system management in OpenDSS
"""

import math
import pathlib
from typing import Any

import pandas as pd

try:
    import opendssdirect as dss
    OPENDSS_AVAILABLE = True
except ImportError:
    dss = None
    OPENDSS_AVAILABLE = False

from src.sim.opendss_interface import _require_opendss


_PV_RATED_PMMP: dict[str, float] = {}


# Default PV placement for IEEE 13-bus feeder
# Based on typical downstream buses that experience voltage rise
IEEE13_DEFAULT_PV_PLACEMENT = [
    {"bus": "675", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "646", "phases": 1, "p_kw": 80, "kva": 100},
    {"bus": "632", "phases": 1, "p_kw": 150, "kva": 180},
    {"bus": "633", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "634", "phases": 1, "p_kw": 120, "kva": 144},
    {"bus": "645", "phases": 1, "p_kw": 80, "kva": 100},
    {"bus": "611", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "652", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "680", "phases": 1, "p_kw": 80, "kva": 100},
    {"bus": "684", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "671", "phases": 3, "p_kw": 200, "kva": 240},
    {"bus": "692", "phases": 1, "p_kw": 80, "kva": 100},
]


# Default PV placement for IEEE 123-bus feeder
IEEE123_DEFAULT_PV_PLACEMENT = [
    {"bus": "13", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "18", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "22", "phases": 1, "p_kw": 150, "kva": 180},
    {"bus": "25", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "35", "phases": 1, "p_kw": 120, "kva": 144},
    {"bus": "40", "phases": 1, "p_kw": 80, "kva": 100},
    {"bus": "45", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "50", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "54", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "57", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "60", "phases": 1, "p_kw": 80, "kva": 100},
    {"bus": "64", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "67", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "71", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "75", "phases": 1, "p_kw": 80, "kva": 100},
    {"bus": "76", "phases": 2, "p_kw": 150, "kva": 180},
    {"bus": "80", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "83", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "88", "phases": 1, "p_kw": 100, "kva": 120},
    {"bus": "90", "phases": 1, "p_kw": 80, "kva": 100},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_default_pv_placement(feeder_name: str) -> list[dict[str, Any]]:
    """Return default PV placement for a given feeder.

    Args:
        feeder_name: Either "ieee13" or "ieee123"

    Returns:
        List of dictionaries with PV placement specs (bus, phases, p_kw, kva)
    """
    if feeder_name == "ieee13":
        return IEEE13_DEFAULT_PV_PLACEMENT.copy()
    elif feeder_name == "ieee123":
        return IEEE123_DEFAULT_PV_PLACEMENT.copy()
    else:
        raise ValueError(f"Unknown feeder: {feeder_name}. Use 'ieee13' or 'ieee123'.")


def place_pv_system(
    pv_id: str,
    bus: str,
    phases: int,
    p_kw: float,
    kva: float,
    pf: float = 1.0,
) -> None:
    """Add a single PV system to the OpenDSS circuit.

    Args:
        pv_id: Unique identifier for the PV system (e.g., "pv_1")
        bus: Bus name where PV is connected
        phases: Number of phases (1, 2, or 3)
        p_kw: Active power rating in kW
        kva: Apparent power rating in kVA
        pf: Power factor (default 1.0 = unity)

    Raises:
        ImportError: If OpenDSS is not available
    """
    _require_opendss()

    bus_spec, kv_base = _resolve_pv_connection(bus, phases)

    # Create the PV system
    dss.run_command(f"New PVSystem.{pv_id} "
                   f"Phases={phases} "
                   f"bus1={bus_spec} "
                   f"kV={kv_base} "
                   f"kVA={kva} "
                   f"Pmpp={p_kw} "
                   f"Irradiance=1.0 "
                   f"pf={pf} "
                   f"%Cutout=0.05 "
                   f"%Pmin=0 "
                   f"Vminpu=0.9 "
                   f"Vmaxpu=1.1")

    # Set to unity power factor for baseline
    dss.run_command(f"PVSystem.{pv_id}.pf=1.0")
    _PV_RATED_PMMP[pv_id] = p_kw


def place_pv_list(
    pv_list: list[dict[str, Any]],
    scale_factor: float = 1.0,
) -> list[str]:
    """Place multiple PV systems from a list of specifications.

    Args:
        pv_list: List of PV specs (bus, phases, p_kw, kva)
        scale_factor: Multiplier for P and kVA ratings

    Returns:
        List of PV IDs that were created
    """
    _require_opendss()
    pv_ids = []

    for i, pv_spec in enumerate(pv_list, 1):
        p_kw = pv_spec["p_kw"] * scale_factor
        kva = pv_spec["kva"] * scale_factor
        pv_id = f"pv_{i:03d}_{pv_spec['bus']}"

        place_pv_system(
            pv_id=pv_id,
            bus=pv_spec["bus"],
            phases=pv_spec["phases"],
            p_kw=p_kw,
            kva=kva,
        )
        pv_ids.append(pv_id)

    return pv_ids


def clear_pv_systems() -> None:
    """Remove all PV systems from the circuit."""
    _require_opendss()

    # Get all PV system names
    pv_names = []
    has_next = dss.PVsystems.First()
    while has_next:
        pv_names.append(dss.PVsystems.Name().strip())
        has_next = dss.PVsystems.Next()

    # Delete each PV system
    for pv_name in pv_names:
        if pv_name:
            dss.run_command(f"Delete PVSystem.{pv_name}")

    _PV_RATED_PMMP.clear()


def set_pv_output(pv_id: str, p_available_kw: float, power_factor: float = 1.0) -> None:
    """Set the output of a specific PV system.

    Args:
        pv_id: PV system identifier
        p_available_kw: Available active power in kW
        power_factor: Power factor setting (1.0 = unity, <1.0 = absorbing vars)
    """
    _require_opendss()

    rated_pmpp = _PV_RATED_PMMP.get(pv_id)
    if rated_pmpp is None:
        if not dss.Circuit.SetActiveElement(f"PVSystem.{pv_id}"):
            return
        rated_pmpp = dss.PVsystems.Pmpp()
        _PV_RATED_PMMP[pv_id] = rated_pmpp

    irradiance = 0.0 if rated_pmpp <= 0 else max(0.0, p_available_kw / rated_pmpp)
    dss.run_command(f"PVSystem.{pv_id}.Pmpp={rated_pmpp}")
    dss.run_command(f"PVSystem.{pv_id}.Irradiance={irradiance}")
    dss.run_command(f"PVSystem.{pv_id}.pf={power_factor}")


def set_all_pv_output(p_available_kw: float, power_factor: float = 1.0) -> None:
    """Set all PV systems to the same output level.

    Args:
        p_available_kw: Available active power in kW
        power_factor: Power factor setting
    """
    _require_opendss()

    total_rated_kw = sum(_PV_RATED_PMMP.values())

    has_next = dss.PVsystems.First()
    while has_next:
        pv_name = dss.PVsystems.Name()
        rated_kw = _PV_RATED_PMMP.get(pv_name, dss.PVsystems.Pmpp())
        share = 0.0 if total_rated_kw <= 0 else rated_kw / total_rated_kw
        set_pv_output(pv_name, p_available_kw * share, power_factor)
        has_next = dss.PVsystems.Next()


def get_pv_names() -> list[str]:
    """Get list of all PV system names in the circuit."""
    _require_opendss()
    pv_names = []

    has_next = dss.PVsystems.First()
    while has_next:
        pv_names.append(dss.PVsystems.Name().strip())
        has_next = dss.PVsystems.Next()

    return pv_names


def get_total_pv_capacity_kw() -> float:
    """Return total nameplate PV capacity in kW."""
    _require_opendss()

    total = 0.0
    has_next = dss.PVsystems.First()
    while has_next:
        total += dss.PVsystems.Pmpp()
        has_next = dss.PVsystems.Next()

    return total


def _resolve_pv_connection(bus: str, phases: int) -> tuple[str, float]:
    """Resolve an existing bus into a valid OpenDSS bus spec and kV."""
    if not dss.Circuit.SetActiveBus(bus):
        raise ValueError(f"Bus not found for PV placement: {bus}")

    nodes = [int(node) for node in dss.Bus.Nodes() if int(node) > 0]
    kv_base_ln = dss.Bus.kVBase()
    if not nodes or kv_base_ln <= 0:
        raise ValueError(f"Bus {bus} has no usable nodes or voltage base")

    if phases == 1:
        connection_nodes = [nodes[0]]
        kv = kv_base_ln
    elif phases == 2:
        if len(nodes) < 2:
            raise ValueError(f"Bus {bus} does not support a 2-phase PV connection")
        connection_nodes = nodes[:2]
        kv = kv_base_ln * math.sqrt(3)
    elif phases == 3:
        if len(nodes) < 3:
            raise ValueError(f"Bus {bus} does not support a 3-phase PV connection")
        connection_nodes = nodes[:3]
        kv = kv_base_ln * math.sqrt(3)
    else:
        raise ValueError(f"Unsupported PV phase count: {phases}")

    bus_spec = f"{bus}." + ".".join(str(node) for node in connection_nodes)
    return bus_spec, kv


def find_overvoltage_threshold(
    base_pv_list: list[dict[str, Any]],
    voltage_limit: float = 1.05,
    scale_min: float = 0.5,
    scale_max: float = 5.0,
    scale_step: float = 0.5,
    load_multiplier: float = 0.5,
    pv_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Sweep PV capacity to find the level where overvoltage begins.

    This is useful for determining a "problem scenario" PV level for
    demonstrating DERMS value.

    Args:
        base_pv_list: Base PV placement list
        voltage_limit: Voltage threshold (default 1.05 pu)
        scale_min: Starting scale factor
        scale_max: Maximum scale factor
        scale_step: Increment for scale factor
        load_multiplier: Load level during test (lower = more stress)
        pv_multiplier: PV production level (1.0 = full production)

    Returns:
        Dict with results: scale_factors, max_voltages, overvoltage_threshold
    """
    _require_opendss()

    results = {
        "scale_factors": [],
        "max_voltages": [],
        "min_voltages": [],
        "overvoltage_found": False,
        "threshold_scale": None,
    }

    from src.sim.opendss_interface import solve_power_flow, get_bus_voltages

    scale = scale_min
    while scale <= scale_max:
        # Clear and place PV at current scale
        clear_pv_systems()
        place_pv_list(base_pv_list, scale_factor=scale)

        # Set PV output level
        total_capacity = get_total_pv_capacity_kw()
        set_all_pv_output(total_capacity * pv_multiplier, 1.0)

        # Solve and check voltages
        try:
            solve_power_flow()
            voltages = get_bus_voltages()
            max_v = max(voltages.values())
            min_v = min(voltages.values())

            results["scale_factors"].append(scale)
            results["max_voltages"].append(max_v)
            results["min_voltages"].append(min_v)

            if max_v > voltage_limit and not results["overvoltage_found"]:
                results["overvoltage_found"] = True
                results["threshold_scale"] = scale

        except RuntimeError:
            # Solve failed - probably too much PV
            break

        scale += scale_step

    # Clean up
    clear_pv_systems()

    return results


def export_pv_placement_csv(
    pv_list: list[dict[str, Any]],
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """Export PV placement list to CSV for documentation/editing.

    Args:
        pv_list: List of PV placement specs
        output_path: Where to write the CSV

    Returns:
        Resolved path to written file
    """
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(pv_list)
    df.to_csv(out, index=False)
    return out
