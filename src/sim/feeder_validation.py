"""
src/sim/feeder_validation.py
----------------------------
Functions for validating feeder models and identifying major components.

This module provides:
- Feeder component identification (head, regulators, capacitors, laterals)
- Circuit topology analysis
- Validation reports
"""

import pathlib
from typing import Any

try:
    import opendssdirect as dss
    OPENDSS_AVAILABLE = True
except ImportError:
    dss = None
    OPENDSS_AVAILABLE = False

from src.sim.opendss_interface import load_feeder, solve_power_flow, _require_opendss


# ---------------------------------------------------------------------------
# Component Identification
# ---------------------------------------------------------------------------


def get_feeder_head() -> dict[str, Any]:
    """Identify the feeder head (source bus) and substation transformer.

    Returns:
        Dict with source bus info and transformer details
    """
    _require_opendss()

    # Get the source bus (typically where the Vsource is connected)
    dss.Circuit.SetActiveClass("Vsource")
    if dss.ActiveClass.Count() > 0:
        dss.ActiveClass.Name(1)
        source_bus = dss.Properties.Value("bus1")
    else:
        source_bus = "unknown"

    has_next = dss.Transformers.First()
    while has_next:
        xfmrs.append({
            "name": dss.Transformers.Name(),
            "phases": dss.Transformers.NumPhases(),
            "kva": dss.Transformers.kVA(),
            "primary_bus": dss.Transformers.Wdg(1).Bus(),
            "secondary_bus": dss.Transformers.Wdg(2).Bus() if dss.Transformers.NumWindings() >= 2 else None,
        })
        has_next = dss.Transformers.Next()

    return {
        "source_bus": source_bus,
        "transformers": xfmrs,
    }


def get_regulators() -> list[dict[str, Any]]:
    """Get all voltage regulators in the circuit.

    Returns:
        List of regulator info dicts
    """
    _require_opendss()

    has_next = dss.RegControls.First()
    while has_next:
        reg = {
            "name": dss.RegControls.Name(),
            "transformer": dss.RegControls.Transformer(),
            "vreg": dss.RegControls.ForwardVreg(),
            "band": dss.RegControls.ForwardBand(),
            "ptratio": dss.RegControls.PTRatio(),
            "ctprim": dss.RegControls.CTPrimary(),
        }
        regulators.append(reg)
        has_next = dss.RegControls.Next()

    return regulators


def get_capacitors() -> list[dict[str, Any]]:
    """Get all capacitor banks in the circuit.

    Returns:
        List of capacitor info dicts
    """
    _require_opendss()

    has_next = dss.Capacitors.First()
    while has_next:
        cap = {
            "name": dss.Capacitors.Name(),
            "bus": dss.Properties.Value("bus1"),
            "kvar": dss.Capacitors.kvar(),
            "phases": dss.CktElement.NumPhases(),
            "state": dss.Capacitors.States(),
        }
        capacitors.append(cap)
        has_next = dss.Capacitors.Next()

    return capacitors


def get_lines() -> list[dict[str, Any]]:
    """Get all line segments in the circuit.

    Returns:
        List of line info dicts
    """
    _require_opendss()

    has_next = dss.Lines.First()
    while has_next:
        line = {
            "name": dss.Lines.Name(),
            "bus1": dss.Lines.Bus1(),
            "bus2": dss.Lines.Bus2(),
            "length": dss.Lines.Length(),
            "units": dss.Lines.Units(),
            "phases": dss.Lines.Phases(),
            "r1": dss.Lines.R1(),
            "x1": dss.Lines.X1(),
        }
        lines.append(line)
        has_next = dss.Lines.Next()

    return lines


def get_loads() -> list[dict[str, Any]]:
    """Get all loads in the circuit.

    Returns:
        List of load info dicts
    """
    _require_opendss()

    has_next = dss.Loads.First()
    while has_next:
        load = {
            "name": dss.Loads.Name(),
            "bus": dss.Properties.Value("bus1"),
            "phases": dss.Loads.Phases(),
            "kw": dss.Loads.kW(),
            "kvar": dss.Loads.kvar(),
            "connection": dss.Properties.Value("conn"),
        }
        loads.append(load)
        has_next = dss.Loads.Next()

    return loads


def get_buses() -> list[dict[str, Any]]:
    """Get all buses with voltage and phase info.

    Returns:
        List of bus info dicts
    """
    _require_opendss()

    buses = []

    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        pu_vmag_angle = dss.Bus.puVmagAngle()

        # Extract magnitudes (every other value starting from 0)
        pu_mags = pu_vmag_angle[::2]
        # Extract angles (every other value starting from 1)
        angles = pu_vmag_angle[1::2]

        # Determine phases present
        phases = dss.Bus.NumPhases()

        buses.append({
            "name": bus_name,
            "phases": phases,
            "pu_voltages": list(pu_mags),
            "mean_pu_voltage": float(sum(pu_mags) / len(pu_mags)) if pu_mags else 0.0,
            "angles": list(angles),
            "kVBase": dss.Bus.kVBase(),
        })

    return buses


def identify_laterals() -> dict[str, list[str]]:
    """Identify major laterals and their buses.

    A simple heuristic: group buses by their first numeric segment
    or by tracing from the main trunk.

    Returns:
        Dict mapping lateral name to list of bus names
    """
    _require_opendss()

    buses = dss.Circuit.AllBusNames()

    # Simple heuristic: group by shared prefix
    laterals = {}
    for bus in buses:
        # Extract prefix (e.g., "670" from "670.1.2.3")
        parts = bus.split(".")
        if len(parts) > 1:
            prefix = parts[0]
        else:
            prefix = "main"

        if prefix not in laterals:
            laterals[prefix] = []
        laterals[prefix].append(bus)

    return laterals


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def validate_feeder(dss_master_file: str | pathlib.Path) -> dict[str, Any]:
    """Run a complete validation of the feeder model.

    Args:
        dss_master_file: Path to the master DSS file

    Returns:
        Dict with validation results
    """
    print(f"\n{'='*60}")
    print(f"  Feeder Validation")
    print(f"{'='*60}")

    # Load the feeder
    print(f"\nLoading feeder: {dss_master_file}")
    load_feeder(dss_master_file)

    # Solve initial power flow
    print("Solving power flow...")
    try:
        solve_power_flow()
        converged = True
    except RuntimeError as e:
        print(f"Warning: Power flow did not converge: {e}")
        converged = False

    # Get circuit summary
    circuit = dss.Circuit

    results = {
        "name": circuit.Name(),
        "converged": converged,
        "num_buses": len(circuit.AllBusNames()),
        "num_elements": circuit.NumCktElements(),
    }

    # Get components
    print("\nIdentifying components...")

    head = get_feeder_head()
    results["head"] = head

    regulators = get_regulators()
    results["regulators"] = regulators
    print(f"  Regulators: {len(regulators)}")

    capacitors = get_capacitors()
    results["capacitors"] = capacitors
    print(f"  Capacitors: {len(capacitors)}")

    loads = get_loads()
    total_kw = sum(l["kw"] for l in loads)
    total_kvar = sum(l["kvar"] for l in loads)
    results["loads"] = loads
    results["total_load_kw"] = total_kw
    results["total_load_kvar"] = total_kvar
    print(f"  Loads: {len(loads)} ({total_kw:.1f} kW, {total_kvar:.1f} kVAR)")

    lines = get_lines()
    results["lines"] = lines
    print(f"  Lines: {len(lines)}")

    # Get buses and voltage info
    buses = get_buses()
    results["buses"] = buses

    if converged:
        v_mins = [b["mean_pu_voltage"] for b in buses]
        v_maxes = [b["mean_pu_voltage"] for b in buses]

        results["voltage_min"] = min(v_mins)
        results["voltage_max"] = max(v_maxes)
        results["voltage_mean"] = sum(v_mins) / len(v_mins)

        print(f"\nVoltage Statistics:")
        print(f"  Min: {results['voltage_min']:.4f} pu")
        print(f"  Max: {results['voltage_max']:.4f} pu")
        print(f"  Mean: {results['voltage_mean']:.4f} pu")

        # Check for violations
        overvoltage = [b for b in buses if b["mean_pu_voltage"] > 1.05]
        undervoltage = [b for b in buses if b["mean_pu_voltage"] < 0.95]

        results["overvoltage_buses"] = [b["name"] for b in overvoltage]
        results["undervoltage_buses"] = [b["name"] for b in undervoltage]

        if overvoltage:
            print(f"  ⚠ Overvoltage buses: {len(overvoltage)}")
        if undervoltage:
            print(f"  ⚠ Undervoltage buses: {len(undervoltage)}")

    # Identify laterals
    laterals = identify_laterals()
    results["laterals"] = laterals
    print(f"\nLaterals identified: {len(laterals)}")

    print(f"\n{'='*60}\n")

    return results


def print_validation_report(results: dict[str, Any]) -> None:
    """Print a formatted validation report.

    Args:
        results: Results dict from validate_feeder
    """
    print(f"\n{'='*60}")
    print(f"  FEEDER VALIDATION REPORT")
    print(f"{'='*60}\n")

    print(f"Circuit: {results['name']}")
    print(f"Converged: {results['converged']}")
    print(f"Buses: {results['num_buses']}")
    print(f"Elements: {results['num_elements']}")

    print(f"\n--- Feeder Head ---")
    head = results.get("head", {})
    print(f"Source bus: {head.get('source_bus', 'unknown')}")
    print(f"Transformers: {len(head.get('transformers', []))}")

    print(f"\n--- Voltage Regulators ---")
    regs = results.get("regulators", [])
    if regs:
        for reg in regs:
            print(f"  {reg['name']}: {reg['vreg']} V ±{reg['band']} V")
    else:
        print("  None")

    print(f"\n--- Capacitor Banks ---")
    caps = results.get("capacitors", [])
    if caps:
        for cap in caps:
            state = "OPEN" if cap["state"] == 0 else "CLOSED"
            print(f"  {cap['name']}: {cap['kvar']} kVAR @ {cap['bus']} [{state}]")
    else:
        print("  None")

    print(f"\n--- Load Summary ---")
    print(f"Total loads: {len(results.get('loads', []))}")
    print(f"Total: {results.get('total_load_kw', 0):.1f} kW, {results.get('total_load_kvar', 0):.1f} kVAR")

    print(f"\n--- Voltage Status ---")
    if "voltage_min" in results:
        print(f"Min: {results['voltage_min']:.4f} pu")
        print(f"Max: {results['voltage_max']:.4f} pu")
        print(f"Mean: {results['voltage_mean']:.4f} pu")

        if results.get("overvoltage_buses"):
            print(f"⚠ Overvoltage: {', '.join(results['overvoltage_buses'][:5])}")
        if results.get("undervoltage_buses"):
            print(f"⚠ Undervoltage: {', '.join(results['undervoltage_buses'][:5])}")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feeder_validation.py <master_dss_file>")
        sys.exit(1)

    dss_file = sys.argv[1]
    results = validate_feeder(dss_file)
    print_validation_report(results)
