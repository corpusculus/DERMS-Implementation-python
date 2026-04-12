"""
src/sim/opendss_interface.py
-----------------------------
Minimal interface to OpenDSS via OpenDSSDirect.py.

Provides four stable functions used by all simulation scripts:
  - load_feeder()
  - solve_power_flow()
  - get_bus_voltages()
  - export_results()
"""

import os
import pathlib
from typing import Any

import pandas as pd

try:
    import opendssdirect as dss

    OPENDSS_AVAILABLE = True
except ImportError:  # pragma: no cover
    dss = None  # type: ignore[assignment]
    OPENDSS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_feeder(dss_master_file: str | pathlib.Path) -> None:
    """Load an OpenDSS feeder model from a master DSS file.

    This compiles the master file and prepares the circuit for solving.

    Args:
        dss_master_file: Absolute or relative path to the OpenDSS master .dss file.

    Raises:
        ImportError: If OpenDSSDirect.py is not installed.
        FileNotFoundError: If the DSS master file does not exist.
        RuntimeError: If OpenDSS reports an error during compilation.
    """
    _require_opendss()
    master = pathlib.Path(dss_master_file).resolve()
    if not master.exists():
        raise FileNotFoundError(f"OpenDSS master file not found: {master}")

    # OpenDSS changes the current working directory during compilation
    # Save the original CWD and restore it after
    original_cwd = os.getcwd()
    
    try:
        dss.run_command("Clear")
        result = dss.run_command(f"Compile '{master}'")
        if result:  # OpenDSS returns an error string on failure, empty string on success
            raise RuntimeError(f"OpenDSS compile error: {result}")
    finally:
        os.chdir(original_cwd)


def solve_power_flow() -> None:
    """Solve the current power-flow snapshot in OpenDSS.

    Raises:
        ImportError: If OpenDSSDirect.py is not installed.
        RuntimeError: If the power flow does not converge.
    """
    _require_opendss()
    dss.run_command("Solve")
    if not dss.Solution.Converged():
        raise RuntimeError("Power flow did not converge.")


def get_bus_voltages() -> dict[str, float]:
    """Return per-unit voltages for all buses in the circuit.

    For multi-phase buses the average (mean) per-unit magnitude across
    phases is returned so that downstream code always works with a single
    float per bus.

    Returns:
        Dictionary mapping bus name (str) to average per-unit voltage magnitude (float).

    Raises:
        ImportError: If OpenDSSDirect.py is not installed.
    """
    _require_opendss()
    voltages: dict[str, float] = {}

    bus_names = dss.Circuit.AllBusNames()
    for bus_name in bus_names:
        dss.Circuit.SetActiveBus(bus_name)
        pu_mags = dss.Bus.puVmagAngle()[::2]  # Every other value is a magnitude
        if pu_mags:
            voltages[bus_name] = float(sum(pu_mags) / len(pu_mags))

    return voltages


def export_results(voltages: dict[str, float], output_path: str | pathlib.Path) -> pathlib.Path:
    """Write bus voltage results to a CSV file.

    Args:
        voltages: Dict mapping bus name → per-unit voltage magnitude.
        output_path: Path where the CSV file will be written.

    Returns:
        Resolved path to the written CSV file.
    """
    out = pathlib.Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [{"bus": bus, "v_pu": v} for bus, v in sorted(voltages.items())],
        columns=["bus", "v_pu"],
    )
    df.to_csv(out, index=False)
    return out


def get_circuit_summary() -> dict[str, Any]:
    """Return a short summary dict of the loaded circuit.

    Returns:
        Dict with keys: name, num_buses, num_elements, converged.
    """
    _require_opendss()
    return {
        "name": dss.Circuit.Name(),
        "num_buses": len(dss.Circuit.AllBusNames()),
        "num_elements": dss.Circuit.NumCktElements(),
        "converged": dss.Solution.Converged(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_opendss() -> None:
    if not OPENDSS_AVAILABLE:
        raise ImportError(
            "OpenDSSDirect.py is not installed. "
            "Install it with: pip install OpenDSSDirect.py"
        )
