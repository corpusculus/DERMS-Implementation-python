"""OpenDSS interface for reading/writing DER state.

This module provides functions to:
- Load DER definitions from CSV files
- Read current DER state from OpenDSS (P, Q, V)
- Apply control setpoints (Q, P curtailment) to DERs
"""

from typing import Any

import pandas as pd

try:
    import opendssdirect as dss

    OPENDSS_AVAILABLE = True
except ImportError:  # pragma: no cover
    dss = None  # type: ignore[assignment]
    OPENDSS_AVAILABLE = False

from src.control.der_models import DER, DERContainer
from src.sim.opendss_interface import _require_opendss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ders_from_csv(
    csv_path: str,
    pv_prefix: str = "pv",
    scale_factor: float = 1.0,
) -> DERContainer:
    """Load DER definitions from CSV (existing format).

    Expected columns: bus, phases, p_kw, kva
    Generates IDs as: {prefix}_{index:03d}_{bus}

    Args:
        csv_path: Path to the CSV file containing DER definitions.
        pv_prefix: Prefix for generated DER IDs (default: "pv").

    Returns:
        A DERContainer populated with DERs from the CSV file.
    """
    df = pd.read_csv(csv_path)

    ders = []
    for i, row in df.iterrows():
        der_id = f"{pv_prefix}_{i+1:03d}_{row['bus']}"
        ders.append(
            DER(
                id=der_id,
                bus=str(row["bus"]),
                phases=int(row["phases"]),
                p_kw_rated=float(row["p_kw"]) * scale_factor,
                s_kva_rated=float(row["kva"]) * scale_factor,
            )
        )
    return DERContainer(ders)


def read_der_state(container: DERContainer) -> None:
    """Update DER runtime state from OpenDSS.

    Updates p_avail_kw, p_dispatch_kw, q_kvar, v_local_pu for each DER.
    This modifies the DER objects in-place.

    Args:
        container: DERContainer whose DERs should have their state updated.
    """
    _require_opendss()

    for der in container.ders:
        # Set active PVSystem element
        if not dss.Circuit.SetActiveElement(f"PVSystem.{der.id}"):
            # DER not found in circuit - set to zero state
            der.p_avail_kw = 0.0
            der.p_dispatch_kw = 0.0
            der.q_kvar = 0.0
            continue

        # Read current state
        der.p_dispatch_kw = abs(dss.PVsystems.kW())
        der.p_avail_kw = max(0.0, dss.PVsystems.Pmpp() * dss.PVsystems.Irradiance())
        der.q_kvar = dss.PVsystems.kvar()

        # Read local voltage at the DER's connected nodes, not the whole bus average.
        bus_spec = dss.CktElement.BusNames()[0]
        der.v_local_pu = _get_bus_spec_voltage_pu(bus_spec)


def apply_q_setpoint(der: DER, q_kvar: float) -> bool:
    """Apply reactive power setpoint to a DER.

    Converts the Q command to a power factor setting in OpenDSS.
    The power factor sign indicates direction: positive = injection,
    negative = absorption.

    Args:
        der: DER to control.
        q_kvar: Reactive power command (positive = injection, negative = absorption).

    Returns:
        True if successful, False otherwise (DER not found in circuit).
    """
    _require_opendss()

    if not dss.Circuit.SetActiveElement(f"PVSystem.{der.id}"):
        return False

    # Convert Q command to power factor
    # pf = P / sqrt(P² + Q²), signed by Q direction
    s = (der.p_dispatch_kw**2 + q_kvar**2) ** 0.5
    if s == 0:
        pf = 1.0
    else:
        pf = der.p_dispatch_kw / s
        # Negative pf in OpenDSS means absorbing reactive power
        if q_kvar < 0:
            pf = -pf

    dss.run_command(f"PVSystem.{der.id}.pf={pf:.4f}")
    return True


def apply_p_curtailment(der: DER, p_kw: float) -> bool:
    """Apply active power curtailment to a DER.

    Sets the Pmpp parameter of the PVSystem, which effectively
    curtails the active power output.

    Args:
        der: DER to control.
        p_kw: Curtailed active power (must be >= 0 and <= p_avail_kw).

    Returns:
        True if successful, False otherwise (DER not found or invalid value).
    """
    _require_opendss()

    if not dss.Circuit.SetActiveElement(f"PVSystem.{der.id}"):
        return False

    # Validate: curtailment must be non-negative and not exceed available
    if p_kw < 0 or p_kw > der.p_avail_kw:
        return False

    target_pmpp = max(0.0, der.p_avail_kw - p_kw)
    dss.run_command(f"PVSystem.{der.id}.Pmpp={target_pmpp:.3f}")
    return True


def apply_setpoints(
    container: DERContainer,
    q_commands: dict[str, float],
    p_commands: dict[str, float] | None = None,
) -> dict[str, str]:
    """Apply multiple setpoints to DERs.

    Applies reactive power and/or active power curtailment commands
    to multiple DERs in a single call.

    Args:
        container: DER container with all DERs.
        q_commands: DER ID -> Q setpoint (kVAR).
        p_commands: Optional DER ID -> P curtailment (kW).

    Returns:
        Dict of DER ID -> status ("applied", "failed", "out_of_range").
    """
    results: dict[str, str] = {}

    # Apply Q commands
    for der_id, q_kvar in q_commands.items():
        try:
            der = container[der_id]
        except KeyError:
            results[der_id] = "failed"
            continue

        if not der.can_provide_q(q_kvar):
            results[der_id] = "out_of_range"
            continue

        if apply_q_setpoint(der, q_kvar):
            results[der_id] = "applied"
        else:
            results[der_id] = "failed"

    # Apply P commands if provided
    if p_commands:
        for der_id, p_kw in p_commands.items():
            try:
                der = container[der_id]
            except KeyError:
                results[der_id] = "failed"
                continue

            if apply_p_curtailment(der, p_kw):
                results[der_id] = "applied"
            else:
                results[der_id] = "failed"

    return results


def _get_bus_spec_voltage_pu(bus_spec: str) -> float:
    """Return the average voltage for the nodes referenced by a bus spec."""
    base_bus, *node_parts = bus_spec.split(".")
    if not dss.Circuit.SetActiveBus(base_bus):
        return 1.0

    pu_values = dss.Bus.puVmagAngle()[::2]
    bus_nodes = dss.Bus.Nodes()
    node_to_pu = {
        node: pu
        for node, pu in zip(bus_nodes, pu_values)
        if node > 0
    }

    requested_nodes = [int(node) for node in node_parts if node]
    selected = [node_to_pu[node] for node in requested_nodes if node in node_to_pu]
    if not selected:
        selected = list(node_to_pu.values())

    return float(sum(selected) / len(selected)) if selected else 1.0
