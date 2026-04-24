"""OpenDSS interface for reading/writing DER state.

This module provides functions to:
- Load DER definitions from CSV files
- Read current DER state from OpenDSS (P, Q, V)
- Apply control setpoints (Q, P curtailment) to DERs
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SetpointApplyResult:
    """Outcome of applying Q/P commands to one DER.

    The aggregate ``status`` field is retained for quick summaries, while the
    command-specific fields preserve partial outcomes for audit logs.
    """

    status: str
    q_status: str = "not_requested"
    p_status: str = "not_requested"
    message: str = ""

    def __str__(self) -> str:
        """Return aggregate status for display."""
        return self.status

    def __eq__(self, other: object) -> bool:
        """Allow legacy comparisons to the aggregate status string."""
        if isinstance(other, str):
            return self.status == other
        return super().__eq__(other)


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
) -> dict[str, SetpointApplyResult]:
    """Apply multiple setpoints to DERs.

    Applies reactive power and/or active power curtailment commands
    to multiple DERs in a single call.

    Args:
        container: DER container with all DERs.
        q_commands: DER ID -> Q setpoint (kVAR).
        p_commands: Optional DER ID -> P curtailment (kW).

    Returns:
        Dict of DER ID -> SetpointApplyResult. ``result.status`` is one of
        "applied", "partial", "failed", "out_of_range", or "no_command".
    """
    results: dict[str, SetpointApplyResult] = {}
    p_commands = p_commands or {}
    der_ids = dict.fromkeys([*q_commands.keys(), *p_commands.keys()])

    for der_id in der_ids:
        try:
            der = container[der_id]
        except KeyError:
            q_status = "failed" if der_id in q_commands else "not_requested"
            p_status = "failed" if der_id in p_commands else "not_requested"
            results[der_id] = SetpointApplyResult(
                status="failed",
                q_status=q_status,
                p_status=p_status,
                message="DER not found in container",
            )
            continue

        q_status = "not_requested"
        p_status = "not_requested"
        messages: list[str] = []

        if der_id in q_commands:
            q_kvar = q_commands[der_id]
            if not der.can_provide_q(q_kvar):
                q_status = "out_of_range"
                messages.append("Q command outside inverter capability")
            else:
                q_status = "applied" if apply_q_setpoint(der, q_kvar) else "failed"
                if q_status == "failed":
                    messages.append("OpenDSS rejected Q command")

        if der_id in p_commands:
            p_kw = p_commands[der_id]
            if p_kw < 0 or p_kw > der.p_avail_kw:
                p_status = "out_of_range"
                messages.append("P curtailment outside available range")
            else:
                p_status = "applied" if apply_p_curtailment(der, p_kw) else "failed"
                if p_status == "failed":
                    messages.append("OpenDSS rejected P curtailment command")

        requested_statuses = [
            status
            for status in (q_status, p_status)
            if status != "not_requested"
        ]
        if not requested_statuses:
            overall_status = "no_command"
        elif all(status == "applied" for status in requested_statuses):
            overall_status = "applied"
        elif any(status == "applied" for status in requested_statuses):
            overall_status = "partial"
        elif any(status == "out_of_range" for status in requested_statuses):
            overall_status = "out_of_range"
        else:
            overall_status = "failed"

        results[der_id] = SetpointApplyResult(
            status=overall_status,
            q_status=q_status,
            p_status=p_status,
            message="; ".join(messages),
        )

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
