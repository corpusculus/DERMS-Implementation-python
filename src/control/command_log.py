"""Audit logging for DER commands.

Provides a complete audit trail of all DER control actions, including
timestamps, setpoints, before/after states, and success status.
"""

from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd

from src.control.der_interface import SetpointApplyResult
from src.control.der_models import DER, DERContainer


DERState = dict[str, dict[str, float]]


class CommandLogger:
    """Logs DER commands with full audit trail.

    Records all control actions to enable:
    - Post-simulation analysis of control decisions
    - Debugging of controller behavior
    - Compliance and verification of control actions
    """

    def __init__(self, output_path: str | Path):
        """Initialize the command logger.

        Args:
            output_path: Path where the CSV log will be written.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []

    def log_command(
        self,
        der: DER | None,
        der_id: str,
        q_commanded: float,
        p_curtail_commanded: float,
        v_before: float,
        v_after: float,
        result: SetpointApplyResult | str,
        timestamp: datetime,
        step: int | None = None,
        time_min: float | None = None,
        time_h: float | None = None,
        controller_mode: str | None = None,
        controller_status: str | None = None,
        command_source: str = "controller",
        reason: str | None = None,
        state_before: dict[str, float] | None = None,
        state_after: dict[str, float] | None = None,
    ) -> None:
        """Record a single DER command.

        Args:
            der: The DER that was commanded.
            q_commanded: Reactive power setpoint that was commanded (kVAR).
            p_curtail_commanded: Active power curtailment that was commanded (kW).
            v_before: Local bus voltage before command (per-unit).
            v_after: Local bus voltage after command (per-unit).
            result: Command status or rich result from apply_setpoints().
            timestamp: When the command was issued.
        """
        if isinstance(result, SetpointApplyResult):
            status = result.status
            q_status = result.q_status
            p_status = result.p_status
            message = result.message
        else:
            status = result
            q_status = result if q_commanded else "not_requested"
            p_status = result if p_curtail_commanded else "not_requested"
            message = ""

        state_before = state_before or {}
        state_after = state_after or {}
        self._records.append(
            {
                "timestamp": timestamp.isoformat(),
                "step": step,
                "time_min": time_min,
                "time_h": time_h,
                "controller_mode": controller_mode,
                "controller_status": controller_status,
                "command_source": command_source,
                "reason": reason or message,
                "der_id": der.id if der else der_id,
                "bus": der.bus if der else "",
                "p_avail_kw": state_before.get("p_avail_kw", der.p_avail_kw if der else 0.0),
                "p_dispatch_before_kw": state_before.get(
                    "p_dispatch_kw",
                    der.p_dispatch_kw if der else 0.0,
                ),
                "p_dispatch_after_kw": state_after.get("p_dispatch_kw", 0.0),
                "q_before_kvar": state_before.get("q_kvar", der.q_kvar if der else 0.0),
                "q_after_kvar": state_after.get("q_kvar", 0.0),
                "q_commanded_kvar": q_commanded,
                "p_curtail_commanded_kw": p_curtail_commanded,
                "v_local_before_pu": v_before,
                "v_local_after_pu": v_after,
                "v_delta_pu": v_after - v_before if v_before and v_after else 0.0,
                "status": status,
                "q_status": q_status,
                "p_status": p_status,
            }
        )

    def log_batch(
        self,
        container: DERContainer,
        q_commands: dict[str, float],
        p_commands: dict[str, float] | None,
        results: dict[str, SetpointApplyResult | str],
        timestamp: datetime,
        voltages_before: dict[str, float] | None = None,
        voltages_after: dict[str, float] | None = None,
        step: int | None = None,
        time_min: float | None = None,
        time_h: float | None = None,
        controller_mode: str | None = None,
        controller_status: str | None = None,
        command_source: str = "controller",
        reason: str | None = None,
        states_before: DERState | None = None,
        states_after: DERState | None = None,
    ) -> None:
        """Log a batch of commands.

        Args:
            container: DER container with all DERs.
            q_commands: DER ID -> Q setpoint (kVAR).
            p_commands: Optional DER ID -> P curtailment (kW).
            results: DER ID -> status/result from apply_setpoints().
            timestamp: When the commands were issued.
            voltages_before: Optional bus -> voltage before commands (pu).
            voltages_after: Optional bus -> voltage after commands (pu).
        """
        states_before = states_before or {}
        states_after = states_after or {}

        for der_id, result in results.items():
            try:
                der = container[der_id]
            except KeyError:
                der = None

            q_cmd = q_commands.get(der_id, 0.0)
            p_cmd = p_commands.get(der_id, 0.0) if p_commands else 0.0
            bus = der.bus if der else ""
            v_before = voltages_before.get(bus, 0.0) if voltages_before else 0.0
            v_after = voltages_after.get(bus, 0.0) if voltages_after else 0.0

            self.log_command(
                der=der,
                der_id=der_id,
                q_commanded=q_cmd,
                p_curtail_commanded=p_cmd,
                v_before=v_before,
                v_after=v_after,
                result=result,
                timestamp=timestamp,
                step=step,
                time_min=time_min,
                time_h=time_h,
                controller_mode=controller_mode,
                controller_status=controller_status,
                command_source=command_source,
                reason=reason,
                state_before=states_before.get(der_id),
                state_after=states_after.get(der_id),
            )

    def save(self) -> Path:
        """Write log records to CSV.

        Returns:
            Path to the written CSV file.
        """
        if not self._records:
            return self.output_path

        df = pd.DataFrame(self._records)
        df.to_csv(self.output_path, index=False)
        return self.output_path

    def clear(self) -> None:
        """Clear all records from the logger."""
        self._records.clear()

    def __len__(self) -> int:
        """Return the number of records currently logged."""
        return len(self._records)


def summarize_command_log(
    command_log: str | Path | pd.DataFrame,
    time_step_minutes: float = 5.0,
) -> dict[str, Any]:
    """Calculate showcase-friendly command audit KPIs.

    Args:
        command_log: Path to ``commands.csv`` or an in-memory DataFrame.
        time_step_minutes: Simulation resolution used for energy totals.

    Returns:
        Dict with command counts, failure counts, and energy/voltage summaries.
    """
    if isinstance(command_log, pd.DataFrame):
        df = command_log.copy()
    else:
        path = Path(command_log)
        if not path.exists():
            return _empty_command_summary()
        df = pd.read_csv(path)

    if df.empty:
        return _empty_command_summary()

    time_step_hours = time_step_minutes / 60.0
    status = df.get("status", pd.Series(dtype=str)).fillna("")
    q_status = df.get("q_status", pd.Series(dtype=str)).fillna("")
    p_status = df.get("p_status", pd.Series(dtype=str)).fillna("")
    q_command = df.get("q_commanded_kvar", pd.Series(dtype=float)).fillna(0.0)
    p_command = df.get("p_curtail_commanded_kw", pd.Series(dtype=float)).fillna(0.0)
    v_delta = df.get("v_delta_pu", pd.Series(dtype=float)).fillna(0.0)

    active = (q_command.abs() > 0) | (p_command > 0)
    failed = status.isin(["failed", "partial"]) | q_status.eq("failed") | p_status.eq("failed")
    out_of_range = status.eq("out_of_range") | q_status.eq("out_of_range") | p_status.eq("out_of_range")

    top_der = ""
    if "der_id" in df.columns and active.any():
        top_der = str(df.loc[active, "der_id"].value_counts().idxmax())

    return {
        "commands_sent": int(active.sum()),
        "commands_applied": int(status.eq("applied").sum()),
        "commands_partial": int(status.eq("partial").sum()),
        "commands_failed": int(failed.sum()),
        "commands_out_of_range": int(out_of_range.sum()),
        "unique_ders_controlled": int(df.loc[active, "der_id"].nunique()) if "der_id" in df.columns else 0,
        "top_controlled_der": top_der,
        "total_reactive_energy_kvarh": float(q_command.abs().sum() * time_step_hours),
        "total_curtailed_energy_kwh": float(p_command.sum() * time_step_hours),
        "avg_voltage_delta_pu": float(v_delta[active].mean()) if active.any() else 0.0,
        "max_voltage_improvement_pu": float(v_delta.abs().max()),
    }


def _empty_command_summary() -> dict[str, Any]:
    """Return an empty command summary with stable keys."""
    return {
        "commands_sent": 0,
        "commands_applied": 0,
        "commands_partial": 0,
        "commands_failed": 0,
        "commands_out_of_range": 0,
        "unique_ders_controlled": 0,
        "top_controlled_der": "",
        "total_reactive_energy_kvarh": 0.0,
        "total_curtailed_energy_kwh": 0.0,
        "avg_voltage_delta_pu": 0.0,
        "max_voltage_improvement_pu": 0.0,
    }
