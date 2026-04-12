"""Audit logging for DER commands.

Provides a complete audit trail of all DER control actions, including
timestamps, setpoints, before/after states, and success status.
"""

from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd

from src.control.der_models import DER, DERContainer


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
        der: DER,
        q_commanded: float,
        p_curtail_commanded: float,
        v_before: float,
        v_after: float,
        status: str,
        timestamp: datetime,
    ) -> None:
        """Record a single DER command.

        Args:
            der: The DER that was commanded.
            q_commanded: Reactive power setpoint that was commanded (kVAR).
            p_curtail_commanded: Active power curtailment that was commanded (kW).
            v_before: Local bus voltage before command (per-unit).
            v_after: Local bus voltage after command (per-unit).
            status: Command status ("applied", "failed", "out_of_range").
            timestamp: When the command was issued.
        """
        self._records.append(
            {
                "timestamp": timestamp.isoformat(),
                "der_id": der.id,
                "bus": der.bus,
                "p_avail_kw": der.p_avail_kw,
                "p_dispatch_before_kw": der.p_dispatch_kw,
                "q_before_kvar": der.q_kvar,
                "q_commanded_kvar": q_commanded,
                "p_curtail_commanded_kw": p_curtail_commanded,
                "v_local_before_pu": v_before,
                "v_local_after_pu": v_after,
                "status": status,
            }
        )

    def log_batch(
        self,
        container: DERContainer,
        q_commands: dict[str, float],
        p_commands: dict[str, float] | None,
        results: dict[str, str],
        timestamp: datetime,
        voltages_before: dict[str, float] | None = None,
        voltages_after: dict[str, float] | None = None,
    ) -> None:
        """Log a batch of commands.

        Args:
            container: DER container with all DERs.
            q_commands: DER ID -> Q setpoint (kVAR).
            p_commands: Optional DER ID -> P curtailment (kW).
            results: DER ID -> status from apply_setpoints().
            timestamp: When the commands were issued.
            voltages_before: Optional bus -> voltage before commands (pu).
            voltages_after: Optional bus -> voltage after commands (pu).
        """
        for der_id, status in results.items():
            try:
                der = container[der_id]
            except KeyError:
                # DER not found - skip logging
                continue

            q_cmd = q_commands.get(der_id, 0.0)
            p_cmd = p_commands.get(der_id, 0.0) if p_commands else 0.0
            v_before = voltages_before.get(der.bus, 0.0) if voltages_before else 0.0
            v_after = voltages_after.get(der.bus, 0.0) if voltages_after else 0.0

            self.log_command(
                der=der,
                q_commanded=q_cmd,
                p_curtail_commanded=p_cmd,
                v_before=v_before,
                v_after=v_after,
                status=status,
                timestamp=timestamp,
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
