"""Heuristic Volt-VAR controller with P curtailment fallback.

Implements a centralized heuristic controller that:
1. Uses reactive power absorption when voltage exceeds q_activation_pu
2. Only curtails active power when voltage exceeds curtailment_pu
3. Applies deadband and ramp limiting to prevent oscillation
4. Prioritizes Q before P to minimize energy curtailment
"""

from dataclasses import dataclass
from typing import Any

from src.control.der_models import DER, DERContainer


@dataclass
class HeuristicConfig:
    """Configuration for the heuristic controller."""

    q_activation_pu: float = 1.03
    # Voltage threshold for Q activation (default 1.03)

    curtailment_pu: float = 1.05
    # Voltage threshold for P curtailment (default 1.05)

    deadband_pu: float = 0.005
    # Deadband around 1.0 pu where no action is taken (default 0.005)

    q_ramp_max_kvar: float = float("inf")
    # Max Q change per timestep (default: no limit)

    q_utilization_limit: float = 0.5
    # Fraction of Q capability that the heuristic may use at full severity.

    p_ramp_max_kw: float = float("inf")
    # Max P change per timestep (default: no limit)

    v_lower_limit: float = 0.95
    # Lower voltage limit (0.95)


class HeuristicController:
    """Centralized heuristic Volt-VAR controller with P curtailment fallback.

    Control logic:
    1. For each DER, check local voltage against thresholds
    2. If v > q_activation_pu: compute Q absorption proportional to severity
    3. Apply deadband and ramp limits
    4. If any bus still > curtailment_pu after Q: add P curtailment for DERs at that bus
    5. Prioritize DERs at highest voltage buses for curtailment
    """

    def __init__(self, config: HeuristicConfig, der_container: DERContainer):
        """Initialize with config and DER container.

        Args:
            config: HeuristicConfig with thresholds and limits
            der_container: DERContainer with all DERs
        """
        self.config = config
        self.der_container = der_container

    def compute_commands(
        self,
        bus_voltages: dict[str, float],
        previous_commands: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute Q and P commands for all DERs.

        Args:
            bus_voltages: Current bus voltages from OpenDSS (pu)
            previous_commands: Previous timestep commands as {der_id: (q, p)}
                for ramp limiting

        Returns:
            (q_commands, p_commands) dictionaries mapping DER ID to setpoints
                - q_commands: DER ID -> Q setpoint (kVAR), negative = absorption
                - p_commands: DER ID -> P curtailment (kW), 0 = no curtailment
        """
        q_commands: dict[str, float] = {}
        p_commands: dict[str, float] = {}

        # Step 1: Compute Q commands for all enabled DERs
        for der in self.der_container.enabled():
            v_local = bus_voltages.get(der.bus, 1.0)
            q_cmd = self._compute_q_command(der, v_local)

            # Only command if non-zero
            if q_cmd != 0.0:
                q_commands[der.id] = q_cmd

        # Step 2: Check if P curtailment is needed (after Q application)
        # For buses exceeding curtailment_pu, compute P curtailment
        buses_over_curtailment = {
            bus: v for bus, v in bus_voltages.items()
            if v > self.config.curtailment_pu
        }

        if buses_over_curtailment:
            # Sort buses by voltage (highest first) for priority
            sorted_buses = sorted(
                buses_over_curtailment.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for bus, v_bus in sorted_buses:
                # Get DERs at this bus
                ders_at_bus = self.der_container.by_bus(bus)
                for der in ders_at_bus:
                    if not der.control_enabled:
                        continue

                    p_cmd = self._compute_p_curtailment(der, v_bus)
                    if p_cmd > 0:
                        # P command is the amount to curtail (reduce from available)
                        p_commands[der.id] = p_cmd

        # Step 3: Apply ramp limits if previous commands provided
        if previous_commands is not None:
            q_commands, p_commands = self._apply_ramp_limits(
                q_commands,
                p_commands,
                previous_commands,
            )

        return q_commands, p_commands

    def _compute_q_command(self, der: DER, v_local: float) -> float:
        """Compute reactive power setpoint for a single DER.

        Args:
            der: The DER to compute command for
            v_local: Local bus voltage (pu)

        Returns:
            Q setpoint (kVAR), negative = absorption, 0 = no action
        """
        cfg = self.config

        # Check deadband around 1.0 pu
        if abs(v_local - 1.0) < cfg.deadband_pu:
            return 0.0

        # Only act if voltage exceeds activation threshold
        if v_local <= cfg.q_activation_pu:
            return 0.0

        # Compute severity based on how far above activation
        # Severity ranges from 0 to 1 as voltage goes from activation to curtailment
        severity = (v_local - cfg.q_activation_pu) / (cfg.curtailment_pu - cfg.q_activation_pu)
        severity = max(0.0, min(1.0, severity))

        # Command is absorption (negative Q) proportional to severity, with
        # an upper bound on how much of the inverter VAR headroom we use.
        q_max = der.q_max_kvar
        q_command = -severity * q_max * cfg.q_utilization_limit

        # Clamp to capability curve
        q_command = max(der.q_min_kvar, min(der.q_max_kvar, q_command))

        return q_command

    def _compute_p_curtailment(self, der: DER, v_local: float) -> float:
        """Compute active power curtailment for a single DER.

        Args:
            der: The DER to compute curtailment for
            v_local: Local bus voltage (pu)

        Returns:
            Amount of power to curtail (kW), 0 = no curtailment
        """
        cfg = self.config

        # Only curtail if voltage exceeds curtailment threshold
        if v_local <= cfg.curtailment_pu:
            return 0.0

        # Compute severity based on how far above curtailment threshold
        # Allow 0.02 pu range from curtailment to full curtailment
        headroom = 0.02  # pu range for proportional curtailment
        severity = (v_local - cfg.curtailment_pu) / headroom
        severity = max(0.0, min(1.0, severity))

        # Curtailment is proportional to severity
        # Return the amount to reduce (not the setpoint)
        curtailment_kw = severity * der.p_avail_kw

        return curtailment_kw

    def _apply_ramp_limits(
        self,
        q_commands: dict[str, float],
        p_commands: dict[str, float],
        previous_commands: dict[str, tuple[float, float]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Limit command changes to prevent oscillation.

        Args:
            q_commands: Current Q commands
            p_commands: Current P curtailment commands
            previous_commands: Previous timestep commands as {der_id: (q, p)}

        Returns:
            (q_commands_limited, p_commands_limited) with ramp rates applied
        """
        q_limited: dict[str, float] = {}
        p_limited: dict[str, float] = {}

        cfg = self.config

        # Process Q commands with ramp limiting
        for der_id, q_cmd in q_commands.items():
            if der_id in previous_commands:
                prev_q, _ = previous_commands[der_id]
                delta = q_cmd - prev_q

                # Limit change
                if abs(delta) > cfg.q_ramp_max_kvar:
                    delta = max(-cfg.q_ramp_max_kvar, min(cfg.q_ramp_max_kvar, delta))
                    q_limited[der_id] = prev_q + delta
                else:
                    q_limited[der_id] = q_cmd
            else:
                # No previous command, apply as-is
                q_limited[der_id] = q_cmd

        # Also include DERs that had previous Q commands but now have 0
        for der_id, (prev_q, _) in previous_commands.items():
            if der_id not in q_commands:
                # DER is being commanded to 0
                if der_id not in q_limited:
                    delta = 0.0 - prev_q
                    if abs(delta) > cfg.q_ramp_max_kvar:
                        delta = max(-cfg.q_ramp_max_kvar, min(cfg.q_ramp_max_kvar, delta))
                        q_limited[der_id] = prev_q + delta
                    else:
                        q_limited[der_id] = 0.0

        # Process P commands with ramp limiting
        for der_id, p_cmd in p_commands.items():
            if der_id in previous_commands:
                _, prev_p = previous_commands[der_id]
                delta = p_cmd - prev_p

                # Limit change
                if abs(delta) > cfg.p_ramp_max_kw:
                    delta = max(-cfg.p_ramp_max_kw, min(cfg.p_ramp_max_kw, delta))
                    p_limited[der_id] = prev_p + delta
                else:
                    p_limited[der_id] = p_cmd
            else:
                # No previous command, apply as-is
                p_limited[der_id] = p_cmd

        # Also include DERs that had previous P commands but now have 0
        for der_id, (_, prev_p) in previous_commands.items():
            if der_id not in p_commands and prev_p != 0:
                # DER is being commanded to 0
                if der_id not in p_limited:
                    delta = 0.0 - prev_p
                    if abs(delta) > cfg.p_ramp_max_kw:
                        delta = max(-cfg.p_ramp_max_kw, min(cfg.p_ramp_max_kw, delta))
                        p_limited[der_id] = prev_p + delta
                    else:
                        p_limited[der_id] = 0.0

        return q_limited, p_limited
