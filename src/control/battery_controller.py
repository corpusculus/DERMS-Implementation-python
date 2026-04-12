"""Battery controller for rule-based energy storage dispatch.

Implements simple rule-based control:
- Charge when PV generation exceeds load by threshold (midday)
- Discharge for voltage support when voltage exceeds threshold
- Respect SOC limits and efficiency
- Coordinate with PV systems to reduce curtailment
"""

from dataclasses import dataclass
from typing import Any

from src.control.battery_models import Battery, BatteryContainer


@dataclass
class BatteryControlConfig:
    """Configuration for the battery controller.

    Combines physical battery limits with control strategy parameters.
    """

    # Physical limits (from BatteryConfig)
    capacity_kwh: float = 200.0
    power_kw: float = 100.0
    efficiency: float = 0.95
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc_init: float = 0.5

    # Control strategy
    charge_threshold_pv_excess_pct: float = 0.2
    # Charge when PV > load * (1 + threshold)

    discharge_voltage_pu: float = 1.04
    # Discharge for voltage support above this voltage

    charge_window_start_h: int = 8
    # Earliest hour for charging

    charge_window_end_h: int = 16
    # Latest hour for charging

    # Coordination settings
    prioritize_voltage_support: bool = True
    # If True, discharge for voltage support even outside discharge window


class BatteryController:
    """Rule-based battery dispatch controller.

    Control logic:
    1. Check for voltage support: If any bus > discharge_voltage_pu, discharge
    2. Check for charging opportunity: If PV > load and within charge window, charge
    3. Respect SOC limits (soc_min, soc_max)
    4. Respect power limits (power_kw)
    5. Apply efficiency losses

    Sign convention:
    - Positive power: Discharging (injecting to grid)
    - Negative power: Charging (absorbing from grid)
    """

    def __init__(
        self,
        config: BatteryControlConfig,
        battery_container: BatteryContainer,
    ):
        """Initialize with config and battery container.

        Args:
            config: BatteryControlConfig with control strategy
            battery_container: BatteryContainer with all batteries
        """
        self.config = config
        self.battery_container = battery_container

    def compute_battery_commands(
        self,
        bus_voltages: dict[str, float],
        pv_generation_kw: float,
        load_kw: float,
        hour_of_day: float,
    ) -> dict[str, float]:
        """Compute battery power commands for all enabled batteries.

        Args:
            bus_voltages: Current bus voltages from OpenDSS (pu)
            pv_generation_kw: Total PV generation (kW)
            load_kw: Total load (kW)
            hour_of_day: Current hour (0-24)

        Returns:
            Dictionary mapping battery ID to power setpoint (kW)
            - Positive: Discharging (injecting power)
            - Negative: Charging (absorbing power)
            - 0: No action
        """
        commands: dict[str, float] = {}

        # Check if voltage support is needed
        max_voltage = max(bus_voltages.values()) if bus_voltages else 1.0
        voltage_support_needed = max_voltage > self.config.discharge_voltage_pu

        # Check if charging conditions are met
        in_charge_window = (
            self.config.charge_window_start_h
            <= hour_of_day
            < self.config.charge_window_end_h
        )
        pv_excess = pv_generation_kw - load_kw
        has_pv_excess = pv_excess > load_kw * self.config.charge_threshold_pv_excess_pct

        # Determine control mode
        if voltage_support_needed and self.config.prioritize_voltage_support:
            # Discharge for voltage support
            mode = "discharge_voltage"
        elif in_charge_window and has_pv_excess:
            # Charge from excess PV
            mode = "charge_pv"
        else:
            # No action
            mode = "idle"

        # Compute commands for each enabled battery
        for battery in self.battery_container.enabled():
            cmd = self._compute_battery_command(
                battery,
                mode,
                pv_excess if mode == "charge_pv" else 0.0,
            )
            if cmd != 0.0:
                commands[battery.id] = cmd

        return commands

    def _compute_battery_command(
        self,
        battery: Battery,
        mode: str,
        pv_excess_kw: float,
    ) -> float:
        """Compute power command for a single battery.

        Args:
            battery: The battery to compute command for
            mode: Control mode ("discharge_voltage", "charge_pv", "idle")
            pv_excess_kw: Excess PV power available for charging (kW)

        Returns:
            Power setpoint (kW), positive = discharge, negative = charge
        """
        if mode == "discharge_voltage":
            # Discharge for voltage support
            # Use maximum available discharge power
            max_power = battery.max_discharge_power_kw
            if max_power > 0 and battery.can_discharge(max_power):
                return max_power
            return 0.0

        elif mode == "charge_pv":
            # Charge from excess PV
            # Limit by available PV excess and battery charge capability
            max_charge = battery.max_charge_power_kw

            # Limit charging to available PV excess (avoid grid import)
            charge_power = min(max_charge, pv_excess_kw)

            if charge_power > 0 and battery.can_charge(-charge_power):
                return -charge_power  # Negative for charging
            return 0.0

        else:  # idle
            return 0.0

    def compute_battery_commands_simple(
        self,
        bus_voltages: dict[str, float],
        previous_commands: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute battery commands based on voltages only (simplified interface).

        This simplified interface matches the signature used by PV controllers
        and doesn't require PV/load data. Batteries discharge for voltage
        support when needed.

        Args:
            bus_voltages: Current bus voltages from OpenDSS (pu)
            previous_commands: Previous timestep commands (not used for batteries)

        Returns:
            Dictionary mapping battery ID to power setpoint (kW)
        """
        commands: dict[str, float] = {}

        max_voltage = max(bus_voltages.values()) if bus_voltages else 1.0

        # Only discharge if voltage exceeds threshold
        if max_voltage > self.config.discharge_voltage_pu:
            for battery in self.battery_container.enabled():
                # Use proportional discharge based on voltage severity
                severity = (max_voltage - self.config.discharge_voltage_pu) / 0.02  # 0.02 pu range
                severity = max(0.0, min(1.0, severity))

                discharge_power = severity * battery.max_discharge_power_kw

                if discharge_power > 0 and battery.can_discharge(discharge_power):
                    commands[battery.id] = discharge_power

        return commands

    def update_battery_state(
        self,
        commands: dict[str, float],
        time_step_hours: float,
    ) -> None:
        """Update battery SOC based on commands.

        Args:
            commands: Battery power commands from compute_battery_commands
            time_step_hours: Duration of timestep in hours
        """
        for battery in self.battery_container.batteries:
            power = commands.get(battery.id, 0.0)
            battery.update_soc(power, time_step_hours)

    def get_battery_summary(self) -> dict[str, Any]:
        """Get summary statistics for all batteries.

        Returns:
            Dictionary with battery statistics:
            - count: Number of batteries
            - total_capacity_kwh: Total capacity
            - total_energy_kwh: Total stored energy
            - average_soc: Average SOC across batteries
            - total_throughput_kwh: Total energy throughput
            - total_cycles: Total equivalent cycles
        """
        return {
            "count": len(self.battery_container),
            "total_capacity_kwh": self.battery_container.total_capacity_kwh(),
            "total_energy_kwh": self.battery_container.total_energy_kwh(),
            "average_soc": self.battery_container.average_soc(),
            "total_throughput_kwh": sum(b.energy_throughput_kwh for b in self.battery_container.batteries),
            "total_cycles": sum(b.cycles_equivalent for b in self.battery_container.batteries),
        }
