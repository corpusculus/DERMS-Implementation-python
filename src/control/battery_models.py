"""Battery storage model for the DERMS control layer.

Defines the Battery data structure with SOC (State of Charge) tracking,
charging/discharging limits, efficiency modeling, and container for
managing multiple battery systems.

Follows the same pattern as DER models in der_models.py.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatteryConfig:
    """Configuration for battery systems loaded from YAML.

    Attributes define battery physical limits and control strategy parameters.
    """

    # Physical specifications
    capacity_kwh: float = 200.0
    # Total energy capacity of the battery (kWh)

    power_kw: float = 100.0
    # Maximum charge/discharge power rating (kW)

    efficiency: float = 0.95
    # Round-trip efficiency (0-1), accounts for conversion losses

    # SOC limits
    soc_init: float = 0.5
    # Initial state of charge (0-1), default 50%

    soc_min: float = 0.1
    # Minimum SOC (0-1), below which discharge is prohibited

    soc_max: float = 0.9
    # Maximum SOC (0-1), above which charging is prohibited

    # Control strategy parameters
    charge_threshold_pv_excess_pct: float = 0.2
    # Charge when PV generation exceeds load by this percentage (0-1)
    # Example: 0.2 means charge when PV > load * 1.2

    discharge_voltage_pu: float = 1.06
    # Discharge for voltage support when any bus exceeds this voltage (pu)

    charge_window_start_h: int = 8
    # Earliest hour for charging (24-hour format)

    charge_window_end_h: int = 16
    # Latest hour for charging (24-hour format)

    power_factor: float = 1.0
    # Power factor for battery injection/absorption (default unity)


@dataclass
class Battery:
    """Battery Energy Storage System with SOC tracking.

    Represents a controllable battery that can:
    - Charge (absorb power, negative power output)
    - Discharge (inject power, positive power output)
    - Track state of charge and energy throughput

    Sign convention:
    - Positive power_kw: Discharging (injecting power to grid)
    - Negative power_kw: Charging (absorbing power from grid)
    """

    id: str
    # Unique identifier for the battery

    bus: str
    # Bus connection point

    phases: int
    # Number of phases (1, 2, or 3)

    # Capacity and power limits
    capacity_kwh: float
    # Total energy capacity (kWh)

    power_limit_kw: float
    # Maximum charge/discharge power (kW)

    efficiency: float = 0.95
    # Round-trip efficiency (0-1)

    # SOC state
    soc: float = 0.5
    # Current state of charge (0-1)

    soc_min: float = 0.1
    # Minimum allowable SOC

    soc_max: float = 0.9
    # Maximum allowable SOC

    # Control state
    control_enabled: bool = True
    # Whether this battery participates in control

    # Runtime state (populated from OpenDSS)
    power_kw: float = 0.0
    # Current power output (kW), positive = discharge, negative = charge

    v_local_pu: float = 1.0
    # Local bus voltage (pu)

    # Statistics
    energy_throughput_kwh: float = 0.0
    # Total energy charged/discharged over simulation lifetime (kWh)

    cycles_equivalent: float = 0.0
    # Equivalent full cycles (energy_throughput / capacity)

    def can_charge(self, power_kw: float) -> bool:
        """Check if battery can accept a charging request.

        Args:
            power_kw: Requested charging power (kW), must be negative

        Returns:
            True if charging is possible within SOC and power limits
        """
        if power_kw >= 0:
            return False  # Not a charging request

        abs_power = abs(power_kw)

        # Check power limit
        if abs_power > self.power_limit_kw:
            return False

        # Check if there's any SOC headroom for charging
        # Use a small time fraction (5 minutes = 1/12 hour) for the check
        time_hours = 5 / 60  # One simulation timestep
        energy_added = abs_power * time_hours / self.efficiency
        new_soc = (self.soc * self.capacity_kwh + energy_added) / self.capacity_kwh

        return new_soc <= self.soc_max

    def can_discharge(self, power_kw: float) -> bool:
        """Check if battery can meet a discharging request.

        Args:
            power_kw: Requested discharging power (kW), must be positive

        Returns:
            True if discharging is possible within SOC and power limits
        """
        if power_kw <= 0:
            return False  # Not a discharging request

        # Check power limit
        if power_kw > self.power_limit_kw:
            return False

        # Check if there's any SOC headroom for discharging
        # Use a small time fraction (5 minutes = 1/12 hour) for the check
        time_hours = 5 / 60  # One simulation timestep
        energy_removed = power_kw * time_hours * self.efficiency
        new_soc = (self.soc * self.capacity_kwh - energy_removed) / self.capacity_kwh

        return new_soc >= self.soc_min

    def update_soc(self, power_kw: float, hours: float = 1.0) -> None:
        """Update SOC based on power transfer over a time period.

        Args:
            power_kw: Power during the period (kW), positive = discharge
            hours: Duration of the power transfer in hours
        """
        if power_kw > 0:
            # Discharging: account for efficiency losses
            energy_out = power_kw * hours * self.efficiency
            self.soc = (self.soc * self.capacity_kwh - energy_out) / self.capacity_kwh
            self.energy_throughput_kwh += energy_out
        elif power_kw < 0:
            # Charging: account for efficiency losses
            energy_in = abs(power_kw) * hours / self.efficiency
            self.soc = (self.soc * self.capacity_kwh + energy_in) / self.capacity_kwh
            self.energy_throughput_kwh += energy_in

        # Clamp SOC to valid range
        self.soc = max(self.soc_min, min(self.soc_max, self.soc))

        # Update equivalent cycles
        self.cycles_equivalent = self.energy_throughput_kwh / self.capacity_kwh

        # Store current power
        self.power_kw = power_kw

    @property
    def energy_kwh(self) -> float:
        """Current stored energy (kWh)."""
        return self.soc * self.capacity_kwh

    @property
    def charge_headroom_kwh(self) -> float:
        """Energy capacity available for charging (kWh)."""
        return (self.soc_max - self.soc) * self.capacity_kwh

    @property
    def discharge_headroom_kwh(self) -> float:
        """Energy capacity available for discharging (kWh)."""
        return (self.soc - self.soc_min) * self.capacity_kwh

    @property
    def max_charge_power_kw(self) -> float:
        """Maximum charge power based on SOC headroom (kW)."""
        # Energy that can be stored
        energy_available = self.charge_headroom_kwh
        # Limit by both power rating and energy headroom
        # Assuming 1-hour timestep for safety
        return min(self.power_limit_kw, energy_available * self.efficiency)

    @property
    def max_discharge_power_kw(self) -> float:
        """Maximum discharge power based on SOC headroom (kW)."""
        # Energy that can be extracted
        energy_available = self.discharge_headroom_kwh
        # Limit by both power rating and energy headroom
        # Assuming 1-hour timestep for safety
        return min(self.power_limit_kw, energy_available / self.efficiency)


@dataclass
class BatteryContainer:
    """Container for all batteries in a simulation.

    Provides convenient lookup and filtering methods for battery collections.
    Follows the same pattern as DERContainer.
    """

    batteries: list[Battery] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of batteries in the container."""
        return len(self.batteries)

    def __getitem__(self, battery_id: str) -> Battery:
        """Get a battery by its ID.

        Args:
            battery_id: The unique identifier of the battery.

        Returns:
            The Battery with the given ID.

        Raises:
            KeyError: If no battery with the given ID exists.
        """
        for battery in self.batteries:
            if battery.id == battery_id:
                return battery
        raise KeyError(f"Battery not found: {battery_id}")

    def __contains__(self, battery_id: str) -> bool:
        """Check if a battery ID exists in the container.

        Args:
            battery_id: The unique identifier to check.

        Returns:
            True if a battery with the given ID exists, False otherwise.
        """
        return any(battery.id == battery_id for battery in self.batteries)

    def by_bus(self, bus: str) -> list[Battery]:
        """Get all batteries connected to a specific bus.

        Args:
            bus: The bus name to filter by.

        Returns:
            List of Batteries connected to the specified bus.
        """
        return [battery for battery in self.batteries if battery.bus == bus]

    def enabled(self) -> list[Battery]:
        """Get only control-enabled batteries.

        Returns:
            List of Batteries with control_enabled=True.
        """
        return [battery for battery in self.batteries if battery.control_enabled]

    def enabled_ids(self) -> list[str]:
        """Get IDs of enabled batteries.

        Returns:
            List of Battery IDs with control_enabled=True.
        """
        return [battery.id for battery in self.batteries if battery.control_enabled]

    def total_capacity_kwh(self) -> float:
        """Total energy capacity of all batteries (kWh)."""
        return sum(b.capacity_kwh for b in self.batteries)

    def total_power_kw(self) -> float:
        """Total power rating of all batteries (kW)."""
        return sum(b.power_limit_kw for b in self.batteries)

    def average_soc(self) -> float:
        """Average state of charge across all batteries."""
        if not self.batteries:
            return 0.0
        return sum(b.soc for b in self.batteries) / len(self.batteries)

    def total_energy_kwh(self) -> float:
        """Total stored energy across all batteries (kWh)."""
        return sum(b.energy_kwh for b in self.batteries)
