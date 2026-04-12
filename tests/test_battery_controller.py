"""Tests for battery controller."""

import pytest

from src.control.battery_models import (
    Battery,
    BatteryContainer,
    BatteryConfig,
)
from src.control.battery_controller import (
    BatteryController,
    BatteryControlConfig,
)


@pytest.fixture
def sample_batteries() -> BatteryContainer:
    """Create sample battery container."""
    return BatteryContainer([
        Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
            soc_min=0.1,
            soc_max=0.9,
            efficiency=0.95,
        ),
        Battery(
            id="batt_002",
            bus="646",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
            soc_min=0.1,
            soc_max=0.9,
            efficiency=0.95,
        ),
    ])


@pytest.fixture
def sample_config() -> BatteryControlConfig:
    """Create sample battery control config."""
    return BatteryControlConfig(
        capacity_kwh=200.0,
        power_kw=100.0,
        efficiency=0.95,
        soc_min=0.1,
        soc_max=0.9,
        soc_init=0.5,
        charge_threshold_pv_excess_pct=0.2,
        discharge_voltage_pu=1.04,
        charge_window_start_h=8,
        charge_window_end_h=16,
    )


class TestBatteryController:
    """Tests for BatteryController."""

    def test_controller_creation(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test controller initialization."""
        controller = BatteryController(sample_config, sample_batteries)

        assert controller.config is sample_config
        assert controller.battery_container is sample_batteries

    def test_compute_voltage_support_discharge(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test discharge for voltage support."""
        controller = BatteryController(sample_config, sample_batteries)

        # Voltage exceeds threshold
        bus_voltages = {"675": 1.045, "646": 1.042}
        pv_gen = 100.0
        load = 80.0
        hour = 12.0

        commands = controller.compute_battery_commands(
            bus_voltages, pv_gen, load, hour
        )

        # Should command discharge (positive power)
        assert len(commands) > 0
        for power in commands.values():
            assert power > 0  # Discharging

    def test_compute_pv_excess_charge(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test charging when PV exceeds load."""
        controller = BatteryController(sample_config, sample_batteries)

        # PV excess condition, within charge window
        bus_voltages = {"675": 1.02, "646": 1.01}
        pv_gen = 200.0
        load = 100.0  # PV is 2x load, exceeds 20% threshold
        hour = 12.0  # Within 8-16 charge window

        commands = controller.compute_battery_commands(
            bus_voltages, pv_gen, load, hour
        )

        # Should command charge (negative power)
        assert len(commands) > 0
        for power in commands.values():
            assert power < 0  # Charging

    def test_compute_no_action_outside_window(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test no action when conditions not met."""
        controller = BatteryController(sample_config, sample_batteries)

        # Low voltage, no PV excess, outside charge window
        bus_voltages = {"675": 1.0, "646": 1.0}
        pv_gen = 50.0
        load = 100.0  # No excess
        hour = 20.0  # Outside charge window

        commands = controller.compute_battery_commands(
            bus_voltages, pv_gen, load, hour
        )

        # Should be idle
        assert len(commands) == 0

    def test_compute_simple_voltage_only(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test simplified voltage-only interface."""
        controller = BatteryController(sample_config, sample_batteries)

        # Voltage below threshold - no action
        commands = controller.compute_battery_commands_simple(
            {"675": 1.0, "646": 1.0}
        )
        assert len(commands) == 0

        # Voltage above threshold - discharge
        commands = controller.compute_battery_commands_simple(
            {"675": 1.045, "646": 1.042}
        )
        assert len(commands) > 0
        for power in commands.values():
            assert power > 0  # Discharging

    def test_discharge_respects_soc_min(
        self,
        sample_config: BatteryControlConfig,
    ) -> None:
        """Test discharge is limited by minimum SOC."""
        # Create battery at minimum SOC
        batteries = BatteryContainer([
            Battery(
                id="batt_001",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
                soc=0.1,  # At minimum
                soc_min=0.1,
                soc_max=0.9,
            ),
        ])

        controller = BatteryController(sample_config, batteries)

        # High voltage calling for discharge
        commands = controller.compute_battery_commands_simple(
            {"675": 1.05}
        )

        # Battery should not discharge (at min SOC)
        assert len(commands) == 0

    def test_charge_respects_soc_max(
        self,
        sample_config: BatteryControlConfig,
    ) -> None:
        """Test charge is limited by maximum SOC."""
        # Create battery at maximum SOC
        batteries = BatteryContainer([
            Battery(
                id="batt_001",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
                soc=0.9,  # At maximum
                soc_min=0.1,
                soc_max=0.9,
            ),
        ])

        controller = BatteryController(sample_config, batteries)

        # PV excess condition
        commands = controller.compute_battery_commands(
            {"675": 1.0},
            pv_generation_kw=200.0,
            load_kw=100.0,
            hour_of_day=12.0,
        )

        # Battery should not charge (at max SOC)
        assert len(commands) == 0

    def test_update_battery_state(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test battery SOC update after commands."""
        controller = BatteryController(sample_config, sample_batteries)

        initial_soc = sample_batteries.batteries[0].soc

        # Discharge command
        commands = {"batt_001": 50.0, "batt_002": 30.0}
        controller.update_battery_state(commands, time_step_hours=1.0)

        # SOC should decrease
        assert sample_batteries.batteries[0].soc < initial_soc
        assert sample_batteries.batteries[0].power_kw == 50.0

        # Throughput should increase
        assert sample_batteries.batteries[0].energy_throughput_kwh > 0

    def test_charge_window_enforcement(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test charging only happens within specified window."""
        controller = BatteryController(sample_config, sample_batteries)

        # Early morning - outside window
        commands_early = controller.compute_battery_commands(
            {"675": 1.0},
            pv_generation_kw=200.0,
            load_kw=100.0,
            hour_of_day=6.0,  # Before 8 AM
        )
        assert len(commands_early) == 0  # Should not charge

        # Mid-day - within window
        commands_midday = controller.compute_battery_commands(
            {"675": 1.0},
            pv_generation_kw=200.0,
            load_kw=100.0,
            hour_of_day=12.0,  # 8 AM - 4 PM
        )
        # Should charge (negative commands)
        if commands_midday:
            for power in commands_midday.values():
                assert power < 0

        # Evening - outside window
        commands_evening = controller.compute_battery_commands(
            {"675": 1.0},
            pv_generation_kw=200.0,
            load_kw=100.0,
            hour_of_day=18.0,  # After 4 PM
        )
        assert len(commands_evening) == 0  # Should not charge

    def test_get_battery_summary(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test battery summary statistics."""
        controller = BatteryController(sample_config, sample_batteries)

        summary = controller.get_battery_summary()

        assert summary["count"] == 2
        assert summary["total_capacity_kwh"] == 400.0
        assert summary["total_energy_kwh"] == 200.0  # 50% of 400
        assert summary["average_soc"] == 0.5

    def test_voltage_support_prioritized(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test voltage support takes priority over charging."""
        controller = BatteryController(sample_config, sample_batteries)

        # High voltage (should discharge) AND PV excess (would charge)
        # Voltage support should win
        commands = controller.compute_battery_commands(
            {"675": 1.05},  # Over voltage threshold
            pv_generation_kw=200.0,
            load_kw=100.0,
            hour_of_day=12.0,
        )

        # Should discharge for voltage support, not charge
        if commands:
            for power in commands.values():
                assert power > 0  # Discharging, not charging

    def test_pv_excess_threshold(
        self,
        sample_config: BatteryControlConfig,
        sample_batteries: BatteryContainer,
    ) -> None:
        """Test PV excess threshold for charging."""
        controller = BatteryController(sample_config, sample_batteries)

        # PV slightly above load (within threshold)
        commands_low = controller.compute_battery_commands(
            {"675": 1.0},
            pv_generation_kw=115.0,
            load_kw=100.0,
            hour_of_day=12.0,
        )
        # 15% excess is below 20% threshold
        assert len(commands_low) == 0

        # PV well above load (exceeds threshold)
        commands_high = controller.compute_battery_commands(
            {"675": 1.0},
            pv_generation_kw=130.0,
            load_kw=100.0,
            hour_of_day=12.0,
        )
        # 30% excess exceeds 20% threshold
        assert len(commands_high) > 0
