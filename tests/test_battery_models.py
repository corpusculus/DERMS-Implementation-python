"""Tests for battery models."""

import pytest

from src.control.battery_models import (
    BatteryConfig,
    Battery,
    BatteryContainer,
)


class TestBatteryConfig:
    """Tests for BatteryConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BatteryConfig()

        assert config.capacity_kwh == 200.0
        assert config.power_kw == 100.0
        assert config.efficiency == 0.95
        assert config.soc_init == 0.5
        assert config.soc_min == 0.1
        assert config.soc_max == 0.9
        assert config.charge_threshold_pv_excess_pct == 0.2
        assert config.discharge_voltage_pu == 1.06
        assert config.charge_window_start_h == 8
        assert config.charge_window_end_h == 16

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = BatteryConfig(
            capacity_kwh=500.0,
            power_kw=250.0,
            efficiency=0.90,
            soc_init=0.75,
        )

        assert config.capacity_kwh == 500.0
        assert config.power_kw == 250.0
        assert config.efficiency == 0.90
        assert config.soc_init == 0.75


class TestBattery:
    """Tests for Battery dataclass."""

    def test_battery_creation(self) -> None:
        """Test battery object creation."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
        )

        assert battery.id == "batt_001"
        assert battery.bus == "675"
        assert battery.phases == 1
        assert battery.capacity_kwh == 200.0
        assert battery.power_limit_kw == 100.0
        assert battery.soc == 0.5  # Default
        assert battery.energy_kwh == 100.0  # 50% of 200

    def test_can_charge_within_limits(self) -> None:
        """Test charging check within power and SOC limits."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,  # Has room to charge
        )

        # Can charge within limits
        assert battery.can_charge(-50.0) is True
        assert battery.can_charge(-100.0) is True  # Max power

    def test_cannot_charge_over_power_limit(self) -> None:
        """Test charging check fails when over power limit."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
        )

        assert battery.can_charge(-150.0) is False  # Over power limit

    def test_cannot_charge_over_soc_limit(self) -> None:
        """Test charging check fails when SOC is at max."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.9,  # At max SOC
            soc_max=0.9,
        )

        assert battery.can_charge(-10.0) is False  # Would exceed max SOC

    def test_can_discharge_within_limits(self) -> None:
        """Test discharging check within power and SOC limits."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,  # Has energy to discharge
        )

        assert battery.can_discharge(50.0) is True
        assert battery.can_discharge(100.0) is True  # Max power

    def test_cannot_discharge_over_power_limit(self) -> None:
        """Test discharging check fails when over power limit."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
        )

        assert battery.can_discharge(150.0) is False  # Over power limit

    def test_cannot_discharge_under_soc_limit(self) -> None:
        """Test discharging check fails when SOC is at min."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.1,  # At min SOC
            soc_min=0.1,
        )

        assert battery.can_discharge(10.0) is False  # Would go below min SOC

    def test_update_soc_discharge(self) -> None:
        """Test SOC update during discharge."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,  # 100 kWh
            efficiency=1.0,  # No losses for simplicity
        )

        # Discharge 50 kWh for 1 hour
        battery.update_soc(50.0, hours=1.0)

        # New SOC = (100 - 50) / 200 = 0.25
        assert battery.soc == 0.25
        assert battery.energy_kwh == 50.0
        assert battery.energy_throughput_kwh == 50.0
        assert battery.cycles_equivalent == 50.0 / 200.0

    def test_update_soc_charge(self) -> None:
        """Test SOC update during charge."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,  # 100 kWh
            efficiency=1.0,
        )

        # Charge 50 kWh for 1 hour
        battery.update_soc(-50.0, hours=1.0)

        # New SOC = (100 + 50) / 200 = 0.75
        assert battery.soc == 0.75
        assert battery.energy_kwh == 150.0
        assert battery.energy_throughput_kwh == 50.0

    def test_update_soc_with_efficiency(self) -> None:
        """Test SOC update accounts for efficiency losses."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
            efficiency=0.90,  # 90% efficiency
        )

        # Discharge: actual energy out = power * time * efficiency
        battery.update_soc(100.0, hours=1.0)

        # Energy removed = 100 * 0.9 = 90 kWh
        # New energy = 100 - 90 = 10 kWh
        # New SOC = 10 / 200 = 0.05, but clamped to min (0.1)
        assert battery.soc == battery.soc_min

        # Reset for charge test
        battery.soc = 0.5
        battery.energy_throughput_kwh = 0

        # Charge: energy added = power * time / efficiency
        battery.update_soc(-100.0, hours=1.0)

        # Energy added = 100 / 0.9 = 111.1 kWh
        # New energy = 100 + 111.1 = 211.1 kWh
        # New SOC = 211.1 / 200 = 1.055, but clamped to max (0.9)
        assert battery.soc == battery.soc_max

    def test_update_soc_clamps_to_limits(self) -> None:
        """Test SOC update clamps to min/max limits."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
            soc_min=0.1,
            soc_max=0.9,
            efficiency=1.0,
        )

        # Try to discharge below minimum
        battery.update_soc(150.0, hours=1.0)  # Would go to -25% SOC
        assert battery.soc == battery.soc_min

        # Reset and try to charge above maximum
        battery.soc = 0.5
        battery.update_soc(-150.0, hours=1.0)  # Would go to 125% SOC
        assert battery.soc == battery.soc_max

    def test_energy_headroom(self) -> None:
        """Test energy headroom calculations."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
            soc_min=0.1,
            soc_max=0.9,
        )

        assert battery.charge_headroom_kwh == 0.9 * 200 - 0.5 * 200  # 80 kWh
        assert battery.discharge_headroom_kwh == 0.5 * 200 - 0.1 * 200  # 80 kWh

    def test_max_power_calculations(self) -> None:
        """Test max charge/discharge power based on SOC headroom."""
        battery = Battery(
            id="batt_001",
            bus="675",
            phases=1,
            capacity_kwh=200.0,
            power_limit_kw=100.0,
            soc=0.5,
            soc_min=0.1,
            soc_max=0.9,
            efficiency=0.95,
        )

        # Max power limited by both power rating and SOC headroom
        # With 80 kWh headroom and 95% efficiency, max for 1 hour is:
        # Charge: min(100, 80 * 0.95) = 76 kW
        # Discharge: min(100, 80 / 0.95) = 84.2 kW
        assert 0 < battery.max_charge_power_kw <= 100
        assert 0 < battery.max_discharge_power_kw <= 100


class TestBatteryContainer:
    """Tests for BatteryContainer."""

    def test_empty_container(self) -> None:
        """Test empty battery container."""
        container = BatteryContainer()

        assert len(container) == 0
        assert container.total_capacity_kwh() == 0
        assert container.average_soc() == 0

    def test_container_with_batteries(self) -> None:
        """Test container with multiple batteries."""
        batteries = [
            Battery(
                id=f"batt_{i:03d}",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
                soc=0.5 + i * 0.1,
            )
            for i in range(1, 4)
        ]

        container = BatteryContainer(batteries)

        assert len(container) == 3
        assert container.total_capacity_kwh() == 600.0
        assert container.total_power_kw() == 300.0
        assert container.average_soc() == pytest.approx((0.6 + 0.7 + 0.8) / 3)

    def test_container_getitem(self) -> None:
        """Test getting battery by ID."""
        batteries = [
            Battery(
                id="batt_001",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
            ),
            Battery(
                id="batt_002",
                bus="646",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
            ),
        ]

        container = BatteryContainer(batteries)

        assert container["batt_001"].id == "batt_001"
        assert container["batt_001"].bus == "675"

    def test_container_getitem_key_error(self) -> None:
        """Test KeyError for missing battery ID."""
        container = BatteryContainer([
            Battery(
                id="batt_001",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
            ),
        ])

        with pytest.raises(KeyError, match="Battery not found"):
            _ = container["batt_999"]

    def test_container_contains(self) -> None:
        """Test checking if battery ID exists."""
        container = BatteryContainer([
            Battery(
                id="batt_001",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
            ),
        ])

        assert "batt_001" in container
        assert "batt_002" not in container

    def test_container_by_bus(self) -> None:
        """Test filtering batteries by bus."""
        batteries = [
            Battery(id="batt_001", bus="675", phases=1, capacity_kwh=200, power_limit_kw=100),
            Battery(id="batt_002", bus="646", phases=1, capacity_kwh=200, power_limit_kw=100),
            Battery(id="batt_003", bus="675", phases=1, capacity_kwh=200, power_limit_kw=100),
        ]

        container = BatteryContainer(batteries)

        bus_675 = container.by_bus("675")
        assert len(bus_675) == 2
        assert all(b.bus == "675" for b in bus_675)

        bus_646 = container.by_bus("646")
        assert len(bus_646) == 1

    def test_container_enabled(self) -> None:
        """Test filtering enabled batteries."""
        batteries = [
            Battery(
                id="batt_001",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
                control_enabled=True,
            ),
            Battery(
                id="batt_002",
                bus="646",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
                control_enabled=False,
            ),
        ]

        container = BatteryContainer(batteries)

        enabled = container.enabled()
        assert len(enabled) == 1
        assert enabled[0].id == "batt_001"

        enabled_ids = container.enabled_ids()
        assert enabled_ids == ["batt_001"]

    def test_container_total_energy(self) -> None:
        """Test total stored energy calculation."""
        batteries = [
            Battery(
                id=f"batt_{i:03d}",
                bus="675",
                phases=1,
                capacity_kwh=200.0,
                power_limit_kw=100.0,
                soc=0.5,
            )
            for i in range(1, 4)
        ]

        container = BatteryContainer(batteries)

        # Each has 100 kWh (50% of 200), total 300 kWh
        assert container.total_energy_kwh() == 300.0
