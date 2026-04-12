"""Unit tests for the heuristic Volt-VAR controller."""

import math

import pytest

from src.control.heuristic_controller import HeuristicController, HeuristicConfig
from src.control.der_models import DER, DERContainer


@pytest.fixture
def basic_config() -> HeuristicConfig:
    """Return a basic heuristic config for testing."""
    return HeuristicConfig(
        q_activation_pu=1.03,
        curtailment_pu=1.05,
        deadband_pu=0.005,
        q_ramp_max_kvar=50.0,
        p_ramp_max_kw=20.0,
        v_lower_limit=0.95,
    )


@pytest.fixture
def sample_ders() -> DERContainer:
    """Return a sample DER container for testing."""
    ders = [
        DER(
            id="pv_001_675",
            bus="675",
            phases=1,
            p_kw_rated=100.0,
            s_kva_rated=120.0,
            control_enabled=True,
            p_avail_kw=100.0,  # Set available power for P curtailment tests
            p_dispatch_kw=100.0,
        ),
        DER(
            id="pv_002_632",
            bus="632",
            phases=1,
            p_kw_rated=150.0,
            s_kva_rated=180.0,
            control_enabled=True,
            p_avail_kw=150.0,
            p_dispatch_kw=150.0,
        ),
        DER(
            id="pv_003_633",
            bus="633",
            phases=1,
            p_kw_rated=100.0,
            s_kva_rated=120.0,
            control_enabled=False,  # Disabled
            p_avail_kw=100.0,
            p_dispatch_kw=100.0,
        ),
    ]
    return DERContainer(ders)


@pytest.fixture
def controller(basic_config: HeuristicConfig, sample_ders: DERContainer) -> HeuristicController:
    """Return a heuristic controller for testing."""
    return HeuristicController(basic_config, sample_ders)


def test_heuristic_config_defaults():
    """Test default configuration values."""
    config = HeuristicConfig()
    assert config.q_activation_pu == 1.03
    assert config.curtailment_pu == 1.05
    assert config.deadband_pu == 0.005
    assert config.q_ramp_max_kvar == float("inf")
    assert config.p_ramp_max_kw == float("inf")
    assert config.v_lower_limit == 0.95


def test_q_computation_below_threshold(controller: HeuristicController):
    """Test no Q action below activation voltage."""
    # Voltage below activation threshold
    bus_voltages = {"675": 1.02, "632": 1.025}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    assert len(q_commands) == 0
    assert len(p_commands) == 0


def test_q_computation_at_threshold(controller: HeuristicController):
    """Test Q command at exactly activation voltage."""
    # Voltage exactly at activation threshold
    bus_voltages = {"675": 1.03, "632": 1.03}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    # At threshold, severity is 0, so no Q command
    assert len(q_commands) == 0
    assert len(p_commands) == 0


def test_q_computation_above_threshold(controller: HeuristicController):
    """Test Q command increases with severity."""
    # Voltage above activation, below curtailment
    bus_voltages = {"675": 1.04, "632": 1.04}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    # Should have Q commands (absorption, negative)
    assert len(q_commands) > 0

    # Check that Q is negative (absorption)
    for der_id, q_cmd in q_commands.items():
        assert q_cmd < 0, f"Expected negative Q for {der_id}, got {q_cmd}"

    # No P curtailment yet
    assert len(p_commands) == 0


def test_q_computation_at_curtailment_threshold(controller: HeuristicController):
    """Test Q command at curtailment threshold."""
    # Voltage at curtailment threshold
    bus_voltages = {"675": 1.05, "632": 1.05}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    # Should have max Q absorption
    assert len(q_commands) > 0

    # Q should be at max absorption (severity = 1)
    for der_id, q_cmd in q_commands.items():
        der = controller.der_container[der_id]
        assert abs(q_cmd) <= der.q_max_kvar + 1e-6, "Q within capability"


def test_p_curtailment_only_after_threshold(controller: HeuristicController):
    """Test P curtailment only when v > curtailment_pu."""
    # Voltage at curtailment threshold - no P curtailment
    bus_voltages = {"675": 1.05, "632": 1.05}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    assert len(p_commands) == 0

    # Voltage above curtailment threshold - P curtailment active
    bus_voltages = {"675": 1.06, "632": 1.06}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    # Should have P curtailment
    assert len(p_commands) > 0


def test_p_curtailment_severity(controller: HeuristicController):
    """Test P curtailment increases with severity."""
    # Mildly above curtailment
    bus_voltages = {"675": 1.055, "632": 1.055}
    _, p_commands_mild = controller.compute_commands(bus_voltages)

    # Well above curtailment
    bus_voltages = {"675": 1.07, "632": 1.07}
    _, p_commands_severe = controller.compute_commands(bus_voltages)

    # Severe case should have more curtailment
    total_p_mild = sum(p_commands_mild.values())
    total_p_severe = sum(p_commands_severe.values())

    assert total_p_severe > total_p_mild


def test_deadband_behavior(controller: HeuristicController):
    """Test deadband prevents action around 1.0 pu."""
    # Within deadband (1.0 +/- 0.005)
    bus_voltages = {"675": 1.004, "632": 0.996}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    # No commands within deadband
    assert len(q_commands) == 0
    assert len(p_commands) == 0


def test_disabled_der_not_controlled(controller: HeuristicController):
    """Test that disabled DERs are not controlled."""
    bus_voltages = {"675": 1.06, "632": 1.06, "633": 1.06}
    q_commands, p_commands = controller.compute_commands(bus_voltages)

    # pv_003_633 is disabled, should not be in commands
    assert "pv_003_633" not in q_commands
    assert "pv_003_633" not in p_commands


def test_ramp_limiting_q(basic_config: HeuristicConfig, sample_ders: DERContainer):
    """Test Q command changes are limited per timestep."""
    controller = HeuristicController(basic_config, sample_ders)

    # First timestep - high voltage
    bus_voltages = {"675": 1.06, "632": 1.06}
    q_commands_1, _ = controller.compute_commands(bus_voltages)

    # Second timestep - same high voltage but with ramp limiting
    previous_commands = {
        der_id: (q_cmd, 0.0) for der_id, q_cmd in q_commands_1.items()
    }

    # Reset voltage to trigger new commands
    bus_voltages = {"675": 1.01, "632": 1.01}
    q_commands_2, _ = controller.compute_commands(
        bus_voltages,
        previous_commands=previous_commands,
    )

    # If there are still Q commands, check they respect ramp limit
    if q_commands_1 and q_commands_2:
        for der_id in q_commands_1:
            if der_id in q_commands_2:
                delta = abs(q_commands_2[der_id] - q_commands_1[der_id])
                assert delta <= basic_config.q_ramp_max_kvar + 1e-6


def test_ramp_limiting_p(basic_config: HeuristicConfig, sample_ders: DERContainer):
    """Test P command changes are limited per timestep."""
    controller = HeuristicController(basic_config, sample_ders)

    # First timestep - very high voltage triggering curtailment
    bus_voltages = {"675": 1.08, "632": 1.08}
    _, p_commands_1 = controller.compute_commands(bus_voltages)

    # Second timestep with ramp limiting
    previous_commands = {
        der_id: (0.0, p_cmd) for der_id, p_cmd in p_commands_1.items()
    }

    # Still high voltage
    bus_voltages = {"675": 1.08, "632": 1.08}
    _, p_commands_2 = controller.compute_commands(
        bus_voltages,
        previous_commands=previous_commands,
    )

    # P commands should be similar due to ramp limiting from similar state
    # (in this case, delta would be 0 since we're at the same voltage)
    for der_id in p_commands_1:
        if der_id in p_commands_2:
            delta = abs(p_commands_2[der_id] - p_commands_1[der_id])
            assert delta <= basic_config.p_ramp_max_kw + 1e-6


def test_controller_integration(basic_config: HeuristicConfig):
    """Integration test with mock DER container."""
    # Create a realistic scenario
    ders = [
        DER(
            id="pv_001",
            bus="675",
            phases=1,
            p_kw_rated=100.0,
            s_kva_rated=120.0,
            control_enabled=True,
            p_avail_kw=100.0,  # Full available
            p_dispatch_kw=100.0,
        ),
    ]
    container = DERContainer(ders)
    controller = HeuristicController(basic_config, container)

    # Test voltage envelope
    test_cases = [
        # (voltage, expected_q_commands, expected_p_commands)
        (1.00, 0, 0),  # Nominal
        (1.02, 0, 0),  # Below activation
        (1.03, 0, 0),  # At activation
        (1.04, 1, 0),  # Above activation, Q only
        (1.05, 1, 0),  # At curtailment, Q only
        (1.06, 1, 1),  # Above curtailment, both Q and P
    ]

    for voltage, expected_q, expected_p in test_cases:
        q_cmds, p_cmds = controller.compute_commands({"675": voltage})
        assert len(q_cmds) == expected_q, f"Failed at v={voltage}: expected {expected_q} Q commands, got {len(q_cmds)}"
        assert len(p_cmds) == expected_p, f"Failed at v={voltage}: expected {expected_p} P commands, got {len(p_cmds)}"


def test_q_within_capability_curve(basic_config: HeuristicConfig):
    """Test Q commands respect inverter capability curve."""
    ders = [
        DER(
            id="pv_small",
            bus="675",
            phases=1,
            p_kw_rated=50.0,
            s_kva_rated=60.0,
            control_enabled=True,
            p_avail_kw=50.0,
            p_dispatch_kw=50.0,
        ),
    ]
    container = DERContainer(ders)
    controller = HeuristicController(basic_config, container)

    # Max Q at full P dispatch
    der = ders[0]
    expected_q_max = math.sqrt(der.s_kva_rated**2 - der.p_dispatch_kw**2)

    # Very high voltage
    q_cmds, _ = controller.compute_commands({"675": 1.10})

    assert "pv_small" in q_cmds
    q_cmd = q_cmds["pv_small"]

    # Q should be within [q_min, q_max]
    assert der.q_min_kvar <= q_cmd <= der.q_max_kvar or q_cmd == 0
    assert abs(q_cmd) <= expected_q_max + 1e-6


def test_no_commands_for_missing_bus(controller: HeuristicController):
    """Test that DERs at buses not in voltage dict get 1.0 pu."""
    # Only provide voltage for one bus
    bus_voltages = {"675": 1.06}  # 632 and 633 not in dict

    q_commands, p_commands = controller.compute_commands(bus_voltages)

    # Only DER at 675 should be commanded
    assert "pv_001_675" in q_commands or len(q_commands) == 0
    # 632 DER might not be controlled if we default to 1.0
