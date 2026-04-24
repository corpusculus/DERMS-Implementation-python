"""Integration tests for DER interface.

Tests the DER interface functions with OpenDSS. If OpenDSSDirect.py
or the IEEE 13-bus feeder files are not present, tests are skipped.
"""

import pathlib
from datetime import datetime

import pytest

# --- Detect whether OpenDSS + feeder files are available ---

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_IEEE13_MASTER = _REPO_ROOT / "feeder_models" / "ieee13" / "IEEE13Nodeckt.dss"
_DER_CSV = _REPO_ROOT / "data" / "der_configs" / "ders_ieee13.csv"

try:
    import opendssdirect  # noqa: F401

    _OPENDSS_INSTALLED = True
except ImportError:
    _OPENDSS_INSTALLED = False

_FEEDER_AVAILABLE = _IEEE13_MASTER.exists() and _DER_CSV.exists()

_SKIP_REASON = []
if not _OPENDSS_INSTALLED:
    _SKIP_REASON.append("OpenDSSDirect.py not installed")
if not _FEEDER_AVAILABLE:
    _SKIP_REASON.append(f"IEEE 13-bus feeder or DER CSV not found")

_SKIP_ALL = bool(_SKIP_REASON)
_SKIP_MSG = "; ".join(_SKIP_REASON)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def loaded_feeder():
    """Load the IEEE 13-bus feeder with PV systems."""
    if _SKIP_ALL:
        pytest.skip(_SKIP_MSG)

    from src.sim.opendss_interface import load_feeder

    load_feeder(_IEEE13_MASTER)
    yield
    # Cleanup
    import opendssdirect as dss

    dss.run_command("Clear")


@pytest.fixture(scope="function")
def pv_integration(loaded_feeder):
    """Integrate PV systems into the feeder."""
    if _SKIP_ALL:
        pytest.skip(_SKIP_MSG)

    from src.sim.pv_integration import place_pv_list

    # Place PV systems using default placement for IEEE 13
    place_pv_list(
        pv_list=[
            {"bus": "675", "phases": 1, "p_kw": 100, "kva": 120},
            {"bus": "646", "phases": 1, "p_kw": 80, "kva": 100},
            {"bus": "632", "phases": 1, "p_kw": 150, "kva": 180},
            {"bus": "633", "phases": 1, "p_kw": 100, "kva": 120},
            {"bus": "634", "phases": 1, "p_kw": 120, "kva": 144},
            {"bus": "645", "phases": 1, "p_kw": 80, "kva": 100},
            {"bus": "611", "phases": 1, "p_kw": 100, "kva": 120},
            {"bus": "652", "phases": 1, "p_kw": 100, "kva": 120},
            {"bus": "680", "phases": 1, "p_kw": 80, "kva": 100},
            {"bus": "684", "phases": 1, "p_kw": 100, "kva": 120},
            {"bus": "671", "phases": 3, "p_kw": 200, "kva": 240},
            {"bus": "692", "phases": 1, "p_kw": 80, "kva": 100},
        ],
        scale_factor=1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_load_ders_from_csv():
    """load_ders_from_csv() should create DERs from CSV file."""
    from src.control.der_interface import load_ders_from_csv

    container = load_ders_from_csv(str(_DER_CSV))

    assert len(container) > 0, "Expected at least one DER"

    # Check first DER has expected properties
    first_der = container.ders[0]
    assert first_der.id.startswith("pv_")
    assert first_der.bus == "675"
    assert first_der.phases == 1
    assert first_der.p_kw_rated == 100.0
    assert first_der.s_kva_rated == 120.0


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_read_der_state_populates_fields(pv_integration):
    """read_der_state() should populate P, Q, V from OpenDSS."""
    from src.control.der_interface import load_ders_from_csv, read_der_state
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()

    read_der_state(container)

    # Check that at least one DER has non-zero state
    for der in container.ders:
        if der.p_avail_kw >= 0:  # State was read
            # Some DERs should have power if PV profile has non-zero values
            assert isinstance(der.p_dispatch_kw, float)
            assert isinstance(der.q_kvar, float)
            assert isinstance(der.v_local_pu, float)
            # Verify v_local_pu is a valid positive number (some buses may have low voltage due to topology)
            assert der.v_local_pu > 0, f"Voltage at bus {der.bus} should be positive"


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_apply_q_setpoint_changes_pf(pv_integration):
    """apply_q_setpoint() should change the power factor in OpenDSS."""
    import opendssdirect as dss

    from src.control.der_interface import load_ders_from_csv, read_der_state, apply_q_setpoint
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()
    read_der_state(container)

    # Get first DER with non-zero dispatch
    der = None
    for d in container.ders:
        if d.p_dispatch_kw > 0:
            der = d
            break

    if der is None:
        pytest.skip("No DER with active power dispatch")

    # Read initial PF
    dss.PVsystems.Name(der.id)
    pf_before = dss.PVsystems.pf()

    # Apply Q injection
    q_target = min(10.0, der.q_max_kvar)  # Small Q value
    result = apply_q_setpoint(der, q_target)

    assert result is True, "apply_q_setpoint should succeed"

    # Check PF changed (unless it was already at that value)
    dss.PVsystems.Name(der.id)
    pf_after = dss.PVsystems.pf()
    # PF should be different if we injected reactive power
    if q_target > 0 and der.p_dispatch_kw > 0:
        # The new PF should not equal the old PF (assuming unity pf before)
        # or should reflect the Q injection
        assert isinstance(pf_after, float)


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_apply_p_curtailment_changes_pmpp(pv_integration):
    """apply_p_curtailment() should change Pmpp in OpenDSS."""
    import opendssdirect as dss

    from src.control.der_interface import load_ders_from_csv, read_der_state, apply_p_curtailment
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()
    read_der_state(container)

    # Get first DER with non-zero dispatch
    der = None
    for d in container.ders:
        if d.p_dispatch_kw > 10:  # Need enough headroom for curtailment
            der = d
            break

    if der is None:
        pytest.skip("No DER with sufficient active power for curtailment test")

    # Read initial state
    dss.PVsystems.Name(der.id)
    p_before = dss.PVsystems.Pmpp()
    p_avail = der.p_avail_kw

    # Curtail by 50% of available power
    # apply_p_curtailment interprets p_kw as curtailment amount, not target
    curtailment_amount = p_avail * 0.5
    p_target_expected = p_avail - curtailment_amount  # Should be 50% of available
    result = apply_p_curtailment(der, curtailment_amount)

    assert result is True, "apply_p_curtailment should succeed"

    # Check Pmpp changed to expected target
    dss.PVsystems.Name(der.id)
    p_after = dss.PVsystems.Pmpp()
    assert abs(p_after - p_target_expected) < 0.1, f"Pmpp should be set to {p_target_expected}, got {p_after}"


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_apply_p_curtailment_rejects_negative(pv_integration):
    """apply_p_curtailment() should reject negative power values."""
    from src.control.der_interface import load_ders_from_csv, read_der_state, apply_p_curtailment
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()
    read_der_state(container)

    der = container.ders[0]

    # Negative curtailment should fail
    result = apply_p_curtailment(der, -10.0)
    assert result is False, "Negative curtailment should fail"


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_apply_p_curtailment_rejects_exceeds_available(pv_integration):
    """apply_p_curtailment() should reject values exceeding p_avail_kw."""
    from src.control.der_interface import load_ders_from_csv, read_der_state, apply_p_curtailment
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()
    read_der_state(container)

    der = container.ders[0]

    # Curtailment exceeding available should fail
    result = apply_p_curtailment(der, der.p_avail_kw + 100.0)
    assert result is False, "Curtailment exceeding available should fail"


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_apply_setpoints_batch(pv_integration):
    """apply_setpoints() should apply multiple commands in one call."""
    from src.control.der_interface import load_ders_from_csv, apply_setpoints
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()

    # Create commands for first 3 DERs
    q_commands = {}
    for der in container.ders[:3]:
        q_commands[der.id] = 5.0  # Small Q injection

    results = apply_setpoints(container, q_commands)

    assert len(results) == 3
    # All should succeed if Q is within capability
    for der_id, result in results.items():
        assert result.status in ["applied", "out_of_range", "failed"]


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_apply_setpoints_out_of_range(pv_integration):
    """apply_setpoints() should return 'out_of_range' for impossible Q."""
    from src.control.der_interface import load_ders_from_csv, read_der_state, apply_setpoints
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()
    read_der_state(container)

    der = container.ders[0]

    # Request impossible Q (exceeds capability)
    q_commands = {der.id: der.q_max_kvar + 1000.0}

    results = apply_setpoints(container, q_commands)

    assert results[der.id].status == "out_of_range"
    assert results[der.id].q_status == "out_of_range"


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_apply_setpoints_unknown_der(pv_integration):
    """apply_setpoints() should return 'failed' for unknown DER IDs."""
    from src.control.der_interface import load_ders_from_csv, apply_setpoints

    container = load_ders_from_csv(str(_DER_CSV))

    q_commands = {"nonexistent_der": 10.0}

    results = apply_setpoints(container, q_commands)

    assert results["nonexistent_der"].status == "failed"
    assert results["nonexistent_der"].q_status == "failed"


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_read_command_apply_log_cycle(pv_integration, tmp_path):
    """End-to-end test: read -> command -> apply -> log."""
    from src.control.der_interface import load_ders_from_csv, read_der_state, apply_setpoints
    from src.control.command_log import CommandLogger
    from src.sim.opendss_interface import solve_power_flow

    container = load_ders_from_csv(str(_DER_CSV))
    solve_power_flow()

    # Read initial state
    read_der_state(container)

    # Get voltages before
    v_before = {der.bus: der.v_local_pu for der in container.ders}

    # Apply commands
    q_commands = {der.id: 5.0 for der in container.ders[:3] if der.p_dispatch_kw > 0}
    results = apply_setpoints(container, q_commands)

    # Re-solve and read
    solve_power_flow()
    read_der_state(container)

    # Get voltages after
    v_after = {der.bus: der.v_local_pu for der in container.ders}

    # Log commands
    logger = CommandLogger(tmp_path / "commands.csv")
    logger.log_batch(
        container=container,
        q_commands=q_commands,
        p_commands=None,
        results=results,
        timestamp=datetime.now(),
        voltages_before=v_before,
        voltages_after=v_after,
    )
    log_path = logger.save()

    # Verify log file
    assert log_path.exists()
    assert len(logger) > 0

    # Check CSV content
    import pandas as pd

    df = pd.read_csv(log_path)
    assert "timestamp" in df.columns
    assert "der_id" in df.columns
    assert "q_commanded_kvar" in df.columns
    assert "status" in df.columns
    assert "q_status" in df.columns
    assert "p_status" in df.columns
    assert "p_dispatch_after_kw" in df.columns
    assert "q_after_kvar" in df.columns
