"""
tests/test_opendss_interface.py
--------------------------------
Smoke tests for the OpenDSS interface module.

If OpenDSSDirect.py is not installed OR the IEEE 13-bus feeder files
are not present, the tests are skipped gracefully.
"""

import pathlib

import pytest

# --- Detect whether OpenDSS + feeder files are available ---

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_IEEE13_MASTER = _REPO_ROOT / "feeder_models" / "ieee13" / "IEEE13Nodeckt.dss"

try:
    import opendssdirect  # noqa: F401

    _OPENDSS_INSTALLED = True
except ImportError:
    _OPENDSS_INSTALLED = False

_FEEDER_AVAILABLE = _IEEE13_MASTER.exists()

_SKIP_REASON = []
if not _OPENDSS_INSTALLED:
    _SKIP_REASON.append("OpenDSSDirect.py not installed")
if not _FEEDER_AVAILABLE:
    _SKIP_REASON.append(f"IEEE 13-bus feeder not found at {_IEEE13_MASTER}")

_SKIP_ALL = bool(_SKIP_REASON)
_SKIP_MSG = "; ".join(_SKIP_REASON)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_load_feeder_does_not_raise():
    """load_feeder() should compile the IEEE 13-bus model without error."""
    from src.sim.opendss_interface import load_feeder

    load_feeder(_IEEE13_MASTER)  # should not raise


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_solve_power_flow_converges():
    """solve_power_flow() should converge for the IEEE 13-bus base case."""
    from src.sim.opendss_interface import load_feeder, solve_power_flow

    load_feeder(_IEEE13_MASTER)
    solve_power_flow()  # should not raise RuntimeError("Power flow did not converge")


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_get_bus_voltages_returns_nonempty_dict():
    """get_bus_voltages() should return a non-empty dict after solving."""
    from src.sim.opendss_interface import get_bus_voltages, load_feeder, solve_power_flow

    load_feeder(_IEEE13_MASTER)
    solve_power_flow()
    voltages = get_bus_voltages()

    assert isinstance(voltages, dict), "Expected a dict"
    assert len(voltages) > 0, "Expected at least one bus"


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_bus_voltages_are_physically_reasonable():
    """All per-unit voltages should be within a physically plausible range [0.8, 1.2] pu."""
    from src.sim.opendss_interface import get_bus_voltages, load_feeder, solve_power_flow

    load_feeder(_IEEE13_MASTER)
    solve_power_flow()
    voltages = get_bus_voltages()

    for bus, v in voltages.items():
        assert 0.8 <= v <= 1.2, (
            f"Bus '{bus}' voltage {v:.4f} pu is outside the physically plausible range [0.8, 1.2]"
        )


@pytest.mark.skipif(_SKIP_ALL, reason=_SKIP_MSG)
def test_export_results_creates_csv(tmp_path):
    """export_results() should write a CSV with 'bus' and 'v_pu' columns."""
    from src.sim.opendss_interface import export_results, get_bus_voltages, load_feeder, solve_power_flow

    load_feeder(_IEEE13_MASTER)
    solve_power_flow()
    voltages = get_bus_voltages()

    out_file = tmp_path / "bus_voltages.csv"
    export_results(voltages, out_file)

    assert out_file.exists(), "CSV file was not created"
    content = out_file.read_text()
    assert "bus" in content
    assert "v_pu" in content


def test_import_without_opendss_raises_import_error():
    """Calling interface functions without OpenDSSDirect.py should raise ImportError."""
    import importlib
    import unittest.mock as mock

    # Temporarily patch OPENDSS_AVAILABLE to False
    import src.sim.opendss_interface as iface

    original = iface.OPENDSS_AVAILABLE
    iface.OPENDSS_AVAILABLE = False
    try:
        with pytest.raises(ImportError, match="OpenDSSDirect.py"):
            iface.load_feeder("dummy.dss")
    finally:
        iface.OPENDSS_AVAILABLE = original
