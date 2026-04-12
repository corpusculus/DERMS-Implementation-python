"""Unit tests for KPI calculation functions."""

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.kpis import (
    calculate_voltage_kpis,
    calculate_control_kpis,
    calculate_system_kpis,
    calculate_all_kpis,
    compare_kpi_dicts,
    format_kpis_for_display,
)


@pytest.fixture
def sample_results() -> pd.DataFrame:
    """Create sample QSTS results DataFrame for testing."""
    return pd.DataFrame({
        "step": range(10),
        "time_min": [i * 5 for i in range(10)],
        "time_h": [i * 5 / 60 for i in range(10)],
        "v_min": [0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
        "v_max": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
        "v_mean": [0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08],
        "violating_buses_count": [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
        "overvoltage_buses_count": [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
        "undervoltage_buses_count": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "losses_kw": [10.0 + i for i in range(10)],
        "pv_generation_kw": [0, 10, 50, 100, 150, 200, 180, 150, 100, 50],
        "total_q_dispatch_kvar": [0, 0, 0, 0, 0, 10, 30, 50, 70, 90],
        "total_p_curtailment_kw": [0, 0, 0, 0, 0, 0, 5, 15, 25, 35],
        "ders_controlled": [0, 0, 0, 0, 0, 2, 3, 3, 3, 3],
    })


@pytest.fixture
def clean_results() -> pd.DataFrame:
    """Create results with no violations."""
    return pd.DataFrame({
        "step": range(10),
        "time_h": [i * 5 / 60 for i in range(10)],
        "v_min": [0.98] * 10,
        "v_max": [1.02] * 10,
        "v_mean": [1.0] * 10,
        "violating_buses_count": [0] * 10,
        "overvoltage_buses_count": [0] * 10,
        "undervoltage_buses_count": [0] * 10,
        "losses_kw": [10.0] * 10,
        "pv_generation_kw": [100.0] * 10,
        "total_q_dispatch_kvar": [0.0] * 10,
        "total_p_curtailment_kw": [0.0] * 10,
        "ders_controlled": [0] * 10,
    })


# ---------------------------------------------------------------------------
# Voltage KPI Tests
# ---------------------------------------------------------------------------


def test_voltage_kpis_no_violations(clean_results: pd.DataFrame):
    """Test voltage KPIs when there are no violations."""
    kpis = calculate_voltage_kpis(clean_results, v_min=0.95, v_max=1.05)

    assert kpis["feeder_violation_minutes"] == 0
    assert kpis["bus_violation_minutes"] == 0
    assert kpis["max_voltage"] == 1.02
    assert kpis["min_voltage"] == 0.98
    assert kpis["buses_exceeding_limit"] == 0
    assert kpis["voltage_severity_score"] == 0.0


def test_voltage_kpis_with_violations(sample_results: pd.DataFrame):
    """Test voltage KPIs with violations."""
    kpis = calculate_voltage_kpis(sample_results, v_min=0.95, v_max=1.05)

    # 4 timesteps with violations (steps 6-9)
    assert kpis["feeder_violation_minutes"] == 4 * 5  # 20 minutes

    # Max voltage should be highest value
    assert kpis["max_voltage"] == 1.09

    # Should have some buses exceeding limit
    assert kpis["buses_exceeding_limit"] > 0

    # Severity score should be positive
    assert kpis["voltage_severity_score"] > 0


def test_voltage_kpis_missing_columns():
    """Test error when required columns are missing."""
    bad_df = pd.DataFrame({"step": [1, 2, 3]})

    with pytest.raises(ValueError, match="missing required columns"):
        calculate_voltage_kpis(bad_df)


def test_voltage_kpis_avg_deviation(clean_results: pd.DataFrame):
    """Test average voltage deviation calculation."""
    kpis = calculate_voltage_kpis(clean_results)

    # All values are 1.0, so deviation should be 0
    assert kpis["avg_voltage_deviation"] == 0.0


def test_voltage_kpis_custom_limits(sample_results: pd.DataFrame):
    """Test with custom voltage limits."""
    kpis = calculate_voltage_kpis(sample_results, v_min=0.9, v_max=1.1)

    # With wider limits, fewer violations
    assert kpis["feeder_violation_minutes"] >= 0


# ---------------------------------------------------------------------------
# Control KPI Tests
# ---------------------------------------------------------------------------


def test_control_kpis_no_control(clean_results: pd.DataFrame):
    """Test control KPIs when no control is applied."""
    kpis = calculate_control_kpis(clean_results)

    assert kpis["total_reactive_energy_kvarh"] == 0.0
    assert kpis["peak_reactive_dispatch_kvar"] == 0.0
    assert kpis["active_control_timesteps"] == 0
    assert kpis["total_curtailed_energy_kwh"] == 0.0
    assert kpis["curtailed_energy_pct"] == 0.0


def test_control_kpis_with_control(sample_results: pd.DataFrame):
    """Test control KPIs with active control."""
    kpis = calculate_control_kpis(sample_results)

    # Should have reactive energy
    time_step_hours = 5 / 60
    expected_q_kvarh = sample_results["total_q_dispatch_kvar"].sum() * time_step_hours
    assert math.isclose(kpis["total_reactive_energy_kvarh"], expected_q_kvarh, rel_tol=1e-5)

    # Should have curtailment energy
    expected_p_kwh = sample_results["total_p_curtailment_kw"].sum() * time_step_hours
    assert math.isclose(kpis["total_curtailed_energy_kwh"], expected_p_kwh, rel_tol=1e-5)

    # Peak should be max value
    assert kpis["peak_reactive_dispatch_kvar"] == 90.0

    # Active control timesteps (Q > 0 or P > 0 or ders_controlled > 0)
    # In sample data: Q > 0 at steps 5-9 (5 values), P > 0 at steps 6-9 (4 values)
    # So any control active = 5 timesteps
    assert kpis["active_control_timesteps"] == 5  # Steps 5-9 have Q or P control


def test_control_kpis_curtailment_percentage(sample_results: pd.DataFrame):
    """Test curtailment percentage calculation."""
    kpis = calculate_control_kpis(sample_results)

    # Curtailment % should be positive
    assert kpis["curtailed_energy_pct"] > 0

    # Should be less than 100%
    assert kpis["curtailed_energy_pct"] < 100


def test_control_kpis_no_pv_column():
    """Test when PV generation column is missing."""
    df = pd.DataFrame({
        "step": [1, 2],
        "time_h": [0, 1],
        "v_min": [1.0, 1.0],
        "v_max": [1.0, 1.0],
        "violating_buses_count": [0, 0],
        "total_q_dispatch_kvar": [10, 20],
    })

    kpis = calculate_control_kpis(df)

    # Should not crash, curtailment % should be 0
    assert kpis["curtailed_energy_pct"] == 0.0


# ---------------------------------------------------------------------------
# System KPI Tests
# ---------------------------------------------------------------------------


def test_system_kpis(clean_results: pd.DataFrame):
    """Test system KPIs calculation."""
    kpis = calculate_system_kpis(clean_results)

    time_step_hours = 5 / 60

    # Losses
    expected_loss = clean_results["losses_kw"].sum() * time_step_hours
    assert math.isclose(kpis["total_losses_kwh"], expected_loss, rel_tol=1e-5)

    # PV generation
    expected_pv = clean_results["pv_generation_kw"].sum() * time_step_hours
    assert math.isclose(kpis["total_pv_generation_kwh"], expected_pv, rel_tol=1e-5)

    # Peak PV
    assert kpis["peak_pv_generation_kw"] == 100.0


def test_system_kpis_missing_columns():
    """Test system KPIs with missing columns."""
    df = pd.DataFrame({
        "step": [1, 2],
        "time_h": [0, 1],
        "v_min": [1.0, 1.0],
        "v_max": [1.0, 1.0],
        "violating_buses_count": [0, 0],
    })

    kpis = calculate_system_kpis(df)

    # All values should be 0 or default
    assert kpis["total_losses_kwh"] == 0.0
    assert kpis["total_pv_generation_kwh"] == 0.0


# ---------------------------------------------------------------------------
# Combined KPI Tests
# ---------------------------------------------------------------------------


def test_calculate_all_kpis(sample_results: pd.DataFrame):
    """Test combined KPI calculation."""
    kpis = calculate_all_kpis(sample_results)

    # Should contain all voltage KPIs
    assert "feeder_violation_minutes" in kpis
    assert "max_voltage" in kpis
    assert "min_voltage" in kpis

    # Should contain all control KPIs
    assert "total_reactive_energy_kvarh" in kpis
    assert "total_curtailed_energy_kwh" in kpis

    # Should contain all system KPIs
    assert "total_losses_kwh" in kpis
    assert "total_pv_generation_kwh" in kpis


# ---------------------------------------------------------------------------
# Comparison Tests
# ---------------------------------------------------------------------------


def test_compare_kpi_dicts():
    """Test KPI comparison between two scenarios."""
    baseline = {
        "feeder_violation_minutes": 100,
        "total_curtailed_energy_kwh": 50,
    }

    controlled = {
        "feeder_violation_minutes": 20,  # 80% reduction
        "total_curtailed_energy_kwh": 10,  # 80% reduction
    }

    comparison = compare_kpi_dicts(baseline, controlled)

    # 80% improvement means controlled - baseline = -80
    assert math.isclose(comparison["feeder_violation_minutes"], -80.0)
    assert math.isclose(comparison["total_curtailed_energy_kwh"], -80.0)


def test_compare_kpi_dicts_zero_baseline():
    """Test comparison when baseline value is zero."""
    baseline = {"feeder_violation_minutes": 0}
    controlled = {"feeder_violation_minutes": 0}

    comparison = compare_kpi_dicts(baseline, controlled)

    # Should be 0 when both are 0
    assert comparison["feeder_violation_minutes"] == 0.0


def test_compare_kpi_dicts_missing_keys():
    """Test comparison with non-overlapping keys."""
    baseline = {"metric_a": 100}
    controlled = {"metric_b": 50}

    comparison = compare_kpi_dicts(baseline, controlled)

    # Empty since no overlapping keys
    assert len(comparison) == 0


# ---------------------------------------------------------------------------
# Formatting Tests
# ---------------------------------------------------------------------------


def test_format_kpis_for_display():
    """Test KPI formatting for display."""
    kpis = {
        "feeder_violation_minutes": 100,
        "max_voltage": 1.05,
        "curtailed_energy_pct": 12.5,
        "total_losses_kwh": 123.456,
    }

    formatted = format_kpis_for_display(kpis)

    assert formatted["feeder_violation_minutes"] == "100 min"
    assert "1.05" in formatted["max_voltage"]
    assert formatted["curtailed_energy_pct"] == "12.50%"


def test_format_kpis_non_numeric():
    """Test formatting with non-numeric values."""
    kpis = {"status": "complete", "value": 100}

    formatted = format_kpis_for_display(kpis)

    assert formatted["status"] == "complete"
    assert "100" in formatted["value"]


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


def test_empty_results():
    """Test KPI calculation with empty DataFrame."""
    df = pd.DataFrame({
        "step": [],
        "time_h": [],
        "v_min": [],
        "v_max": [],
        "violating_buses_count": [],
    })

    kpis = calculate_voltage_kpis(df)

    # Should handle gracefully
    assert kpis["feeder_violation_minutes"] == 0
    # Max/min of empty series will be NaN, but function should return floats
