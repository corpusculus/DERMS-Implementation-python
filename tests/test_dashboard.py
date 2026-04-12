"""Tests for the Plotly dashboard module."""

import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

from src.analysis.dashboard import (
    plot_voltage_envelope_plotly,
    plot_voltage_heatmap,
    plot_q_dispatch_plotly,
    plot_p_curtailment_plotly,
    plot_battery_power_plotly,
    plot_battery_soc_plotly,
    create_kpi_cards,
    create_dashboard,
)


@pytest.fixture
def sample_results() -> pd.DataFrame:
    """Create sample QSTS results for testing."""
    np = pytest.importorskip("numpy")

    time_steps = 288  # 24 hours at 5-minute steps
    time_h = [i * 5 / 60 for i in range(time_steps)]

    return pd.DataFrame({
        "step": range(time_steps),
        "time_min": [i * 5 for i in range(time_steps)],
        "time_h": time_h,
        "v_min": 0.98 + 0.01 * np.sin([i * 0.1 for i in range(time_steps)]),
        "v_max": 1.02 + 0.02 * np.cos([i * 0.1 for i in range(time_steps)]),
        "v_mean": 1.0 + 0.005 * np.sin([i * 0.1 for i in range(time_steps)]),
        "total_q_dispatch_kvar": [10 * max(0, np.sin(i * 0.1)) for i in range(time_steps)],
        "total_p_curtailment_kw": [5 * max(0, np.sin(i * 0.1 - 1)) for i in range(time_steps)],
        "pv_generation_kw": [100 * max(0, np.sin(i * 0.1)) for i in range(time_steps)],
        "violating_buses_count": [0] * time_steps,
    })


@pytest.fixture
def sample_results_with_buses(sample_results: pd.DataFrame) -> pd.DataFrame:
    """Add per-bus voltage data to sample results."""
    # Add some per-bus voltage columns for buses that occasionally violate
    result = sample_results.copy()
    np = pytest.importorskip("numpy")

    for i, bus in enumerate(["675", "646", "632"]):
        result[f"overv_{bus}"] = (
            1.03 + 0.02 * np.sin([j * 0.1 + i for j in range(len(result))])
        )

    return result


@pytest.fixture
def sample_battery_results(sample_results: pd.DataFrame) -> pd.DataFrame:
    """Add battery metrics to sample results."""
    result = sample_results.copy()
    np = pytest.importorskip("numpy")

    result["total_battery_power_kw"] = [
        50 * np.sin(i * 0.1 - 0.5) for i in range(len(result))
    ]
    result["battery_soc_mean"] = [
        0.5 + 0.3 * np.sin(i * 0.1 - 1) for i in range(len(result))
    ]
    result["battery_count"] = [2] * len(result)
    result["battery_energy_kwh"] = [
        100 * (0.5 + 0.3 * np.sin(i * 0.1 - 1)) for i in range(len(result))
    ]

    return result


def test_plot_voltage_envelope_plotly(sample_results: pd.DataFrame) -> None:
    """Test voltage envelope plot creation."""
    fig = plot_voltage_envelope_plotly(sample_results)

    assert fig is not None
    assert fig.layout.title.text == "Voltage Envelope - 24 Hour QSTS"
    assert len(fig.data) >= 3  # min, mean, max traces


def test_plot_voltage_heatmap_no_data(sample_results: pd.DataFrame) -> None:
    """Test voltage heatmap with no per-bus data."""
    fig = plot_voltage_heatmap(sample_results)

    assert fig is not None
    # Should show "No per-bus voltage data available" message
    assert "No Data" in fig.layout.title.text


def test_plot_voltage_heatmap_with_data(sample_results_with_buses: pd.DataFrame) -> None:
    """Test voltage heatmap with per-bus data."""
    fig = plot_voltage_heatmap(sample_results_with_buses)

    assert fig is not None
    assert len(fig.data) == 1  # One heatmap trace
    assert fig.data[0].type == "heatmap"


def test_plot_q_dispatch_plotly(sample_results: pd.DataFrame) -> None:
    """Test reactive power dispatch plot creation."""
    fig = plot_q_dispatch_plotly(sample_results)

    assert fig is not None
    assert fig.layout.title.text == "Reactive Power Dispatch Timeline"
    assert len(fig.data) == 1


def test_plot_p_curtailment_plotly(sample_results: pd.DataFrame) -> None:
    """Test active power curtailment plot creation."""
    fig = plot_p_curtailment_plotly(sample_results)

    assert fig is not None
    assert fig.layout.title.text == "Active Power Curtailment Timeline"
    assert len(fig.data) >= 1


def test_plot_battery_power_plotly(sample_results: pd.DataFrame) -> None:
    """Test battery power plot with no battery data."""
    fig = plot_battery_power_plotly(sample_results)

    assert fig is not None
    assert "No Data" in fig.layout.title.text


def test_plot_battery_power_plotly_with_data(sample_battery_results: pd.DataFrame) -> None:
    """Test battery power plot with battery data."""
    fig = plot_battery_power_plotly(sample_battery_results)

    assert fig is not None
    assert fig.layout.title.text == "Battery Power Timeline"
    assert len(fig.data) == 1


def test_plot_battery_soc_plotly(sample_results: pd.DataFrame) -> None:
    """Test battery SOC plot with no battery data."""
    fig = plot_battery_soc_plotly(sample_results)

    assert fig is not None
    assert "No Data" in fig.layout.title.text


def test_plot_battery_soc_plotly_with_data(sample_battery_results: pd.DataFrame) -> None:
    """Test battery SOC plot with battery data."""
    fig = plot_battery_soc_plotly(sample_battery_results)

    assert fig is not None
    assert fig.layout.title.text == "Battery State of Charge"
    assert len(fig.data) == 1


def test_create_kpi_cards_basic() -> None:
    """Test KPI card creation with basic data."""
    controlled_kpis = {
        "max_voltage": 1.042,
        "min_voltage": 0.982,
        "feeder_violation_minutes": 0,
        "total_curtailed_energy_kwh": 12.5,
        "total_reactive_energy_kvarh": 145.2,
    }

    html = create_kpi_cards(
        baseline_kpis=None,
        controlled_kpis=controlled_kpis,
        battery_kpis=None,
    )

    assert isinstance(html, str)
    assert len(html) > 0
    assert "1.042" in html
    assert "No violations" in html
    assert "12.5" in html


def test_create_kpi_cards_with_battery() -> None:
    """Test KPI card creation with battery data."""
    controlled_kpis = {
        "max_voltage": 1.042,
        "min_voltage": 0.982,
    }
    battery_kpis = {
        "energy_throughput_kwh": 85.3,
        "cycles_equivalent": 0.43,
        "soc_final": 0.62,
    }

    html = create_kpi_cards(
        baseline_kpis=None,
        controlled_kpis=controlled_kpis,
        battery_kpis=battery_kpis,
    )

    assert isinstance(html, str)
    assert "85.3" in html
    assert "0.43" in html
    assert "62%" in html


def test_create_dashboard(tmp_path: pathlib.Path, sample_results: pd.DataFrame) -> None:
    """Test full dashboard creation."""
    # Write sample results to CSV
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    csv_path = results_dir / "qsts_baseline.csv"
    sample_results.to_csv(csv_path, index=False)

    # Generate dashboard
    output_path = create_dashboard(results_dir)

    assert pathlib.Path(output_path).exists()

    # Check HTML content
    html_content = pathlib.Path(output_path).read_text()
    assert "<!DOCTYPE html>" in html_content
    assert "DERMS QSTS Dashboard" in html_content
    assert "voltage-plot" in html_content


def test_create_dashboard_with_battery(
    tmp_path: pathlib.Path,
    sample_battery_results: pd.DataFrame,
) -> None:
    """Test dashboard creation with battery data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    csv_path = results_dir / "qsts_battery.csv"
    sample_battery_results.to_csv(csv_path, index=False)

    output_path = create_dashboard(results_dir)

    html_content = pathlib.Path(output_path).read_text()
    assert "Battery Power" in html_content
    assert "battery-power-plot" in html_content
