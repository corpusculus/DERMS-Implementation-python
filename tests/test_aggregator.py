"""Unit tests for result aggregation functions."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.aggregator import (
    load_simulation_results,
    load_all_results,
    compare_modes,
    calculate_improvements,
    export_comparison_table,
    export_improvement_table,
    generate_comparison_report,
)


@pytest.fixture
def sample_results_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create sample results CSV
    df = pd.DataFrame({
        "step": range(10),
        "time_min": [i * 5 for i in range(10)],
        "time_h": [i * 5 / 60 for i in range(10)],
        "load_multiplier": [1.0] * 10,
        "pv_multiplier": [0.5 + i * 0.05 for i in range(10)],
        "pv_generation_kw": [50 + i * 10 for i in range(10)],
        "v_min": [0.98 + i * 0.005 for i in range(10)],
        "v_max": [1.02 + i * 0.01 for i in range(10)],
        "v_mean": [1.0 + i * 0.005 for i in range(10)],
        "v_min_bus": [f"bus_{i}" for i in range(10)],
        "v_max_bus": [f"bus_{i}" for i in range(10)],
        "violating_buses_count": [0] * 5 + [1, 2, 3, 4, 5],
        "overvoltage_buses_count": [0] * 5 + [1, 2, 3, 4, 5],
        "undervoltage_buses_count": [0] * 10,
        "losses_kw": [10.0 + i for i in range(10)],
        "control_mode": ["baseline"] * 10,
        "total_q_dispatch_kvar": [0] * 10,
        "total_p_curtailment_kw": [0] * 10,
        "ders_controlled": [0] * 10,
    })

    df.to_csv(results_dir / "qsts_baseline.csv", index=False)

    return results_dir


@pytest.fixture
def sample_heuristic_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with heuristic results."""
    results_dir = tmp_path / "heuristic"
    results_dir.mkdir()

    # Create results with control (fewer violations)
    df = pd.DataFrame({
        "step": range(10),
        "time_min": [i * 5 for i in range(10)],
        "time_h": [i * 5 / 60 for i in range(10)],
        "load_multiplier": [1.0] * 10,
        "pv_multiplier": [0.5 + i * 0.05 for i in range(10)],
        "pv_generation_kw": [50 + i * 10 for i in range(10)],
        "v_min": [0.98 + i * 0.005 for i in range(10)],
        "v_max": [1.01 + i * 0.005 for i in range(10)],  # Lower max
        "v_mean": [1.0 + i * 0.003 for i in range(10)],
        "v_min_bus": [f"bus_{i}" for i in range(10)],
        "v_max_bus": [f"bus_{i}" for i in range(10)],
        "violating_buses_count": [0] * 8 + [1, 1],  # Fewer violations
        "overvoltage_buses_count": [0] * 8 + [1, 1],
        "undervoltage_buses_count": [0] * 10,
        "losses_kw": [10.0 + i for i in range(10)],
        "control_mode": ["heuristic"] * 10,
        "total_q_dispatch_kvar": [0, 0, 0, 5, 10, 15, 20, 15, 10, 5],  # Q control
        "total_p_curtailment_kw": [0, 0, 0, 0, 0, 0, 0, 2, 5, 3],  # P curtailment
        "ders_controlled": [0, 0, 0, 1, 2, 3, 3, 3, 3, 2],
    })

    df.to_csv(results_dir / "qsts_baseline.csv", index=False)

    return results_dir


# ---------------------------------------------------------------------------
# Load Results Tests
# ---------------------------------------------------------------------------


def test_load_simulation_results(sample_results_dir: Path):
    """Test loading simulation results from directory."""
    df = load_simulation_results(sample_results_dir, "test")

    assert len(df) == 10
    assert "v_min" in df.columns
    assert "v_max" in df.columns
    assert "time_h" in df.columns


def test_load_simulation_results_missing_file(tmp_path: Path):
    """Test error when results file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_simulation_results(tmp_path / "nonexistent", "test")


def test_load_simulation_results_missing_columns(tmp_path: Path):
    """Test error when required columns are missing."""
    bad_dir = tmp_path / "bad_results"
    bad_dir.mkdir()

    # Create CSV with missing columns
    pd.DataFrame({"step": [1, 2]}).to_csv(bad_dir / "qsts_baseline.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_simulation_results(bad_dir, "test")


def test_load_all_results(sample_results_dir: Path, sample_heuristic_dir: Path):
    """Test loading results for all modes."""
    results = load_all_results(sample_results_dir, sample_heuristic_dir)

    assert "baseline" in results
    assert "heuristic" in results
    assert len(results["baseline"]) == 10
    assert len(results["heuristic"]) == 10


def test_load_all_results_with_missing_optimization(
    sample_results_dir: Path,
    sample_heuristic_dir: Path,
    tmp_path: Path,
):
    """Test loading all results when optimization is missing."""
    nonexistent = tmp_path / "nonexistent_opt"

    results = load_all_results(
        sample_results_dir,
        sample_heuristic_dir,
        str(nonexistent),
    )

    assert "baseline" in results
    assert "heuristic" in results
    assert "optimization" not in results


# ---------------------------------------------------------------------------
# Compare Modes Tests
# ---------------------------------------------------------------------------


def test_compare_modes(sample_results_dir: Path, sample_heuristic_dir: Path):
    """Test comparing modes."""
    baseline_df = load_simulation_results(sample_results_dir, "baseline")
    heuristic_df = load_simulation_results(sample_heuristic_dir, "heuristic")

    comparison = compare_modes(baseline_df, heuristic_df)

    assert isinstance(comparison, pd.DataFrame)
    assert "baseline" in comparison.index
    assert "heuristic" in comparison.index

    # Check that KPI columns exist
    assert "feeder_violation_minutes" in comparison.columns
    assert "max_voltage" in comparison.columns


def test_compare_modes_with_optimization(
    sample_results_dir: Path,
    sample_heuristic_dir: Path,
    tmp_path: Path,
):
    """Test comparing all three modes."""
    # Create optimization results
    opt_dir = tmp_path / "optimization"
    opt_dir.mkdir()

    df = pd.DataFrame({
        "step": range(10),
        "time_h": [i * 5 / 60 for i in range(10)],
        "v_min": [0.98 + i * 0.005 for i in range(10)],
        "v_max": [1.0 + i * 0.003 for i in range(10)],  # Even lower
        "v_mean": [1.0 + i * 0.002 for i in range(10)],
        "violating_buses_count": [0] * 10,  # No violations
        "overvoltage_buses_count": [0] * 10,
        "undervoltage_buses_count": [0] * 10,
        "losses_kw": [10.0] * 10,
        "pv_generation_kw": [100.0] * 10,
        "total_q_dispatch_kvar": [10, 15, 20, 25, 30, 25, 20, 15, 10, 5],
        "total_p_curtailment_kw": [0, 0, 1, 2, 3, 2, 1, 0, 0, 0],
        "ders_controlled": [3] * 10,
    })
    df.to_csv(opt_dir / "qsts_baseline.csv", index=False)

    baseline_df = load_simulation_results(sample_results_dir, "baseline")
    heuristic_df = load_simulation_results(sample_heuristic_dir, "heuristic")
    opt_df = load_simulation_results(opt_dir, "optimization")

    comparison = compare_modes(baseline_df, heuristic_df, opt_df)

    assert "optimization" in comparison.index


# ---------------------------------------------------------------------------
# Calculate Improvements Tests
# ---------------------------------------------------------------------------


def test_calculate_improvements(sample_results_dir: Path, sample_heuristic_dir: Path):
    """Test calculating improvements vs baseline."""
    baseline_df = load_simulation_results(sample_results_dir, "baseline")
    heuristic_df = load_simulation_results(sample_heuristic_dir, "heuristic")

    improvements = calculate_improvements(baseline_df, heuristic_df)

    assert isinstance(improvements, pd.DataFrame)
    assert "heuristic_improvement_pct" in improvements.index

    # Violations should be reduced (negative improvement = good for violations)
    violation_improvement = improvements.loc["heuristic_improvement_pct", "feeder_violation_minutes"]
    assert violation_improvement < 0  # Negative means improvement


# ---------------------------------------------------------------------------
# Export Tests
# ---------------------------------------------------------------------------


def test_export_comparison_table(sample_results_dir: Path, sample_heuristic_dir: Path, tmp_path: Path):
    """Test exporting comparison table."""
    baseline_df = load_simulation_results(sample_results_dir, "baseline")
    heuristic_df = load_simulation_results(sample_heuristic_dir, "heuristic")

    comparison = compare_modes(baseline_df, heuristic_df)
    output_path = tmp_path / "comparison"

    export_comparison_table(comparison, output_path)

    # Check files were created
    assert (output_path.with_suffix(".csv")).exists()
    assert (output_path.with_suffix(".md")).exists()

    # Check CSV content
    loaded_csv = pd.read_csv(output_path.with_suffix(".csv"), index_col=0)
    assert "baseline" in loaded_csv.index
    assert "heuristic" in loaded_csv.index


def test_export_improvement_table(sample_results_dir: Path, sample_heuristic_dir: Path, tmp_path: Path):
    """Test exporting improvement table."""
    baseline_df = load_simulation_results(sample_results_dir, "baseline")
    heuristic_df = load_simulation_results(sample_heuristic_dir, "heuristic")

    improvements = calculate_improvements(baseline_df, heuristic_df)
    output_path = tmp_path / "improvements"

    export_improvement_table(improvements, output_path)

    # Check files were created
    assert (output_path.with_suffix(".csv")).exists()
    assert (output_path.with_suffix(".md")).exists()


def test_export_markdown_format(sample_results_dir: Path, sample_heuristic_dir: Path, tmp_path: Path):
    """Test Markdown format in exported table."""
    baseline_df = load_simulation_results(sample_results_dir, "baseline")
    heuristic_df = load_simulation_results(sample_heuristic_dir, "heuristic")

    comparison = compare_modes(baseline_df, heuristic_df)
    output_path = tmp_path / "comparison"

    export_comparison_table(comparison, output_path)

    # Read markdown and check structure
    md_content = (output_path.with_suffix(".md")).read_text()

    assert "# KPI Comparison Results" in md_content
    assert "|" in md_content  # Table format
    assert "Metric" in md_content


# ---------------------------------------------------------------------------
# End-to-End Report Tests
# ---------------------------------------------------------------------------


def test_generate_comparison_report(
    sample_results_dir: Path,
    sample_heuristic_dir: Path,
    tmp_path: Path,
):
    """Test generating complete comparison report."""
    output_dir = tmp_path / "output"

    paths = generate_comparison_report(
        str(sample_results_dir),
        str(sample_heuristic_dir),
        None,  # No optimization
        str(output_dir),
    )

    # Check that outputs were created
    assert "comparison_csv" in paths
    assert "comparison_md" in paths
    assert "improvement_csv" in paths
    assert "improvement_md" in paths

    for path_key, path_val in paths.items():
        assert Path(path_val).exists(), f"{path_key} not created"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


def test_compare_empty_dataframes():
    """Test comparison with empty results."""
    empty_df = pd.DataFrame({
        "step": [],
        "time_h": [],
        "v_min": [],
        "v_max": [],
        "violating_buses_count": [],
    })

    # Should not crash, though results may be NaN
    comparison = compare_modes(empty_df, empty_df)

    assert isinstance(comparison, pd.DataFrame)


def test_single_timestep_results(tmp_path: Path):
    """Test with single timestep results."""
    results_dir = tmp_path / "single"
    results_dir.mkdir()

    df = pd.DataFrame({
        "step": [0],
        "time_h": [0.0],
        "v_min": [1.0],
        "v_max": [1.0],
        "violating_buses_count": [0],
        "overvoltage_buses_count": [0],
        "undervoltage_buses_count": [0],
        "losses_kw": [10.0],
        "pv_generation_kw": [100.0],
        "total_q_dispatch_kvar": [0.0],
        "total_p_curtailment_kw": [0.0],
        "ders_controlled": [0],
    })

    df.to_csv(results_dir / "qsts_baseline.csv", index=False)

    loaded = load_simulation_results(results_dir, "single")
    assert len(loaded) == 1
