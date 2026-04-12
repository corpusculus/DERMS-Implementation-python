"""
src/analysis/aggregator.py
---------------------------
Result aggregation and comparison for multiple control modes.

Provides functions to:
- Load simulation results from CSV files
- Compare KPIs across baseline, heuristic, and optimization modes
- Export comparison tables to CSV and Markdown
"""

import pathlib
from typing import Any

import pandas as pd

from src.analysis.kpis import (
    calculate_all_kpis,
    calculate_voltage_kpis,
    calculate_control_kpis,
    calculate_system_kpis,
    compare_kpi_dicts,
    format_kpis_for_display,
)


# ---------------------------------------------------------------------------
# Result Loading
# ---------------------------------------------------------------------------


def load_simulation_results(
    results_dir: pathlib.Path | str,
    mode: str,
) -> pd.DataFrame:
    """Load simulation results from a directory.

    Args:
        results_dir: Directory containing qsts_baseline.csv
        mode: Control mode name (for error messages)

    Returns:
        DataFrame with simulation results

    Raises:
        FileNotFoundError: If results file doesn't exist
        ValueError: If required columns are missing
    """
    results_path = pathlib.Path(results_dir) / "qsts_baseline.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found for {mode} mode: {results_path}"
        )

    df = pd.read_csv(results_path)

    # Validate required columns
    required_cols = ["step", "time_h", "v_min", "v_max"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Results for {mode} mode missing required columns: {missing}"
        )

    return df


def load_all_results(
    baseline_dir: pathlib.Path | str,
    heuristic_dir: pathlib.Path | str,
    optimization_dir: pathlib.Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load results for all available modes.

    Args:
        baseline_dir: Directory for baseline results
        heuristic_dir: Directory for heuristic results
        optimization_dir: Optional directory for optimization results

    Returns:
        Dictionary mapping mode name to results DataFrame
    """
    results: dict[str, pd.DataFrame] = {}

    results["baseline"] = load_simulation_results(baseline_dir, "baseline")
    results["heuristic"] = load_simulation_results(heuristic_dir, "heuristic")

    if optimization_dir is not None:
        try:
            results["optimization"] = load_simulation_results(
                optimization_dir, "optimization"
            )
        except FileNotFoundError:
            print(f"Warning: Optimization results not found at {optimization_dir}")

    return results


# ---------------------------------------------------------------------------
# KPI Comparison
# ---------------------------------------------------------------------------


def compare_modes(
    baseline_results: pd.DataFrame,
    heuristic_results: pd.DataFrame,
    optimization_results: pd.DataFrame | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> pd.DataFrame:
    """Calculate KPIs for each mode and return comparison DataFrame.

    Args:
        baseline_results: Baseline mode QSTS results
        heuristic_results: Heuristic mode QSTS results
        optimization_results: Optional optimization mode QSTS results
        v_min: Lower voltage limit
        v_max: Upper voltage limit

    Returns:
        DataFrame with modes as rows and KPIs as columns
    """
    # Calculate KPIs for each mode
    baseline_kpis = calculate_all_kpis(baseline_results, v_min, v_max)
    heuristic_kpis = calculate_all_kpis(heuristic_results, v_min, v_max)

    # Build comparison DataFrame
    modes_data = {
        "baseline": baseline_kpis,
        "heuristic": heuristic_kpis,
    }

    if optimization_results is not None:
        optimization_kpis = calculate_all_kpis(optimization_results, v_min, v_max)
        modes_data["optimization"] = optimization_kpis

    # Convert to DataFrame
    comparison_df = pd.DataFrame.from_dict(modes_data, orient="index")

    # Reorder columns by category for better readability
    column_order = [
        # Voltage KPIs
        "feeder_violation_minutes",
        "bus_violation_minutes",
        "max_voltage",
        "min_voltage",
        "buses_exceeding_limit",
        "avg_voltage_deviation",
        "voltage_severity_score",
        # Control KPIs
        "total_reactive_energy_kvarh",
        "peak_reactive_dispatch_kvar",
        "active_control_timesteps",
        "total_curtailed_energy_kwh",
        "curtailed_energy_pct",
        "avg_q_dispatch_kvar",
        "avg_p_curtailment_kw",
        # System KPIs
        "total_losses_kwh",
        "total_pv_generation_kwh",
        "avg_pv_generation_kw",
        "peak_pv_generation_kw",
    ]

    # Filter to existing columns and reorder
    existing_cols = [c for c in column_order if c in comparison_df.columns]
    comparison_df = comparison_df[existing_cols]

    return comparison_df


def calculate_improvements(
    baseline_results: pd.DataFrame,
    heuristic_results: pd.DataFrame,
    optimization_results: pd.DataFrame | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> pd.DataFrame:
    """Calculate percentage improvements vs baseline.

    Args:
        baseline_results: Baseline mode QSTS results
        heuristic_results: Heuristic mode QSTS results
        optimization_results: Optional optimization mode QSTS results
        v_min: Lower voltage limit
        v_max: Upper voltage limit

    Returns:
        DataFrame with percentage improvements (negative = improvement for
        metrics where lower is better, like violations)
    """
    baseline_kpis = calculate_all_kpis(baseline_results, v_min, v_max)
    heuristic_kpis = calculate_all_kpis(heuristic_results, v_min, v_max)

    # Calculate improvements
    heuristic_improvement = compare_kpi_dicts(baseline_kpis, heuristic_kpis)

    improvements_data = {
        "heuristic_improvement_pct": heuristic_improvement,
    }

    if optimization_results is not None:
        optimization_kpis = calculate_all_kpis(optimization_results, v_min, v_max)
        optimization_improvement = compare_kpi_dicts(baseline_kpis, optimization_kpis)
        improvements_data["optimization_improvement_pct"] = optimization_improvement

    return pd.DataFrame.from_dict(improvements_data, orient="index")


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------


def export_comparison_table(
    comparison_df: pd.DataFrame,
    output_path: pathlib.Path | str,
) -> None:
    """Export comparison table to CSV and Markdown formats.

    Args:
        comparison_df: DataFrame from compare_modes()
        output_path: Base path for outputs (extension will be added)
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export CSV
    csv_path = output_path.with_suffix(".csv")
    comparison_df.to_csv(csv_path, float_format="%.4f")
    print(f"  Exported CSV: {csv_path}")

    # Export Markdown
    md_path = output_path.with_suffix(".md")
    with md_path.open("w") as f:
        f.write("# KPI Comparison Results\n\n")

        # Table header
        f.write("| Metric |")
        for mode in comparison_df.index:
            f.write(f" {mode.capitalize()} |")
        f.write("\n")

        f.write("|--------|")
        for _ in comparison_df.index:
            f.write("-----------|")
        f.write("\n")

        # Data rows
        for metric in comparison_df.columns:
            f.write(f"| {metric.replace('_', ' ').title()} |")

            for mode in comparison_df.index:
                value = comparison_df.loc[mode, metric]

                # Format based on metric type
                if pd.isna(value):
                    f.write(" N/A |")
                elif isinstance(value, float):
                    if "pct" in metric:
                        f.write(f" {value:.2f}% |")
                    elif "minutes" in metric:
                        f.write(f" {int(value)} |")
                    elif "voltage" in metric and "pu" not in metric:
                        f.write(f" {value:.4f} |")
                    elif "energy" in metric or "kwh" in metric.lower():
                        f.write(f" {value:.2f} |")
                    else:
                        f.write(f" {value:.2f} |")
                else:
                    f.write(f" {value} |")

            f.write("\n")

    print(f"  Exported Markdown: {md_path}")


def export_improvement_table(
    improvement_df: pd.DataFrame,
    output_path: pathlib.Path | str,
) -> None:
    """Export improvement table to CSV and Markdown formats.

    Args:
        improvement_df: DataFrame from calculate_improvements()
        output_path: Base path for outputs (extension will be added)
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export CSV
    csv_path = output_path.with_suffix(".csv")
    improvement_df.to_csv(csv_path, float_format="%.4f")
    print(f"  Exported CSV: {csv_path}")

    # Export Markdown
    md_path = output_path.with_suffix(".md")
    with md_path.open("w") as f:
        f.write("# KPI Improvements vs Baseline\n\n")
        f.write("*Positive values = improvement (lower violations, less curtailment)*\n\n")

        # Table header
        f.write("| Metric |")
        for mode in improvement_df.index:
            mode_name = mode.replace("_improvement_pct", "").capitalize()
            f.write(f" {mode_name} |")
        f.write("\n")

        f.write("|--------|")
        for _ in improvement_df.index:
            f.write("-------------|")
        f.write("\n")

        # Data rows
        for metric in improvement_df.columns:
            f.write(f"| {metric.replace('_', ' ').title()} |")

            for mode in improvement_df.index:
                value = improvement_df.loc[mode, metric]

                if pd.isna(value):
                    f.write(" N/A |")
                elif isinstance(value, float):
                    f.write(f" {value:+.2f}% |")
                else:
                    f.write(f" {value} |")

            f.write("\n")

    print(f"  Exported Markdown: {md_path}")


# ---------------------------------------------------------------------------
# End-to-End Report Generation
# ---------------------------------------------------------------------------


def generate_comparison_report(
    baseline_path: str,
    heuristic_path: str,
    optimization_path: str | None,
    output_dir: str | pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict[str, pathlib.Path]:
    """Generate complete comparison report from result directories.

    Args:
        baseline_path: Path to baseline results directory
        heuristic_path: Path to heuristic results directory
        optimization_path: Optional path to optimization results directory
        output_dir: Directory for outputs
        v_min: Lower voltage limit
        v_max: Upper voltage limit

    Returns:
        Dictionary mapping output name to file path
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating KPI comparison report...")

    # Load results
    print("  Loading simulation results...")
    baseline_results = load_simulation_results(baseline_path, "baseline")
    heuristic_results = load_simulation_results(heuristic_path, "heuristic")

    optimization_results = None
    if optimization_path is not None:
        try:
            optimization_results = load_simulation_results(optimization_path, "optimization")
            print(f"    Loaded optimization results")
        except FileNotFoundError:
            print(f"    Warning: Optimization results not found")

    # Calculate comparison
    print("  Calculating KPIs...")
    comparison_df = compare_modes(
        baseline_results,
        heuristic_results,
        optimization_results,
        v_min,
        v_max,
    )

    # Calculate improvements
    improvement_df = calculate_improvements(
        baseline_results,
        heuristic_results,
        optimization_results,
        v_min,
        v_max,
    )

    # Export tables
    print("  Exporting tables...")
    paths: dict[str, pathlib.Path] = {}

    comparison_path = output_dir / "kpi_comparison"
    export_comparison_table(comparison_df, comparison_path)
    paths["comparison_csv"] = comparison_path.with_suffix(".csv")
    paths["comparison_md"] = comparison_path.with_suffix(".md")

    improvement_path = output_dir / "kpi_improvements"
    export_improvement_table(improvement_df, improvement_path)
    paths["improvement_csv"] = improvement_path.with_suffix(".csv")
    paths["improvement_md"] = improvement_path.with_suffix(".md")

    print(f"\nComparison report complete: {output_dir}")

    return paths
