"""
src/analysis/hosting_capacity.py
---------------------------------
Automated PV hosting capacity study.

Provides functions to:
- Run PV scaling sweeps to find voltage limits
- Calculate hosting capacity with interpolation
- Compare hosting capacity across control modes
- Generate hosting capacity comparison plots
"""

import itertools
import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kpis import calculate_all_kpis
from src.sim.run_qsts import run_qsts


# Set up consistent plot styling
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# ---------------------------------------------------------------------------
# PV Sweep Functions
# ---------------------------------------------------------------------------


def run_pv_sweep(
    config_path: str | pathlib.Path,
    pv_scales: list[float],
    output_base_dir: str | pathlib.Path,
) -> pd.DataFrame:
    """Run QSTS simulation at multiple PV scale factors.

    Args:
        config_path: Path to study config YAML
        pv_scales: List of PV scale factors (e.g., [0.5, 1.0, 1.5, 2.0])
        output_base_dir: Base directory for sweep outputs

    Returns:
        DataFrame with KPIs for each PV scale
    """
    config_path = pathlib.Path(config_path)
    output_base_dir = pathlib.Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    results_list = []

    print(f"\n{'='*60}")
    print(f"  PV Hosting Capacity Sweep")
    print(f"{'='*60}")
    print(f"  Config     : {config_path}")
    print(f"  PV scales  : {pv_scales}")
    print(f"  Output dir : {output_base_dir}")
    print(f"{'='*60}\n")

    for i, pv_scale in enumerate(pv_scales):
        print(f"\n[{i+1}/{len(pv_scales)}] Running PV scale {pv_scale:.2f}x...")

        # Create output directory for this scale
        scale_dir = output_base_dir / f"pv_scale_{pv_scale:.2f}"

        # Run simulation
        try:
            results_df = run_qsts(config_path, pv_scale=pv_scale)

            # Calculate KPIs
            kpis = calculate_all_kpis(results_df)
            kpis["pv_scale"] = pv_scale
            kpis["sweep_successful"] = True

            # Save results for this scale
            scale_dir.mkdir(parents=True, exist_ok=True)
            results_path = scale_dir / "qsts_baseline.csv"
            results_df.to_csv(results_path, index=False)

            results_list.append(kpis)
            print(f"  Completed: {len(results_df)} timesteps")

        except Exception as e:
            print(f"  Failed: {e}")
            results_list.append({
                "pv_scale": pv_scale,
                "sweep_successful": False,
                "error": str(e),
            })

    # Create summary DataFrame
    sweep_df = pd.DataFrame(results_list)

    # Save summary
    summary_path = output_base_dir / "sweep_summary.csv"
    sweep_df.to_csv(summary_path, index=False)
    print(f"\nSweep summary saved: {summary_path}")

    return sweep_df


def find_hosting_capacity(
    results: pd.DataFrame,
    v_max: float = 1.05,
    max_violation_minutes: int = 0,
) -> float:
    """Find maximum PV scale with acceptable voltage violations.

    Args:
        results: Sweep results DataFrame from run_pv_sweep
        v_max: Upper voltage limit
        max_violation_minutes: Maximum allowed violation minutes

    Returns:
        Maximum PV scale factor within limits
    """
    # Filter to successful runs
    successful = results[results.get("sweep_successful", True)]

    if len(successful) == 0:
        return 0.0

    # Find scales that meet violation criteria
    acceptable = successful[
        successful["feeder_violation_minutes"] <= max_violation_minutes
    ]

    if len(acceptable) == 0:
        return 0.0

    # Return max acceptable scale
    return float(acceptable["pv_scale"].max())


def find_hosting_capacity_interpolated(
    results: pd.DataFrame,
    v_max: float = 1.05,
    max_violation_minutes: int = 0,
) -> dict[str, Any]:
    """Find hosting capacity with linear interpolation.

    Args:
        results: Sweep results DataFrame from run_pv_sweep
        v_max: Upper voltage limit
        max_violation_minutes: Maximum allowed violation minutes

    Returns:
        Dictionary with:
        - hosting_capacity: Estimated PV scale at violation threshold
        - max_safe_scale: Maximum scale with zero violations
        - violation_at_max_scale: Violations at highest tested scale
        - interpolation_valid: Whether interpolation was possible
    """
    # Filter to successful runs
    successful = results[results.get("sweep_successful", True)].copy()
    successful = successful.sort_values("pv_scale")

    if len(successful) == 0:
        return {
            "hosting_capacity": 0.0,
            "max_safe_scale": 0.0,
            "violation_at_max_scale": 0,
            "interpolation_valid": False,
        }

    # Find the boundary between acceptable and unacceptable
    acceptable = successful[
        successful["feeder_violation_minutes"] <= max_violation_minutes
    ]
    unacceptable = successful[
        successful["feeder_violation_minutes"] > max_violation_minutes
    ]

    max_safe = float(acceptable["pv_scale"].max()) if len(acceptable) > 0 else 0.0

    if len(unacceptable) == 0:
        # All tested scales are acceptable
        return {
            "hosting_capacity": max_safe,
            "max_safe_scale": max_safe,
            "violation_at_max_scale": int(successful.iloc[-1]["feeder_violation_minutes"]),
            "interpolation_valid": False,
        }

    if len(acceptable) == 0:
        # Even the minimum scale causes violations
        min_unacceptable_scale = float(unacceptable["pv_scale"].min())
        return {
            "hosting_capacity": 0.0,
            "max_safe_scale": 0.0,
            "violation_at_max_scale": int(unacceptable.iloc[0]["feeder_violation_minutes"]),
            "interpolation_valid": True,
        }

    # Interpolate between max acceptable and min unacceptable
    last_acceptable = acceptable.iloc[-1]
    first_unacceptable = unacceptable.iloc[0]

    x1 = last_acceptable["pv_scale"]
    y1 = last_acceptable["feeder_violation_minutes"]
    x2 = first_unacceptable["pv_scale"]
    y2 = first_unacceptable["feeder_violation_minutes"]

    # Linear interpolation to find where violations = max_violation_minutes
    if y2 != y1:
        hosting_capacity = x1 + (max_violation_minutes - y1) * (x2 - x1) / (y2 - y1)
    else:
        hosting_capacity = x1

    return {
        "hosting_capacity": float(hosting_capacity),
        "max_safe_scale": max_safe,
        "violation_at_max_scale": int(unacceptable.iloc[0]["feeder_violation_minutes"]),
        "interpolation_valid": True,
    }


def find_hosting_capacity_binary(
    config_path: str | pathlib.Path,
    initial_range: tuple[float, float] = (0.5, 5.0),
    tolerance: float = 0.1,
    max_iterations: int = 10,
    max_violation_minutes: int = 0,
) -> dict[str, Any]:
    """Find hosting capacity using binary search.

    More efficient than linear sweep when range is known.

    Args:
        config_path: Path to study config YAML
        initial_range: (min_scale, max_scale) to search within
        tolerance: Stop when range is smaller than this
        max_iterations: Maximum number of iterations
        max_violation_minutes: Maximum allowed violation minutes

    Returns:
        Dictionary with search results
    """
    low, high = initial_range
    output_dir = pathlib.Path(config_path).parent.parent / "results" / "hosting_capacity_binary"

    print(f"\nBinary search for hosting capacity...")
    print(f"  Initial range: [{low}, {high}]")
    print(f"  Tolerance: {tolerance}")

    results = {
        "iterations": [],
        "final_capacity": 0.0,
        "converged": False,
    }

    for i in range(max_iterations):
        mid = (low + high) / 2
        print(f"\n  Iteration {i+1}: testing scale {mid:.2f}")

        # Create temporary output dir for this test
        test_dir = output_dir / f"test_{i}"

        try:
            results_df = run_qsts(config_path, pv_scale=mid)
            kpis = calculate_all_kpis(results_df)
            violations = kpis["feeder_violation_minutes"]

            results["iterations"].append({
                "iteration": i,
                "scale": mid,
                "violations": violations,
            })

            if violations <= max_violation_minutes:
                low = mid
                print(f"    Acceptable: {violations} violations")
            else:
                high = mid
                print(f"    Too high: {violations} violations")

        except Exception as e:
            print(f"    Failed: {e}")
            # Treat failure as too high
            high = mid

        # Check convergence
        if high - low < tolerance:
            results["final_capacity"] = (low + high) / 2
            results["converged"] = True
            print(f"\n  Converged to: {results['final_capacity']:.2f}")
            break

    if not results["converged"]:
        results["final_capacity"] = low
        print(f"\n  Did not converge, returning last acceptable: {low:.2f}")

    # Save iteration history
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "binary_search_history.json"
    with history_path.open("w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Comparison Functions
# ---------------------------------------------------------------------------


def compare_hosting_capacity(
    baseline_config: str,
    heuristic_config: str,
    optimization_config: str | None,
    pv_scales: list[float],
    output_dir: str | pathlib.Path,
) -> dict[str, Any]:
    """Compare hosting capacity across control modes.

    Args:
        baseline_config: Path to baseline config
        heuristic_config: Path to heuristic config
        optimization_config: Optional path to optimization config
        pv_scales: PV scale factors to test
        output_dir: Directory for outputs

    Returns:
        Dictionary with comparison results
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run sweeps
    for mode, config in [
        ("baseline", baseline_config),
        ("heuristic", heuristic_config),
    ]:
        print(f"\n{'='*60}")
        print(f"Running {mode} sweep...")
        print(f"{'='*60}")

        mode_dir = output_dir / mode
        sweep_df = run_pv_sweep(config, pv_scales, mode_dir)

        capacity_info = find_hosting_capacity_interpolated(sweep_df)
        results[mode] = {
            "sweep_results": sweep_df,
            "capacity_info": capacity_info,
        }

        print(f"\n{mode.capitalize()} hosting capacity: {capacity_info['hosting_capacity']:.2f}x")

    # Run optimization if provided
    if optimization_config is not None:
        print(f"\n{'='*60}")
        print(f"Running optimization sweep...")
        print(f"{'='*60}")

        mode_dir = output_dir / "optimization"
        sweep_df = run_pv_sweep(optimization_config, pv_scales, mode_dir)

        capacity_info = find_hosting_capacity_interpolated(sweep_df)
        results["optimization"] = {
            "sweep_results": sweep_df,
            "capacity_info": capacity_info,
        }

        print(f"\nOptimization hosting capacity: {capacity_info['hosting_capacity']:.2f}x")

    # Calculate improvement ratios
    baseline_capacity = results["baseline"]["capacity_info"]["hosting_capacity"]
    improvements = {}

    for mode in ["heuristic", "optimization"]:
        if mode in results:
            mode_capacity = results[mode]["capacity_info"]["hosting_capacity"]
            if baseline_capacity > 0:
                improvements[f"{mode}_improvement_ratio"] = mode_capacity / baseline_capacity
            elif mode_capacity == 0:
                improvements[f"{mode}_improvement_ratio"] = 1.0
            else:
                improvements[f"{mode}_improvement_ratio"] = float("inf")

    results["improvements"] = improvements

    # Save summary
    summary_path = output_dir / "hosting_capacity_summary.json"
    with summary_path.open("w") as f:
        # Convert to JSON-serializable format
        serializable_results = {
            mode: {
                "hosting_capacity": data["capacity_info"]["hosting_capacity"],
                "max_safe_scale": data["capacity_info"]["max_safe_scale"],
                "interpolation_valid": data["capacity_info"]["interpolation_valid"],
            }
            for mode, data in results.items()
            if mode != "improvements"
        }
        serializable_results["improvements"] = improvements
        json.dump(serializable_results, f, indent=2)

    print(f"\nHosting capacity summary saved: {summary_path}")

    return results


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------


def plot_hosting_capacity_comparison(
    results: dict[str, Any],
    output_path: str | pathlib.Path,
) -> None:
    """Create bar chart comparing hosting capacity across modes.

    Args:
        results: Results dictionary from compare_hosting_capacity
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = []
    capacities = []

    for mode in ["baseline", "heuristic", "optimization"]:
        if mode in results and "capacity_info" in results[mode]:
            modes.append(mode.capitalize())
            capacities.append(results[mode]["capacity_info"]["hosting_capacity"])

    if not modes:
        print("No hosting capacity data to plot")
        return

    # Color scheme
    colors = ["#e74c3c", "#f39c12", "#2ecc71"][:len(modes)]

    bars = ax.bar(modes, capacities, color=colors, alpha=0.8, edgecolor="white", linewidth=2)

    # Add value labels on bars
    for bar, cap in zip(bars, capacities):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{cap:.2f}x",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Hosting Capacity (PV Scale Factor)", fontsize=12)
    ax.set_title("PV Hosting Capacity Comparison", fontsize=14, fontweight="bold")
    max_capacity = max(capacities) if capacities else 0
    ax.set_ylim(0, max(1.0, max_capacity * 1.2))

    # Add grid
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plot saved: {output_path}")


def plot_sweep_results(
    sweep_df: pd.DataFrame,
    output_path: str | pathlib.Path,
    v_max: float = 1.05,
) -> None:
    """Plot violation minutes vs PV scale.

    Args:
        sweep_df: Sweep results DataFrame
        output_path: Where to save the plot
        v_max: Upper voltage limit (for reference)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to successful runs
    successful = sweep_df[sweep_df.get("sweep_successful", True)]

    if len(successful) == 0:
        print("No successful sweep results to plot")
        return

    x = successful["pv_scale"].values
    y = successful["feeder_violation_minutes"].values

    # Plot line and markers
    ax.plot(x, y, "o-", color="#3498db", linewidth=2, markersize=8, label="Violation Minutes")

    # Highlight zero-violation region
    ax.axhline(0, color="#2ecc71", linewidth=2, linestyle="--", label="No Violations")
    ax.fill_between(x, 0, y, where=(y == 0), alpha=0.2, color="#2ecc71")

    # Highlight violation region
    ax.fill_between(x, 0, y, where=(y > 0), alpha=0.2, color="#e74c3c")

    ax.set_xlabel("PV Scale Factor", fontsize=12)
    ax.set_ylabel("Violation Minutes (per day)", fontsize=12)
    ax.set_title("Voltage Violations vs PV Penetration", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plot saved: {output_path}")


def plot_voltage_vs_pv_scale(
    results: dict[str, Any],
    output_path: str | pathlib.Path,
) -> None:
    """Plot max voltage vs PV scale for all modes.

    Args:
        results: Results dictionary from compare_hosting_capacity
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    v_max_ref = 1.05

    colors = {
        "baseline": "#e74c3c",
        "heuristic": "#f39c12",
        "optimization": "#2ecc71",
    }

    for mode, color in colors.items():
        if mode in results and "sweep_results" in results[mode]:
            sweep_df = results[mode]["sweep_results"]
            successful = sweep_df[sweep_df.get("sweep_successful", True)]

            if len(successful) > 0:
                x = successful["pv_scale"].values
                y = successful["max_voltage"].values
                ax.plot(x, y, "o-", color=color, linewidth=2, markersize=6,
                       label=mode.capitalize())

    # Reference line
    ax.axhline(v_max_ref, color="gray", linewidth=2, linestyle="--",
              label=f"Voltage Limit ({v_max_ref} pu)")

    ax.set_xlabel("PV Scale Factor", fontsize=12)
    ax.set_ylabel("Maximum Voltage (pu)", fontsize=12)
    ax.set_title("Maximum Voltage vs PV Penetration", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plot saved: {output_path}")
