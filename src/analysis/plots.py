"""
src/analysis/plots.py
---------------------
Visualization functions for QSTS simulation results.

Provides plotting functions for:
- Feeder-wide min/max voltage over time
- Top N worst buses over time
- Snapshot voltage profile
- Histogram of voltage violations
- PV output vs maximum voltage
"""

import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Set up consistent plot styling
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# ---------------------------------------------------------------------------
# Main Plotting Functions
# ---------------------------------------------------------------------------


def plot_voltage_envelope(
    results: pd.DataFrame,
    output_path: str | pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> None:
    """Plot feeder-wide min/max voltage envelope over time.

    Args:
        results: QSTS results DataFrame
        output_path: Where to save the plot
        v_min: Lower voltage limit
        v_max: Upper voltage limit
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    time_hours = results["time_h"]

    # Plot envelope
    ax.fill_between(
        time_hours,
        results["v_min"],
        results["v_max"],
        alpha=0.3,
        label="Voltage Envelope",
        color="#3498db",
    )
    ax.plot(time_hours, results["v_max"], "b-", linewidth=1.5, label="Max Voltage")
    ax.plot(time_hours, results["v_min"], "b--", linewidth=1, label="Min Voltage")
    ax.plot(time_hours, results["v_mean"], "g:", linewidth=1, label="Mean Voltage")

    # Voltage limits
    ax.axhline(v_max, color="#e74c3c", linewidth=2, linestyle="--", label=f"Upper Limit ({v_max} pu)")
    ax.axhline(v_min, color="#e67e22", linewidth=2, linestyle="--", label=f"Lower Limit ({v_min} pu)")
    ax.axhline(1.0, color="#95a5a6", linewidth=1, linestyle=":", label="Nominal (1.0 pu)")

    # Highlight violations
    overvoltage = results["v_max"] > v_max
    if overvoltage.any():
        ax.scatter(
            time_hours[overvoltage],
            results.loc[overvoltage, "v_max"],
            color="#e74c3c",
            s=30,
            zorder=5,
            label="Overvoltage",
        )

    undervoltage = results["v_min"] < v_min
    if undervoltage.any():
        ax.scatter(
            time_hours[undervoltage],
            results.loc[undervoltage, "v_min"],
            color="#e67e22",
            s=30,
            zorder=5,
            label="Undervoltage",
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title("Feeder Voltage Envelope - 24 Hour QSTS")
    ax.set_xlim(0, 24)
    ax.set_ylim(max(0.85, results["v_min"].min() - 0.02), min(1.15, results["v_max"].max() + 0.02))
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_worst_buses(
    results: pd.DataFrame,
    bus_voltages_over_time: dict[str, list[float]],
    output_path: str | pathlib.Path,
    top_n: int = 10,
    v_max: float = 1.05,
) -> None:
    """Plot the top N worst buses over time.

    Args:
        results: QSTS results DataFrame
        bus_voltages_over_time: Dict mapping bus name to list of voltages
        output_path: Where to save the plot
        top_n: Number of worst buses to show
        v_max: Upper voltage limit
    """
    # Find worst buses by maximum voltage
    bus_max_voltages = {
        bus: max(voltages) for bus, voltages in bus_voltages_over_time.items()
    }
    worst_buses = sorted(bus_max_voltages.items(), key=lambda x: x[1], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(14, 6))

    time_hours = results["time_h"]

    for bus, _ in worst_buses:
        voltages = bus_voltages_over_time[bus]
        ax.plot(time_hours, voltages, linewidth=1.5, label=f"Bus {bus}")

    ax.axhline(v_max, color="#e74c3c", linewidth=2, linestyle="--", label=f"Limit ({v_max} pu)")
    ax.axhline(1.0, color="#95a5a6", linewidth=1, linestyle=":", label="Nominal")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(f"Top {top_n} Highest Voltage Buses - 24 Hour QSTS")
    ax.set_xlim(0, 24)
    ax.set_ylim(0.95, min(1.1, max(v for _, v in worst_buses) + 0.01))
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_snapshot_voltage_profile(
    voltages: dict[str, float],
    output_path: str | pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
    title: str = "Voltage Profile - Snapshot",
) -> None:
    """Plot a snapshot voltage profile by bus.

    Args:
        voltages: Dict mapping bus name to voltage
        output_path: Where to save the plot
        v_min: Lower voltage limit
        v_max: Upper voltage limit
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(max(10, len(voltages) * 0.4), 5))

    buses = sorted(voltages.keys())
    v_vals = [voltages[b] for b in buses]

    # Color code violations
    colors = [
        "#e74c3c" if (v > v_max or v < v_min) else "#2ecc71"
        for v in v_vals
    ]

    x_pos = np.arange(len(buses))
    ax.bar(x_pos, v_vals, color=colors, width=0.7, zorder=3)

    ax.axhline(v_max, color="#e74c3c", linewidth=1.5, linestyle="--", label=f"Upper ({v_max} pu)")
    ax.axhline(v_min, color="#e67e22", linewidth=1.5, linestyle="--", label=f"Lower ({v_min} pu)")
    ax.axhline(1.0, color="#95a5a6", linewidth=1, linestyle=":", label="Nominal")

    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(title)
    ax.set_xticks(x_pos[::max(1, len(buses) // 20)])  # Show fewer labels if many buses
    ax.set_xticklabels([buses[i] for i in range(0, len(buses), max(1, len(buses) // 20))], rotation=70, fontsize=8)
    ax.set_ylim(max(0.85, min(v_vals) - 0.02), min(1.15, max(v_vals) + 0.02))
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_voltage_histogram(
    results: pd.DataFrame,
    output_path: str | pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> None:
    """Plot histogram of all voltage measurements.

    Args:
        results: QSTS results DataFrame
        output_path: Where to save the plot
        v_min: Lower voltage limit
        v_max: Upper voltage limit
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all voltage readings
    all_voltages = []
    all_voltages.extend(results["v_min"].tolist())
    all_voltages.extend(results["v_max"].tolist())
    all_voltages.extend(results["v_mean"].tolist())

    bins = np.linspace(0.90, 1.10, 100)

    # Plot histogram
    counts, _, patches = ax.hist(all_voltages, bins=bins, alpha=0.7, color="#3498db", edgecolor="white")

    # Color code violation regions
    for patch in patches:
        bin_center = (patch.get_x() + patch.get_x() + patch.get_width()) / 2
        if bin_center > v_max or bin_center < v_min:
            patch.set_facecolor("#e74c3c")

    ax.axvline(v_max, color="#e74c3c", linewidth=2, linestyle="--", label=f"Upper limit ({v_max} pu)")
    ax.axvline(v_min, color="#e67e22", linewidth=2, linestyle="--", label=f"Lower limit ({v_min} pu)")
    ax.axvline(1.0, color="#95a5a6", linewidth=1.5, linestyle="-", label="Nominal (1.0 pu)")

    # Add violation percentage
    violations = sum(1 for v in all_voltages if v > v_max or v < v_min)
    pct = 100 * violations / len(all_voltages)
    ax.text(0.98, 0.95, f"Violations: {pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Voltage (pu)")
    ax.set_ylabel("Count")
    ax.set_title("Voltage Distribution - All Measurements (24h QSTS)")
    ax.set_xlim(0.90, 1.10)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pv_vs_voltage(
    results: pd.DataFrame,
    output_path: str | pathlib.Path,
    v_max: float = 1.05,
) -> None:
    """Plot PV generation vs maximum feeder voltage.

    Args:
        results: QSTS results DataFrame
        output_path: Where to save the plot
        v_max: Upper voltage limit
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    time_hours = results["time_h"]

    # Plot voltage on primary axis
    color1 = "#e74c3c"
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Max Voltage (pu)", color=color1)
    line1 = ax1.plot(time_hours, results["v_max"], color=color1, linewidth=2, label="Max Voltage")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(v_max, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.5)
    ax1.set_ylim(0.95, 1.10)

    # Plot PV on secondary axis
    ax2 = ax1.twinx()
    color2 = "#f39c12"
    ax2.set_ylabel("PV Generation (kW)", color=color2)
    line2 = ax2.plot(time_hours, results["pv_generation_kw"], color=color2, linewidth=2, label="PV Generation", alpha=0.8)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.fill_between(time_hours, results["pv_generation_kw"], alpha=0.2, color=color2)

    ax1.set_xlim(0, 24)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_title("PV Generation vs Maximum Feeder Voltage")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_violation_timeline(
    results: pd.DataFrame,
    output_path: str | pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> None:
    """Plot a timeline showing when voltage violations occur.

    Args:
        results: QSTS results DataFrame
        output_path: Where to save the plot
        v_min: Lower voltage limit
        v_max: Upper voltage limit
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    time_hours = results["time_h"]

    # Create violation bands
    overvoltage = results["v_max"] > v_max
    undervoltage = results["v_min"] < v_min
    any_violation = results["violating_buses_count"] > 0

    # Plot violation bands
    ax.fill_between(
        time_hours,
        0,
        np.where(overvoltage, 1, 0),
        step="post",
        alpha=0.5,
        color="#e74c3c",
        label="Overvoltage",
    )
    ax.fill_between(
        time_hours,
        0,
        np.where(undervoltage, 1, 0),
        step="post",
        alpha=0.5,
        color="#e67e22",
        label="Undervoltage",
    )

    # Add count of violating buses
    ax2 = ax.twinx()
    ax2.plot(time_hours, results["violating_buses_count"], "k:", linewidth=1, alpha=0.5)
    ax2.fill_between(time_hours, results["violating_buses_count"], alpha=0.1, color="gray")
    ax2.set_ylabel("Number of Violating Buses", fontsize=9)
    ax2.set_ylim(0, max(5, results["violating_buses_count"].max() + 1))

    ax.set_xlabel("Time (hours)")
    ax.set_yticks([])
    ax.set_xlim(0, 24)
    ax.set_title("Voltage Violation Timeline - 24 Hour QSTS")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary Plot Collection
# ---------------------------------------------------------------------------


def create_baseline_plots(
    results: pd.DataFrame,
    output_dir: str | pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
    snapshot_voltages: dict[str, float] | None = None,
    bus_voltages_over_time: dict[str, list[float]] | None = None,
) -> dict[str, pathlib.Path]:
    """Create all standard baseline plots.

    Args:
        results: QSTS results DataFrame
        output_dir: Directory for outputs
        v_min: Lower voltage limit
        v_max: Upper voltage limit
        snapshot_voltages: Optional snapshot voltages for noon profile
        bus_voltages_over_time: Optional per-bus voltage time series

    Returns:
        Dict mapping plot name to output path
    """
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    print("\nGenerating baseline plots...")

    # 1. Voltage envelope
    path = out_dir / "plot_voltage_envelope.png"
    plot_voltage_envelope(results, path, v_min, v_max)
    paths["envelope"] = path
    print(f"  Created: {path.name}")

    # 2. Worst buses (if per-bus data available)
    if bus_voltages_over_time:
        path = out_dir / "plot_worst_buses.png"
        plot_worst_buses(results, bus_voltages_over_time, path, top_n=10, v_max=v_max)
        paths["worst_buses"] = path
        print(f"  Created: {path.name}")

    # 3. Snapshot profile (if provided)
    if snapshot_voltages:
        path = out_dir / "plot_snapshot_voltage.png"
        plot_snapshot_voltage_profile(snapshot_voltages, path, v_min, v_max, "Voltage Profile - Noon Snapshot")
        paths["snapshot"] = path
        print(f"  Created: {path.name}")

    # 4. Voltage histogram
    path = out_dir / "plot_voltage_histogram.png"
    plot_voltage_histogram(results, path, v_min, v_max)
    paths["histogram"] = path
    print(f"  Created: {path.name}")

    # 5. PV vs voltage
    path = out_dir / "plot_pv_vs_voltage.png"
    plot_pv_vs_voltage(results, path, v_max)
    paths["pv_vs_voltage"] = path
    print(f"  Created: {path.name}")

    # 6. Violation timeline
    path = out_dir / "plot_violation_timeline.png"
    plot_violation_timeline(results, path, v_min, v_max)
    paths["violation_timeline"] = path
    print(f"  Created: {path.name}")

    return paths


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def extract_per_bus_voltages(results: pd.DataFrame, all_buses: list[str]) -> dict[str, list[float]]:
    """Extract per-bus voltage time series from results.

    This looks for columns named 'overv_<bus>' or similar in the results.

    Args:
        results: QSTS results DataFrame
        all_buses: List of all bus names

    Returns:
        Dict mapping bus name to list of voltages over time
    """
    bus_voltages = {}

    # Initialize with mean voltage for all timesteps
    for bus in all_buses:
        bus_voltages[bus] = results["v_mean"].tolist()

    # Fill in specific bus data if available
    for col in results.columns:
        if col.startswith("overv_"):
            bus = col.replace("overv_", "")
            if bus in bus_voltages:
                bus_voltages[bus] = results[col].tolist()

    return bus_voltages


# ---------------------------------------------------------------------------
# Comparison Plots (Baseline vs. Controlled)
# ---------------------------------------------------------------------------


def plot_comparison_voltage_envelope(
    baseline_results: pd.DataFrame,
    controlled_results: pd.DataFrame,
    output_path: str | pathlib.Path,
    v_max: float = 1.05,
) -> None:
    """Plot baseline vs controlled voltage envelopes on same axes.

    Args:
        baseline_results: Baseline QSTS results DataFrame
        controlled_results: Controlled QSTS results DataFrame
        output_path: Where to save the plot
        v_max: Upper voltage limit for reference line
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    time_hours = baseline_results["time_h"]

    # Plot baseline envelope
    ax.fill_between(
        time_hours,
        baseline_results["v_min"],
        baseline_results["v_max"],
        alpha=0.2,
        label="Baseline Envelope",
        color="#e74c3c",
    )
    ax.plot(time_hours, baseline_results["v_max"], "r-", linewidth=1.5, label="Baseline Max")
    ax.plot(time_hours, baseline_results["v_min"], "r--", linewidth=1, label="Baseline Min")

    # Plot controlled envelope
    ax.fill_between(
        time_hours,
        controlled_results["v_min"],
        controlled_results["v_max"],
        alpha=0.2,
        label="Controlled Envelope",
        color="#2ecc71",
    )
    ax.plot(time_hours, controlled_results["v_max"], "g-", linewidth=1.5, label="Controlled Max")
    ax.plot(time_hours, controlled_results["v_min"], "g--", linewidth=1, label="Controlled Min")

    # Voltage limits
    ax.axhline(v_max, color="#34495e", linewidth=2, linestyle="--", label=f"Limit ({v_max} pu)")
    ax.axhline(1.0, color="#95a5a6", linewidth=1, linestyle=":", label="Nominal")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title("Voltage Envelope Comparison: Baseline vs. Heuristic Control")
    ax.set_xlim(0, 24)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_q_dispatch(
    results: pd.DataFrame,
    output_path: str | pathlib.Path,
) -> None:
    """Plot total reactive power dispatch over time.

    Args:
        results: QSTS results DataFrame with total_q_dispatch_kvar column
        output_path: Where to save the plot
    """
    if "total_q_dispatch_kvar" not in results.columns:
        # No control data - create empty plot or skip
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.text(0.5, 0.5, "No reactive power dispatch data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    time_hours = results["time_h"]
    q_dispatch = results["total_q_dispatch_kvar"]

    # Plot Q dispatch
    ax.fill_between(time_hours, q_dispatch, alpha=0.3, color="#3498db")
    ax.plot(time_hours, q_dispatch, "b-", linewidth=2, label="Total Q Absorption")

    # Mark zero line
    ax.axhline(0, color="#95a5a6", linewidth=1, linestyle="-")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Reactive Power Absorption (kVAR)")
    ax.set_title("Reactive Power Dispatch - Heuristic Controller")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, max(10, q_dispatch.max() * 1.1))
    ax.legend(loc="upper right", fontsize=9)

    # Add total energy absorbed
    total_kvarh = (q_dispatch * 5 / 60).sum()
    ax.text(0.98, 0.95, f"Total: {total_kvarh:.1f} kVARh",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_p_curtailment(
    results: pd.DataFrame,
    output_path: str | pathlib.Path,
) -> None:
    """Plot total active power curtailment over time.

    Args:
        results: QSTS results DataFrame with total_p_curtailment_kw column
        output_path: Where to save the plot
    """
    if "total_p_curtailment_kw" not in results.columns:
        # No control data - create empty plot or skip
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.text(0.5, 0.5, "No active power curtailment data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    time_hours = results["time_h"]
    p_curtail = results["total_p_curtailment_kw"]

    # Plot P curtailment
    ax.fill_between(time_hours, p_curtail, alpha=0.3, color="#e67e22")
    ax.plot(time_hours, p_curtail, "orange", linewidth=2, label="Total P Curtailment")

    # Mark zero line
    ax.axhline(0, color="#95a5a6", linewidth=1, linestyle="-")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Active Power Curtailment (kW)")
    ax.set_title("Active Power Curtailment - Heuristic Controller")
    ax.set_xlim(0, 24)

    # Set y-limit with some headroom
    y_max = max(10, p_curtail.max() * 1.1) if p_curtail.max() > 0 else 10
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", fontsize=9)

    # Add total energy curtailed
    total_curtail_kwh = (p_curtail * 5 / 60).sum()
    ax.text(0.98, 0.95, f"Total: {total_curtail_kwh:.1f} kWh curtailed",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_violation_comparison(
    baseline_results: pd.DataFrame,
    controlled_results: pd.DataFrame,
    output_path: str | pathlib.Path,
    v_max: float = 1.05,
) -> None:
    """Plot overvoltage occurrence comparison.

    Args:
        baseline_results: Baseline QSTS results DataFrame
        controlled_results: Controlled QSTS results DataFrame
        output_path: Where to save the plot
        v_max: Upper voltage limit
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    time_hours = baseline_results["time_h"]

    # Calculate overvoltage flags
    baseline_over = (baseline_results["v_max"] > v_max).astype(int)
    controlled_over = (controlled_results["v_max"] > v_max).astype(int)

    # Plot violation bands
    ax.fill_between(
        time_hours,
        0,
        baseline_over,
        step="post",
        alpha=0.4,
        color="#e74c3c",
        label="Baseline Overvoltage",
    )
    ax.fill_between(
        time_hours,
        0,
        -controlled_over,
        step="post",
        alpha=0.4,
        color="#2ecc71",
        label="Controlled Overvoltage",
    )

    ax.set_xlabel("Time (hours)")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Controlled", "No", "Baseline"])
    ax.set_xlim(0, 24)
    ax.set_title("Overvoltage Occurrence Comparison")
    ax.legend(loc="upper right", fontsize=9)

    # Add statistics
    baseline_count = baseline_over.sum()
    controlled_count = controlled_over.sum()
    reduction_pct = 100 * (1 - controlled_count / baseline_count) if baseline_count > 0 else 0

    ax.text(0.02, 0.05,
            f"Baseline: {baseline_count} steps\n"
            f"Controlled: {controlled_count} steps\n"
            f"Reduction: {reduction_pct:.1f}%",
            transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_comparison_plots(
    baseline_results: pd.DataFrame,
    controlled_results: pd.DataFrame,
    output_dir: str | pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict[str, pathlib.Path]:
    """Generate all baseline vs controlled comparison plots.

    Args:
        baseline_results: Baseline QSTS results DataFrame
        controlled_results: Controlled QSTS results DataFrame
        output_dir: Directory for outputs
        v_min: Lower voltage limit
        v_max: Upper voltage limit

    Returns:
        Dict mapping plot name to output path
    """
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    print("\nGenerating comparison plots...")

    # 1. Voltage envelope comparison
    path = out_dir / "plot_voltage_comparison.png"
    plot_comparison_voltage_envelope(baseline_results, controlled_results, path, v_max)
    paths["voltage_comparison"] = path
    print(f"  Created: {path.name}")

    # 2. Q dispatch
    path = out_dir / "plot_q_dispatch.png"
    plot_q_dispatch(controlled_results, path)
    paths["q_dispatch"] = path
    print(f"  Created: {path.name}")

    # 3. P curtailment
    path = out_dir / "plot_p_curtailment.png"
    plot_p_curtailment(controlled_results, path)
    paths["p_curtailment"] = path
    print(f"  Created: {path.name}")

    # 4. Violation comparison
    path = out_dir / "plot_violation_comparison.png"
    plot_violation_comparison(baseline_results, controlled_results, path, v_max)
    paths["violation_comparison"] = path
    print(f"  Created: {path.name}")

    return paths
