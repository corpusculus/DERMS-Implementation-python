"""
src/sim/run_snapshot.py
------------------------
CLI entry-point for running a single OpenDSS power-flow snapshot.

Usage:
    python -m src.sim.run_snapshot --config config/feeder_dev_ieee13.yaml

Outputs (to the path configured under 'output.snapshot_dir'):
    - bus_voltages.csv   : Per-unit voltage for every bus
    - bus_voltages.png   : Bar chart of bus voltages with limit lines
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is importable when run as __main__
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sim.opendss_interface import (
    export_results,
    get_bus_voltages,
    get_circuit_summary,
    load_feeder,
    solve_power_flow,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_snapshot(config_path: str | pathlib.Path) -> None:
    """Load feeder, solve one snapshot, and export results.

    Args:
        config_path: Path to the study YAML config file.
    """
    cfg = load_config(config_path)

    # Resolve feeder path relative to config directory
    config_dir = pathlib.Path(config_path).resolve().parent
    dss_master = config_dir.parent / cfg["feeder"]["master_file"]

    print(f"\n{'='*60}")
    print(f"  DERMS MVP — Snapshot Run")
    print(f"{'='*60}")
    print(f"  Config  : {config_path}")
    print(f"  Feeder  : {dss_master}")
    print(f"{'='*60}\n")

    # --- Load and solve ---
    print("Loading feeder ...")
    load_feeder(dss_master)

    print("Solving power flow ...")
    solve_power_flow()

    summary = get_circuit_summary()
    print(f"\nCircuit   : {summary['name']}")
    print(f"Buses     : {summary['num_buses']}")
    print(f"Elements  : {summary['num_elements']}")
    print(f"Converged : {summary['converged']}\n")

    # --- Read voltages ---
    voltages = get_bus_voltages()

    # Print summary table
    v_vals = list(voltages.values())
    print(f"{'Bus':>20}  {'V (pu)':>8}")
    print("-" * 32)
    for bus, v in sorted(voltages.items()):
        flag = "  ⚠" if v > 1.05 or v < 0.95 else ""
        print(f"{bus:>20}  {v:>8.4f}{flag}")

    print("-" * 32)
    print(f"{'Min':>20}  {min(v_vals):>8.4f}")
    print(f"{'Max':>20}  {max(v_vals):>8.4f}")
    print(f"{'Mean':>20}  {sum(v_vals)/len(v_vals):>8.4f}\n")

    # --- Export CSV ---
    out_dir_str = cfg.get("output", {}).get("snapshot_dir", "results/baseline")
    # Resolve relative to project root
    out_dir = (_ROOT / out_dir_str).resolve()
    ensure_dir(out_dir)

    csv_path = export_results(voltages, out_dir / "bus_voltages.csv")
    print(f"CSV saved : {csv_path}")

    # --- Plot ---
    plot_path = out_dir / "bus_voltages.png"
    _plot_voltages(voltages, plot_path, cfg)
    print(f"Plot saved: {plot_path}\n")


def _plot_voltages(
    voltages: dict[str, float],
    output_path: pathlib.Path,
    cfg: dict,
) -> None:
    """Generate and save a bar chart of bus voltages."""
    v_limits = cfg.get("voltage_limits", {})
    v_min = v_limits.get("lower", 0.95)
    v_max = v_limits.get("upper", 1.05)

    buses = sorted(voltages.keys())
    v_vals = [voltages[b] for b in buses]

    colors = [
        "#e74c3c" if (v > v_max or v < v_min) else "#2ecc71"
        for v in v_vals
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(buses) * 0.45), 5))
    ax.bar(buses, v_vals, color=colors, width=0.7, zorder=3)
    ax.axhline(v_max, color="#e74c3c", linewidth=1.2, linestyle="--", label=f"Upper limit ({v_max} pu)")
    ax.axhline(v_min, color="#e67e22", linewidth=1.2, linestyle="--", label=f"Lower limit ({v_min} pu)")
    ax.axhline(1.0, color="#95a5a6", linewidth=0.8, linestyle=":", label="Nominal (1.0 pu)")

    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title("Bus Voltage Profile — Snapshot")
    ax.tick_params(axis="x", rotation=70, labelsize=8 if len(buses) > 20 else 9)
    ax.set_ylim(min(0.85, min(v_vals) - 0.02), max(1.15, max(v_vals) + 0.02))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single OpenDSS power-flow snapshot and export results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML study config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_snapshot(args.config)
