"""
src/analysis/run_phase5_analysis.py
------------------------------------
CLI entry point for running the DERMS analysis end-to-end.

The analysis includes:
1. Running all simulations (baseline, heuristic, optimization)
2. Calculating and comparing KPIs
3. Generating comparison plots
4. Running hosting capacity study
5. Creating final report outputs

Usage:
    python -m src.analysis.run_phase5_analysis \\
        --baseline-config config/study_mvp.yaml \\
        --heuristic-config config/study_heuristic.yaml \\
        --optimization-config config/study_optimization.yaml \\
        --output results
"""

import argparse
import json
import pathlib
import shutil
import sys
from typing import Any

# Ensure project root is importable
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sim.run_qsts import run_qsts, export_qsts_results
from src.analysis.aggregator import generate_comparison_report
from src.analysis.dashboard import create_dashboard
from src.analysis.hosting_capacity import (
    compare_hosting_capacity,
    plot_hosting_capacity_comparison,
    plot_sweep_results,
    plot_voltage_vs_pv_scale,
)
from src.analysis.kpis import (
    calculate_all_kpis,
    calculate_battery_curtailment_reduction,
    calculate_battery_kpis,
)
from src.analysis.plots import (
    create_baseline_plots,
    create_comparison_plots,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------


def _get_available_results_dirs(
    simulation_results: dict[str, Any],
    required_modes: list[str],
) -> dict[str, str]:
    """Return result directories for modes with exported QSTS outputs."""
    available: dict[str, str] = {}
    missing: list[str] = []

    for mode in required_modes:
        mode_result = simulation_results.get(mode, {})
        results_dir = mode_result.get("results_dir")
        results_csv = pathlib.Path(results_dir) / "qsts_baseline.csv" if results_dir else None

        if results_dir and results_csv is not None and results_csv.exists():
            available[mode] = results_dir
        else:
            missing.append(mode)

    if missing:
        raise FileNotFoundError(
            "Missing simulation results for: "
            + ", ".join(missing)
        )

    return available


def _export_supporting_docs(
    standards_appendix: str | None,
    output_dir: pathlib.Path,
) -> dict[str, str]:
    """Copy supporting docs into the packaged output directory."""
    exported_docs: dict[str, str] = {}

    if not standards_appendix:
        return exported_docs

    appendix_path = _ROOT / standards_appendix
    if appendix_path.exists():
        docs_dir = output_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        target_path = docs_dir / appendix_path.name
        shutil.copy2(appendix_path, target_path)
        exported_docs["standards_alignment"] = str(target_path)

    return exported_docs


def _build_battery_summary(
    baseline_results: Any,
    heuristic_results: Any,
    battery_results: Any,
    output_dir: pathlib.Path,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict[str, Any]:
    """Create and persist a battery-specific summary package."""
    battery_dir = output_dir / "battery"
    battery_dir.mkdir(parents=True, exist_ok=True)

    baseline_voltage_kpis = calculate_all_kpis(baseline_results, v_min=v_min, v_max=v_max)
    heuristic_voltage_kpis = calculate_all_kpis(heuristic_results, v_min=v_min, v_max=v_max)
    battery_voltage_kpis = calculate_all_kpis(battery_results, v_min=v_min, v_max=v_max)
    battery_storage_kpis = calculate_battery_kpis(battery_results)
    vs_heuristic = calculate_battery_curtailment_reduction(heuristic_results, battery_results)

    summary = {
        "status": "success",
        "battery_voltage_kpis": battery_voltage_kpis,
        "battery_storage_kpis": battery_storage_kpis,
        "vs_baseline": {
            "violation_minutes_delta": float(
                battery_voltage_kpis.get("feeder_violation_minutes", 0.0)
                - baseline_voltage_kpis.get("feeder_violation_minutes", 0.0)
            ),
            "max_voltage_delta": float(
                battery_voltage_kpis.get("max_voltage", 0.0)
                - baseline_voltage_kpis.get("max_voltage", 0.0)
            ),
        },
        "vs_heuristic": {
            "violation_minutes_delta": float(
                battery_voltage_kpis.get("feeder_violation_minutes", 0.0)
                - heuristic_voltage_kpis.get("feeder_violation_minutes", 0.0)
            ),
            "max_voltage_delta": float(
                battery_voltage_kpis.get("max_voltage", 0.0)
                - heuristic_voltage_kpis.get("max_voltage", 0.0)
            ),
            **vs_heuristic,
        },
    }

    summary_path = battery_dir / "battery_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    summary["summary_path"] = str(summary_path)
    return summary


def run_full_phase5_analysis(
    baseline_config: str,
    heuristic_config: str,
    optimization_config: str,
    battery_config: str | None = None,
    output_dir: str = "results",
    run_hosting_capacity: bool = True,
    pv_scales: list[float] | None = None,
    skip_simulations: bool = False,
    generate_dashboard_output: bool = True,
    standards_appendix: str | None = "docs/standards_alignment.md",
) -> dict[str, Any]:
    """Run complete DERMS analysis.

    Args:
        baseline_config: Path to baseline study config
        heuristic_config: Path to heuristic study config
        optimization_config: Path to optimization study config
        battery_config: Optional path to battery study config
        output_dir: Root directory for mode folders and comparison outputs
        run_hosting_capacity: Whether to run hosting capacity study
        pv_scales: PV scale factors for hosting capacity sweep
        skip_simulations: Skip running simulations if results exist
        generate_dashboard_output: Whether to generate dashboard package
        standards_appendix: Optional supporting standards appendix path

    Returns:
        Dictionary with paths to all outputs
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "output_dir": str(output_dir),
        "simulations": {},
        "comparison": {},
        "hosting_capacity": {},
        "plots": {},
        "battery": {},
        "dashboard": {},
        "artifacts": {},
    }

    print(f"\n{'='*70}")
    print(f"  DERMS MVP — Analysis")
    print(f"{'='*70}")
    print(f"  Output directory: {output_dir}")
    print(f"  Hosting capacity: {'Enabled' if run_hosting_capacity else 'Disabled'}")
    print(f"{'='*70}\n")

    # -----------------------------------------------------------------------
    # Step 1: Run simulations
    # -----------------------------------------------------------------------
    if not skip_simulations:
        print("\n" + "="*70)
        print("  STEP 1: Running Simulations")
        print("="*70)

        configs = {
            "baseline": baseline_config,
            "heuristic": heuristic_config,
            "optimization": optimization_config,
        }
        if battery_config is not None:
            configs["battery"] = battery_config

        for mode, config_path in configs.items():
            print(f"\n--- Running {mode} simulation ---")

            # Load config to get output directory
            cfg = load_config(config_path)
            qsts_dir = cfg.get("output", {}).get("qsts_dir", f"results/{mode}")

            # Run simulation
            try:
                sim_results = run_qsts(config_path)

                # Export results
                out_dir = _ROOT / qsts_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                export_paths = export_qsts_results(sim_results, out_dir)

                results["simulations"][mode] = {
                    "status": "success",
                    "results_dir": str(out_dir),
                    "timesteps": len(sim_results),
                }

                print(f"  Completed: {len(sim_results)} timesteps")
                print(f"  Results: {out_dir}")

            except Exception as e:
                results["simulations"][mode] = {
                    "status": "failed",
                    "error": str(e),
                }
                print(f"  Failed: {e}")
    else:
        print("\nSkipping simulations (using existing results)")

        # Load config to find existing results
        configs = {
            "baseline": baseline_config,
            "heuristic": heuristic_config,
            "optimization": optimization_config,
        }
        if battery_config is not None:
            configs["battery"] = battery_config

        for mode, config_path in configs.items():
            cfg = load_config(config_path)
            qsts_dir = cfg.get("output", {}).get("qsts_dir", f"results/{mode}")
            results["simulations"][mode] = {
                "status": "existing",
                "results_dir": str(_ROOT / qsts_dir),
            }

    # -----------------------------------------------------------------------
    # Step 2: KPI comparison
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  STEP 2: KPI Comparison")
    print("="*70)

    try:
        results_dirs = _get_available_results_dirs(
            results["simulations"],
            ["baseline", "heuristic", "optimization"],
        )
        baseline_dir = results_dirs["baseline"]
        heuristic_dir = results_dirs["heuristic"]
        optimization_dir = results_dirs["optimization"]

        comparison_paths = generate_comparison_report(
            baseline_dir,
            heuristic_dir,
            optimization_dir,
            output_dir / "comparison",
        )

        results["comparison"] = {
            "status": "success",
            "paths": {k: str(v) for k, v in comparison_paths.items()},
        }

    except Exception as e:
        results["comparison"] = {
            "status": "failed",
            "error": str(e),
        }
        print(f"  KPI comparison failed: {e}")

    # -----------------------------------------------------------------------
    # Step 3: Generate comparison plots
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  STEP 3: Generating Comparison Plots")
    print("="*70)

    try:
        from src.analysis.aggregator import load_simulation_results

        results_dirs = _get_available_results_dirs(
            results["simulations"],
            ["baseline", "heuristic", "optimization"],
        )
        baseline_dir = results_dirs["baseline"]
        heuristic_dir = results_dirs["heuristic"]
        optimization_dir = results_dirs["optimization"]

        baseline_results = load_simulation_results(baseline_dir, "baseline")
        heuristic_results = load_simulation_results(heuristic_dir, "heuristic")
        optimization_results = load_simulation_results(optimization_dir, "optimization")
        battery_results = None
        if "battery" in results["simulations"]:
            try:
                battery_dir = _get_available_results_dirs(
                    results["simulations"],
                    ["battery"],
                )["battery"]
                battery_results = load_simulation_results(battery_dir, "battery")
            except FileNotFoundError:
                battery_results = None

        # Baseline plots
        print("\n  Generating baseline plots...")
        baseline_plots = create_baseline_plots(
            baseline_results,
            pathlib.Path(baseline_dir) / "plots",
            v_min=0.95,
            v_max=1.05,
        )

        # Heuristic comparison plots
        print("\n  Generating heuristic comparison plots...")
        heuristic_plots = create_comparison_plots(
            baseline_results,
            heuristic_results,
            pathlib.Path(heuristic_dir) / "plots",
            v_min=0.95,
            v_max=1.05,
        )

        # Optimization comparison plots
        print("\n  Generating optimization comparison plots...")
        opt_plots = create_comparison_plots(
            baseline_results,
            optimization_results,
            pathlib.Path(optimization_dir) / "plots",
            v_min=0.95,
            v_max=1.05,
        )

        results["plots"] = {
            "status": "success",
            "baseline": {k: str(v) for k, v in baseline_plots.items()},
            "heuristic_comparison": {k: str(v) for k, v in heuristic_plots.items()},
            "optimization_comparison": {k: str(v) for k, v in opt_plots.items()},
        }

        if battery_results is not None:
            print("\n  Generating battery comparison plots...")
            battery_plots = create_comparison_plots(
                heuristic_results,
                battery_results,
                pathlib.Path(battery_dir) / "plots",
                v_min=0.95,
                v_max=1.05,
            )
            results["plots"]["battery_comparison"] = {
                k: str(v) for k, v in battery_plots.items()
            }
            results["battery"] = _build_battery_summary(
                baseline_results,
                heuristic_results,
                battery_results,
                output_dir,
                v_min=0.95,
                v_max=1.05,
            )

        print("\n  Plots saved to mode directories")

    except Exception as e:
        results["plots"] = {
            "status": "failed",
            "error": str(e),
        }
        print(f"  Plot generation failed: {e}")

    # -----------------------------------------------------------------------
    # Step 4: Hosting capacity study
    # -----------------------------------------------------------------------
    if run_hosting_capacity:
        print("\n" + "="*70)
        print("  STEP 4: Hosting Capacity Study")
        print("="*70)

        if pv_scales is None:
            pv_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        print(f"  PV scales: {pv_scales}")

        try:
            hc_dir = output_dir / "comparison" / "hosting_capacity"
            mode_hc_dirs = {
                "baseline": pathlib.Path(baseline_config).parent.parent / "results" / "baseline" / "hosting_capacity",
                "heuristic": pathlib.Path(heuristic_config).parent.parent / "results" / "heuristic" / "hosting_capacity",
                "optimization": pathlib.Path(optimization_config).parent.parent / "results" / "optimization" / "hosting_capacity",
            }
            try:
                results_dirs = _get_available_results_dirs(
                    results["simulations"],
                    ["baseline", "heuristic", "optimization"],
                )
                mode_hc_dirs = {
                    mode: pathlib.Path(path) / "hosting_capacity"
                    for mode, path in results_dirs.items()
                }
            except FileNotFoundError:
                pass
            hc_results = compare_hosting_capacity(
                baseline_config,
                heuristic_config,
                optimization_config,
                pv_scales,
                hc_dir,
                mode_output_dirs=mode_hc_dirs,
            )

            # Generate hosting capacity comparison plots
            plot_dir = hc_dir
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Comparison bar chart
            plot_hosting_capacity_comparison(
                hc_results,
                plot_dir / "hosting_capacity_comparison.png",
            )

            # Sweep result plots for each mode
            for mode in ["baseline", "heuristic", "optimization"]:
                if mode in hc_results and "sweep_results" in hc_results[mode]:
                    plot_sweep_results(
                        hc_results[mode]["sweep_results"],
                        plot_dir / f"{mode}_sweep_results.png",
                    )

            # Voltage vs scale plot
            plot_voltage_vs_pv_scale(
                hc_results,
                plot_dir / "voltage_vs_pv_scale.png",
            )

            results["hosting_capacity"] = {
                "status": "success",
                "results_dir": str(hc_dir),
                "improvements": hc_results.get("improvements", {}),
            }

            # Print summary
            print("\n  Hosting Capacity Summary:")
            for mode in ["baseline", "heuristic", "optimization"]:
                if mode in hc_results:
                    cap = hc_results[mode]["capacity_info"]["hosting_capacity"]
                    print(f"    {mode.capitalize()}: {cap:.2f}x")

            for key, val in hc_results.get("improvements", {}).items():
                print(f"    {key.replace('_', ' ').title()}: {val:.2f}x")

        except Exception as e:
            results["hosting_capacity"] = {
                "status": "failed",
                "error": str(e),
            }
            print(f"  Hosting capacity study failed: {e}")

    # -----------------------------------------------------------------------
    # Step 4.5: Package stretch outputs
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  STEP 4.5: Packaging Stretch Outputs")
    print("="*70)

    exported_docs = _export_supporting_docs(standards_appendix, output_dir)
    if exported_docs:
        results["artifacts"]["docs"] = exported_docs

    if generate_dashboard_output:
        try:
            dashboard_path = pathlib.Path(
                create_dashboard(output_dir, output_dir / "dashboard.html")
            )
            results["dashboard"] = {
                "status": "success",
                "path": str(dashboard_path),
            }
            print(f"  Dashboard: {dashboard_path}")
        except Exception as e:
            results["dashboard"] = {
                "status": "failed",
                "error": str(e),
            }
            print(f"  Dashboard packaging failed: {e}")

    # -----------------------------------------------------------------------
    # Step 5: Create final summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  STEP 5: Creating Final Summary")
    print("="*70)

    summary_dir = output_dir / "comparison"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "analysis_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Summary saved: {summary_path}")

    # Print completion message
    print("\n" + "="*70)
    print("  Analysis Complete")
    print("="*70)
    print(f"\n  Output directory: {output_dir}")
    print(f"\n  Generated files:")
    print(f"    - KPI comparison: {output_dir / 'comparison'}")
    print("    - Plots: mode directories")
    if run_hosting_capacity:
        print(f"    - Hosting capacity summary: {output_dir / 'comparison' / 'hosting_capacity'}")
        print("    - Hosting capacity sweeps: mode directories")
    if results.get("battery", {}).get("status") == "success":
        print(f"    - Battery summary: {results['battery']['summary_path']}")
    if results.get("dashboard", {}).get("status") == "success":
        print(f"    - Dashboard: {results['dashboard']['path']}")
    if results.get("artifacts", {}).get("docs", {}).get("standards_alignment"):
        print(f"    - Standards appendix: {results['artifacts']['docs']['standards_alignment']}")
    print(f"    - Summary: {summary_path}")
    print("="*70 + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for DERMS analysis."""
    parser = argparse.ArgumentParser(
        description="Run DERMS analysis across baseline, heuristic, and optimization modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--baseline-config",
        type=str,
        default="config/study_mvp.yaml",
        help="Path to baseline study config.",
    )

    parser.add_argument(
        "--heuristic-config",
        type=str,
        default="config/study_heuristic.yaml",
        help="Path to heuristic study config.",
    )

    parser.add_argument(
        "--optimization-config",
        type=str,
        default="config/study_optimization.yaml",
        help="Path to optimization study config.",
    )

    parser.add_argument(
        "--battery-config",
        type=str,
        default=None,
        help="Optional path to battery study config for Phase 6 outputs.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Root output directory containing mode folders and comparison outputs.",
    )

    parser.add_argument(
        "--skip-hosting-capacity",
        action="store_true",
        help="Skip hosting capacity study.",
    )

    parser.add_argument(
        "--pv-scales",
        type=str,
        default=None,
        help="Comma-separated list of PV scale factors for hosting capacity.",
    )

    parser.add_argument(
        "--skip-simulations",
        action="store_true",
        help="Skip running simulations if results exist.",
    )

    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip generating the interactive dashboard package.",
    )

    parser.add_argument(
        "--standards-appendix",
        type=str,
        default="docs/standards_alignment.md",
        help="Optional standards appendix to copy into the output package.",
    )

    args = parser.parse_args()

    # Parse PV scales
    pv_scales = None
    if args.pv_scales:
        pv_scales = [float(x) for x in args.pv_scales.split(",")]

    # Run analysis
    run_full_phase5_analysis(
        baseline_config=args.baseline_config,
        heuristic_config=args.heuristic_config,
        optimization_config=args.optimization_config,
        battery_config=args.battery_config,
        output_dir=args.output,
        run_hosting_capacity=not args.skip_hosting_capacity,
        pv_scales=pv_scales,
        skip_simulations=args.skip_simulations,
        generate_dashboard_output=not args.skip_dashboard,
        standards_appendix=args.standards_appendix,
    )


if __name__ == "__main__":
    main()
