"""
src/sim/run_qsts.py
-------------------
CLI entry-point for running 24-hour quasi-static time-series (QSTS) simulation.

Usage:
    python -m src.sim.run_qsts --config config/study_mvp.yaml

This script:
- Loads load and PV profiles
- Places PV systems on the feeder
- Runs a 24-hour simulation at 5-minute steps (288 timesteps)
- At each step: updates load/PV, solves power flow, stores metrics
- Exports results CSV and generates plots
"""

import argparse
import pathlib
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is importable
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sim.opendss_interface import (
    load_feeder,
    solve_power_flow,
    get_bus_voltages,
    get_circuit_summary,
    _require_opendss,
)
from src.sim.pv_integration import (
    place_pv_list,
    set_all_pv_output,
    get_total_pv_capacity_kw,
    get_default_pv_placement,
)
from src.sim.battery_interface import (
    place_battery_list,
    apply_battery_power_dispatch,
    get_battery_names,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.analysis.plots import create_baseline_plots

# Control layer imports
from src.control import (
    DERContainer,
    load_ders_from_csv,
    read_der_state,
    apply_setpoints,
    CommandLogger,
    summarize_command_log,
    HeuristicController,
    HeuristicConfig,
    OptimizationController,
    OptimizationConfig,
    SensitivityConfig,
    Battery,
    BatteryContainer,
    BatteryController,
    BatteryControlConfig,
)

try:
    import opendssdirect as dss
    OPENDSS_AVAILABLE = True
except ImportError:
    dss = None
    OPENDSS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Profile Loading
# ---------------------------------------------------------------------------


_LOAD_BASE_SNAPSHOT: dict[str, tuple[float, float]] = {}


def _snapshot_der_states(der_container: DERContainer) -> dict[str, dict[str, float]]:
    """Capture DER runtime state for command audit rows."""
    return {
        der.id: {
            "p_avail_kw": der.p_avail_kw,
            "p_dispatch_kw": der.p_dispatch_kw,
            "q_kvar": der.q_kvar,
            "v_local_pu": der.v_local_pu,
        }
        for der in der_container.ders
    }


def load_profile(
    profile_path: str | pathlib.Path,
    time_step_minutes: int = 5,
    value_column: str | None = None,
) -> dict[float, float]:
    """Load a time-series profile from CSV.

    Args:
        profile_path: Path to CSV with columns (time_h, value)
        time_step_minutes: Resolution for interpolation

    Returns:
        Dict mapping time in minutes to profile value
    """
    df = pd.read_csv(profile_path)
    value_col = value_column or df.columns[1]

    # Create continuous mapping at specified resolution
    profile = {}
    for t_min in range(0, 24 * 60, time_step_minutes):
        t_hour = t_min / 60.0

        # Find the surrounding data points for interpolation
        if t_hour <= df["time_h"].iloc[0]:
            value = df.iloc[0][value_col]
        elif t_hour >= df["time_h"].iloc[-1]:
            value = df.iloc[-1][value_col]
        else:
            # Linear interpolation
            idx = np.searchsorted(df["time_h"].values, t_hour)
            x0 = df["time_h"].iloc[idx - 1]
            x1 = df["time_h"].iloc[idx]
            y0 = df.iloc[idx - 1][value_col]
            y1 = df.iloc[idx][value_col]
            value = y0 + (y1 - y0) * (t_hour - x0) / (x1 - x0)

        profile[t_min] = float(value)

    return profile


def load_load_profile(
    profile_path: str | pathlib.Path,
    time_step_minutes: int = 5,
) -> dict[float, float]:
    """Load load multiplier profile."""
    return load_profile(profile_path, time_step_minutes, value_column="load_multiplier")


def load_pv_profile(
    profile_path: str | pathlib.Path,
    time_step_minutes: int = 5,
) -> dict[float, float]:
    """Load PV production profile (returns pu value 0-1)."""
    df = pd.read_csv(profile_path, nrows=1)
    value_column = "pv_production_pu" if "pv_production_pu" in df.columns else df.columns[1]
    return load_profile(profile_path, time_step_minutes, value_column=value_column)


# ---------------------------------------------------------------------------
# Load/PV Update Functions
# ---------------------------------------------------------------------------


def update_loads(load_multiplier: float) -> None:
    """Update all loads in the circuit by a multiplier.

    Args:
        load_multiplier: Multiplier applied to all load kW values
    """
    _require_opendss()
    dss.Circuit.SetActiveClass("Load")

    has_next = dss.Loads.First()
    while has_next:
        # Get current kw
        kw_current = dss.Loads.kW()
        # Apply multiplier
        dss.Loads.kW(kw_current * load_multiplier)
        has_next = dss.Loads.Next()


def get_original_load_totals() -> dict[str, float]:
    """Get the original total load kW and kVAR from the circuit.

    Returns:
        Dict with 'kw' and 'kvar' totals
    """
    _require_opendss()

    total_kw = 0.0
    total_kvar = 0.0
    _LOAD_BASE_SNAPSHOT.clear()

    dss.Circuit.SetActiveClass("Load")
    has_next = dss.Loads.First()
    while has_next:
        kw = dss.Loads.kW()
        kvar = dss.Loads.kvar()
        _LOAD_BASE_SNAPSHOT[dss.Loads.Name()] = (kw, kvar)
        total_kw += kw
        total_kvar += kvar
        has_next = dss.Loads.Next()

    return {"kw": total_kw, "kvar": total_kvar}


def update_loads_from_base(
    base_kw: float,
    base_kvar: float,
    load_multiplier: float,
) -> None:
    """Update loads from a base level with multiplier.

    Args:
        base_kw: Base total load kW
        base_kvar: Base total load kVAR
        load_multiplier: Multiplier to apply
    """
    _require_opendss()

    current_total_kw = base_kw * load_multiplier

    # Scale all loads proportionally
    dss.Circuit.SetActiveClass("Load")
    has_next = dss.Loads.First()
    while has_next:
        load_name = dss.Loads.Name()
        original_kw, original_kvar = _LOAD_BASE_SNAPSHOT.get(
            load_name,
            (dss.Loads.kW(), dss.Loads.kvar()),
        )

        # Calculate proportion
        proportion = original_kw / base_kw if base_kw > 0 else 0.0

        # Set new values
        new_kw = current_total_kw * proportion
        # Maintain same power factor
        pf = original_kw / (original_kw**2 + original_kvar**2)**0.5 if original_kw > 0 else 0.9
        new_kvar = new_kw * ((1/pf**2) - 1)**0.5 if pf < 1.0 else new_kw * 0.2

        dss.Loads.kW(new_kw)
        dss.Loads.kvar(new_kvar)
        
        has_next = dss.Loads.Next()


def update_pv(pv_multiplier: float) -> None:
    """Update all PV systems to a production level.

    Args:
        pv_multiplier: Production level (0-1) of rated capacity
    """
    _require_opendss()

    total_capacity = get_total_pv_capacity_kw()
    if total_capacity > 0:
        set_all_pv_output(total_capacity * pv_multiplier, power_factor=1.0)


# ---------------------------------------------------------------------------
# Battery Loading Functions
# ---------------------------------------------------------------------------


def load_batteries_from_csv(
    csv_path: str | pathlib.Path,
    battery_config: BatteryControlConfig | None = None,
) -> BatteryContainer:
    """Load battery definitions from a CSV file.

    Args:
        csv_path: Path to CSV file with battery definitions
        battery_config: Optional battery control config (uses defaults if None)

    CSV format:
        id,bus,phases,capacity_kwh,power_kw,soc_init,control_enabled

    Returns:
        BatteryContainer with all loaded batteries
    """
    df = pd.read_csv(csv_path)

    batteries = []
    for _, row in df.iterrows():
        battery = Battery(
            id=str(row["id"]),
            bus=str(row["bus"]),
            phases=int(row["phases"]),
            capacity_kwh=float(row["capacity_kwh"]),
            power_limit_kw=float(row["power_kw"]),
            soc=float(row.get("soc_init", battery_config.soc_init if battery_config else 0.5)),
            soc_min=battery_config.soc_min if battery_config else 0.1,
            soc_max=battery_config.soc_max if battery_config else 0.9,
            efficiency=battery_config.efficiency if battery_config else 0.95,
            control_enabled=bool(row.get("control_enabled", True)),
        )
        batteries.append(battery)

    return BatteryContainer(batteries)


# ---------------------------------------------------------------------------
# QSTS Simulation Loop
# ---------------------------------------------------------------------------


def run_qsts(
    config_path: str | pathlib.Path,
    pv_scale: float = 4.0,  # Increased from 1.0 to create overvoltage scenario for DERMS demonstration
) -> pd.DataFrame:
    """Run a 24-hour QSTS simulation.

    Args:
        config_path: Path to study config YAML
        pv_scale: Scale factor for PV capacity (for testing scenarios)

    Returns:
        DataFrame with timestep results
    """
    cfg = load_config(config_path)
    
    # Define project root (assuming configs are in project_root/config/)
    project_root = pathlib.Path(config_path).resolve().parent.parent

    # Load feeder config
    feeder_cfg_path = project_root / cfg["feeder_config"]
    feeder_cfg = load_config(feeder_cfg_path)

    # Load profiles
    profile_cfg = cfg.get("profiles", {})
    load_profile_path = project_root / profile_cfg["load_profile"]
    pv_profile_path = project_root / profile_cfg["pv_profile"]

    time_step = feeder_cfg["simulation"]["time_step_minutes"]
    load_profile = load_load_profile(load_profile_path, time_step)
    pv_profile = load_pv_profile(pv_profile_path, time_step)

    # Voltage limits
    v_limits = feeder_cfg.get("voltage_limits", {})
    v_min = v_limits.get("lower", 0.95)
    v_max = v_limits.get("upper", 1.05)
    detailed_voltage_logging = bool(cfg.get("output", {}).get("detailed_voltage_logging", False))

    # Get DER placement
    der_cfg = profile_cfg.get("der_config", "") or cfg.get("der_config", "")
    if der_cfg:
        der_path = project_root / der_cfg
        pv_df = pd.read_csv(der_path)
        if "bus" in pv_df.columns:
            pv_df["bus"] = pv_df["bus"].astype(str)
        pv_list = pv_df.to_dict("records")
    else:
        # Use default placement based on feeder name
        feeder_name = feeder_cfg["feeder"]["name"]
        pv_list = get_default_pv_placement(feeder_name)

    # Resolve feeder path
    config_dir = pathlib.Path(config_path).resolve().parent
    dss_master = config_dir.parent / feeder_cfg["feeder"]["master_file"]

    print(f"\n{'='*60}")
    print(f"  DERMS MVP — 24-Hour QSTS Simulation")
    print(f"{'='*60}")
    print(f"  Config      : {config_path}")
    print(f"  Feeder      : {feeder_cfg['feeder']['name']}")
    print(f"  Master file : {dss_master}")
    print(f"  PV systems  : {len(pv_list)}")
    print(f"  PV scale    : {pv_scale:.2f}x")
    print(f"  Time step   : {time_step} minutes")
    print(f"{'='*60}\n")

    # Initialize OpenDSS
    print("Initializing OpenDSS...")
    load_feeder(dss_master)

    # Get base load totals
    base_loads = get_original_load_totals()
    print(f"Base load: {base_loads['kw']:.1f} kW, {base_loads['kvar']:.1f} kVAR")

    # Place PV systems
    print(f"Placing {len(pv_list)} PV systems...")
    pv_ids = place_pv_list(pv_list, scale_factor=pv_scale)
    total_pv_kw = get_total_pv_capacity_kw()
    print(f"Total PV capacity: {total_pv_kw:.1f} kW")

    # Initialize batteries if specified
    battery_container = None
    battery_controller = None
    battery_cfg_path = profile_cfg.get("battery_config", "")
    if battery_cfg_path:
        battery_path = project_root / battery_cfg_path
        print(f"Loading battery config from: {battery_path}")

        # Load battery control config from YAML
        battery_cfg_dict = cfg.get("battery", {})
        battery_control_config = BatteryControlConfig(
            capacity_kwh=battery_cfg_dict.get("capacity_kwh", 200.0),
            power_kw=battery_cfg_dict.get("power_kw", 100.0),
            efficiency=battery_cfg_dict.get("efficiency", 0.95),
            soc_init=battery_cfg_dict.get("soc_init", 0.5),
            soc_min=battery_cfg_dict.get("soc_min", 0.1),
            soc_max=battery_cfg_dict.get("soc_max", 0.9),
            charge_threshold_pv_excess_pct=battery_cfg_dict.get(
                "charge_threshold_pv_excess_pct", 0.2
            ),
            discharge_voltage_pu=battery_cfg_dict.get("discharge_voltage_pu", 1.04),
            charge_window_start_h=battery_cfg_dict.get("charge_window_start_h", 8),
            charge_window_end_h=battery_cfg_dict.get("charge_window_end_h", 16),
            prioritize_voltage_support=battery_cfg_dict.get(
                "prioritize_voltage_support", True
            ),
        )

        # Load battery definitions
        battery_container = load_batteries_from_csv(battery_path, battery_control_config)
        print(f"Loaded {len(battery_container)} batteries")
        print(f"  Total capacity: {battery_container.total_capacity_kwh():.1f} kWh")
        print(f"  Total power: {battery_container.total_power_kw():.1f} kW")

        # Place batteries in OpenDSS
        print("Placing batteries in OpenDSS...")
        battery_list = pd.read_csv(battery_path).to_dict("records")
        battery_ids = place_battery_list([
            {
                "bus": b["bus"],
                "phases": b["phases"],
                "capacity_kwh": b["capacity_kwh"],
                "power_kw": b["power_kw"],
            }
            for b in battery_list
        ])
        print(f"  Placed {len(battery_ids)} battery systems")

        # Initialize battery controller
        battery_controller = BatteryController(battery_control_config, battery_container)

    # Initialize controller if needed
    der_container = None
    controller = None
    command_logger = None
    controller_mode = cfg.get("controller", {}).get("mode", "baseline")
    print(f"Controller mode: {controller_mode}")

    if controller_mode in ("heuristic", "optimization"):
        der_path = project_root / der_cfg
        der_container = load_ders_from_csv(der_path, scale_factor=pv_scale)
        print(f"Loaded {len(der_container)} DERs")

        # Set up output directory for commands
        out_dir_str = cfg.get("output", {}).get("qsts_dir", "results/baseline")
        out_dir = (_ROOT / out_dir_str).resolve()
        command_logger = CommandLogger(out_dir / "commands.csv")

        if controller_mode == "heuristic":
            control_cfg = cfg.get("control_thresholds", {})
            heuristic_config = HeuristicConfig(
                q_activation_pu=control_cfg.get("q_activation_pu", 1.03),
                curtailment_pu=control_cfg.get("curtailment_pu", 1.05),
                deadband_pu=control_cfg.get("deadband_pu", 0.005),
                q_ramp_max_kvar=control_cfg.get("q_ramp_max_kvar", float('inf')),
                p_ramp_max_kw=control_cfg.get("p_ramp_max_kw", float('inf')),
                v_lower_limit=v_limits.get("lower", 0.95),
            )
            controller = HeuristicController(heuristic_config, der_container)
            print(f"  Q activation: {heuristic_config.q_activation_pu} pu")
            print(f"  Curtailment: {heuristic_config.curtailment_pu} pu")
            print(f"  Deadband: ±{heuristic_config.deadband_pu} pu")

        elif controller_mode == "optimization":
            from src.control import (
                OptimizationController, OptimizationConfig,
                SensitivityConfig,
            )

            # Load optimization config
            opt_cfg_dict = cfg.get("optimization", {})
            sens_cfg_dict = opt_cfg_dict.get("sensitivity", {})

            # Create heuristic fallback
            control_cfg = cfg.get("control_thresholds", {})
            heuristic_config = HeuristicConfig(
                q_activation_pu=control_cfg.get("q_activation_pu", 1.03),
                curtailment_pu=control_cfg.get("curtailment_pu", 1.05),
                deadband_pu=control_cfg.get("deadband_pu", 0.005),
                q_ramp_max_kvar=control_cfg.get("q_ramp_max_kvar", float('inf')),
                p_ramp_max_kw=control_cfg.get("p_ramp_max_kw", float('inf')),
                v_lower_limit=v_limits.get("lower", 0.95),
            )
            heuristic_fallback = HeuristicController(heuristic_config, der_container)

            # Create sensitivity config
            sensitivity_config = SensitivityConfig(
                q_perturbation_pct=sens_cfg_dict.get("q_perturbation_pct", 0.10),
                p_perturbation_pct=sens_cfg_dict.get("p_perturbation_pct", 0.05),
                min_perturbation=sens_cfg_dict.get("min_perturbation", 1.0),
                cache_sensitivities=sens_cfg_dict.get("cache_sensitivities", True),
                cache_valid_minutes=sens_cfg_dict.get("cache_valid_minutes", 30),
            )

            # Create optimization controller
            opt_config = OptimizationConfig(
                alpha_violation=opt_cfg_dict.get("alpha_violation", 1000.0),
                beta_q_effort=opt_cfg_dict.get("beta_q_effort", 1.0),
                gamma_p_curtail=opt_cfg_dict.get("gamma_p_curtail", 100.0),
                v_min=v_limits.get("lower", 0.95),
                v_max=v_limits.get("upper", 1.05),
                solver=opt_cfg_dict.get("solver", "ECOS"),
                max_iterations=opt_cfg_dict.get("max_iterations", 1000),
                tolerance=opt_cfg_dict.get("tolerance", 1e-6),
                enable_fallback=opt_cfg_dict.get("enable_fallback", True),
                sensitivity_config=sensitivity_config,
            )
            controller = OptimizationController(opt_config, der_container, heuristic_fallback)

            print(f"  Solver: {opt_config.solver}")
            print(f"  Alpha (violation): {opt_config.alpha_violation}")
            print(f"  Beta (Q effort): {opt_config.beta_q_effort}")
            print(f"  Gamma (P curtail): {opt_config.gamma_p_curtail}")
            print(f"  Fallback enabled: {opt_config.enable_fallback}")
            print(f"  Sensitivity cache: {sensitivity_config.cache_sensitivities}")

    # Prepare results storage
    results = []
    num_steps = len(load_profile)

    # Track previous commands for ramp limiting
    previous_q_and_p: dict[str, tuple[float, float]] = {}

    print(f"\nRunning {num_steps} timesteps...")
    print(f"[{'='*50}]")

    for step, (t_min, load_mult) in enumerate(sorted(load_profile.items())):
        t_hour = t_min / 60.0
        pv_mult = pv_profile.get(t_min, 0.0)

        # Update loads and PV
        update_loads_from_base(base_loads["kw"], base_loads["kvar"], load_mult)
        update_pv(pv_mult)

        # Solve power flow
        try:
            solve_power_flow()
        except RuntimeError as e:
            print(f"\nWarning: Solve failed at t={t_hour:.2f}h: {e}")
            continue

        # Get results before control
        voltages_before = get_bus_voltages()

        # Apply control if enabled
        q_commands: dict[str, float] = {}
        p_commands: dict[str, float] = {}
        battery_commands: dict[str, float] = {}

        if controller is not None and der_container is not None:
            # Read DER states from OpenDSS
            read_der_state(der_container)
            der_states_before = _snapshot_der_states(der_container)

            # Compute commands
            q_commands, p_commands = controller.compute_commands(
                voltages_before,
                previous_commands=previous_q_and_p if previous_q_and_p else None,
            )

            # Store for next timestep's ramp limiting
            previous_q_and_p = {
                der_id: (q_commands.get(der_id, 0), p_commands.get(der_id, 0))
                for der_id in der_container.enabled_ids()
            }

            # Apply commands to OpenDSS
            apply_results = apply_setpoints(der_container, q_commands, p_commands)

            # Re-solve power flow after control. If the controlled case is
            # infeasible, revert to no-control for this timestep instead of
            # aborting the whole simulation run.
            try:
                solve_power_flow()
                voltages_after = get_bus_voltages()
                read_der_state(der_container)
                der_states_after = _snapshot_der_states(der_container)
            except RuntimeError as e:
                print(f"\nWarning: Controlled solve failed at t={t_hour:.2f}h: {e}. Reverting commands.")
                q_commands = {der.id: 0.0 for der in der_container.enabled()}
                p_commands = {der.id: 0.0 for der in der_container.enabled()}
                apply_results = apply_setpoints(der_container, q_commands, p_commands)
                solve_power_flow()
                voltages_after = get_bus_voltages()
                read_der_state(der_container)
                der_states_after = _snapshot_der_states(der_container)

            # Log commands
            if command_logger is not None:
                command_logger.log_batch(
                    der_container, q_commands, p_commands,
                    apply_results, datetime.now(),
                    voltages_before, voltages_after,
                    step=step,
                    time_min=t_min,
                    time_h=t_hour,
                    controller_mode=controller_mode,
                    controller_status=getattr(controller, "last_status", "N/A"),
                    command_source=controller_mode,
                    states_before=der_states_before,
                    states_after=der_states_after,
                )

            # Use post-control voltages for metrics
            voltages = voltages_after
        else:
            voltages = voltages_before

        # Apply battery control if enabled
        if battery_controller is not None and battery_container is not None:
            # Compute battery commands
            # Use simplified voltage-only interface for coordination
            battery_commands = battery_controller.compute_battery_commands_simple(
                voltages,
                previous_commands=None,  # No ramp limiting for batteries
            )

            # Apply battery commands to OpenDSS
            if battery_commands:
                apply_battery_power_dispatch(battery_commands)

                # Update battery state
                time_step_hours = time_step / 60.0
                battery_controller.update_battery_state(
                    battery_commands,
                    time_step_hours,
                )

                # Re-solve power flow after battery action
                try:
                    solve_power_flow()
                    voltages = get_bus_voltages()
                except RuntimeError as e:
                    print(f"\nWarning: Solve failed after battery at t={t_hour:.2f}h: {e}")

        # Calculate metrics
        v_vals = list(voltages.values())
        v_mean = np.mean(v_vals)
        v_min_bus = min(voltages, key=voltages.get)
        v_max_bus = max(voltages, key=voltages.get)

        # Count violations
        violating_buses = [b for b, v in voltages.items() if v > v_max or v < v_min]
        overvoltage_buses = [b for b, v in voltages.items() if v > v_max]
        undervoltage_buses = [b for b, v in voltages.items() if v < v_min]

        # Get circuit losses
        dss.Solution.Solve()
        losses_kw = dss.Circuit.Losses()[0] / 1000.0  # Convert W to kW

        result = {
            "step": step,
            "time_min": t_min,
            "time_h": t_hour,
            "load_multiplier": load_mult,
            "pv_multiplier": pv_mult,
            "pv_generation_kw": total_pv_kw * pv_mult,
            "v_min": min(v_vals),
            "v_max": max(v_vals),
            "v_mean": v_mean,
            "v_min_bus": v_min_bus,
            "v_max_bus": v_max_bus,
            "violating_buses_count": len(violating_buses),
            "overvoltage_buses_count": len(overvoltage_buses),
            "undervoltage_buses_count": len(undervoltage_buses),
            "losses_kw": losses_kw,
            "control_mode": controller_mode,
            "total_q_dispatch_kvar": sum(abs(q) for q in q_commands.values()),
            "total_p_curtailment_kw": sum(p_commands.values()),
            "ders_controlled": len([d for d in q_commands if d != 0]),
        }

        # Add battery metrics
        if battery_container is not None:
            result["battery_count"] = len(battery_container)
            result["battery_soc_mean"] = battery_container.average_soc()
            result["battery_energy_kwh"] = battery_container.total_energy_kwh()
            # Total battery power (positive = discharge, negative = charge)
            result["total_battery_power_kw"] = sum(
                b.power_kw for b in battery_container.batteries
            )

        # Add optimization-specific metrics if applicable
        if controller_mode == "optimization" and controller is not None:
            result["optimization_status"] = getattr(controller, "last_status", "N/A")
            result["sensitivity_cache_hit"] = getattr(controller, "cache_hit", False)

        # Store per-bus voltages. By default keep the export sparse for
        # compact files, but allow full bus traces for dashboards/debugging.
        buses_to_log = voltages.keys() if detailed_voltage_logging else overvoltage_buses
        for bus in buses_to_log:
            result[f"overv_{bus}"] = voltages[bus]

        results.append(result)

        # Progress bar
        if step % 30 == 0 or step == num_steps - 1:
            pct = int(50 * (step + 1) / num_steps)
            print(f"[{'='*pct}{'.'*(50-pct)}] {step+1}/{num_steps}", end="\r")

    print(f"\n\nSimulation complete!")

    # Save command log if active
    if command_logger is not None:
        command_log_path = command_logger.save()
        print(f"Command log saved to: {command_logger.output_path}")
        command_summary = summarize_command_log(
            command_log_path,
            time_step_minutes=time_step,
        )
        command_summary_path = command_logger.output_path.with_name("command_summary.csv")
        pd.DataFrame(
            {
                "metric": list(command_summary.keys()),
                "value": list(command_summary.values()),
            }
        ).to_csv(command_summary_path, index=False)
        print(f"Command summary saved to: {command_summary_path}")

    # Print battery summary if active
    if battery_controller is not None:
        summary = battery_controller.get_battery_summary()
        print(f"\nBattery Summary:")
        print(f"  Total throughput: {summary['total_throughput_kwh']:.1f} kWh")
        print(f"  Final energy: {summary['total_energy_kwh']:.1f} kWh")
        print(f"  Average SOC: {summary['average_soc']:.2%}")
        print(f"  Equivalent cycles: {summary['total_cycles']:.2f}")

    # Reset the OpenDSS circuit without turning cleanup issues into a failed run.
    try:
        dss.run_command("Clear")
    except Exception as e:
        print(f"Warning: Failed to reset OpenDSS state: {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Results Export
# ---------------------------------------------------------------------------


def export_qsts_results(
    results: pd.DataFrame,
    output_dir: str | pathlib.Path,
) -> dict[str, pathlib.Path]:
    """Export QSTS results to CSV files.

    Args:
        results: Results DataFrame from run_qsts
        output_dir: Directory for outputs

    Returns:
        Dict of output file paths
    """
    out_dir = pathlib.Path(output_dir)
    ensure_dir(out_dir)

    paths = {}

    # Main results CSV
    main_csv = out_dir / "qsts_baseline.csv"
    results.to_csv(main_csv, index=False)
    paths["main"] = main_csv

    # Summary CSV
    summary = {
        "metric": [],
        "value": [],
    }

    for col in ["v_min", "v_max", "v_mean", "violating_buses_count",
                "overvoltage_buses_count", "losses_kw", "pv_generation_kw"]:
        if col in results.columns:
            summary["metric"].extend([f"{col}_min", f"{col}_max", f"{col}_mean"])
            summary["value"].extend([
                results[col].min(),
                results[col].max(),
                results[col].mean(),
            ])

    summary_df = pd.DataFrame(summary)
    summary_csv = out_dir / "qsts_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    paths["summary"] = summary_csv

    # Violation timeline
    violation_mask = results["violating_buses_count"] > 0
    if violation_mask.any():
        violation_csv = out_dir / "qsts_violations.csv"
        results[violation_mask].to_csv(violation_csv, index=False)
        paths["violations"] = violation_csv

    return paths


def print_summary(results: pd.DataFrame, v_max_limit: float = 1.05) -> None:
    """Print a summary of the QSTS results.

    Args:
        results: Results DataFrame
        v_max_limit: Upper voltage limit for violation counting
    """
    print(f"\n{'='*60}")
    print(f"  QSTS Simulation Summary")
    print(f"{'='*60}")

    # Voltage statistics
    print(f"\nVoltage Statistics:")
    print(f"  Max voltage : {results['v_max'].max():.4f} pu @ {results.loc[results['v_max'].idxmax(), 'time_h']:.2f}h")
    print(f"  Min voltage : {results['v_min'].min():.4f} pu @ {results.loc[results['v_min'].idxmin(), 'time_h']:.2f}h")
    print(f"  Mean voltage: {results['v_mean'].mean():.4f} pu")

    # Violation statistics
    overvoltage_steps = (results["v_max"] > v_max_limit).sum()
    violating_steps = (results["violating_buses_count"] > 0).sum()
    total_steps = len(results)

    print(f"\nViolation Statistics:")
    print(f"  Steps with overvoltage      : {overvoltage_steps}/{total_steps} ({100*overvoltage_steps/total_steps:.1f}%)")
    print(f"  Steps with any violation    : {violating_steps}/{total_steps} ({100*violating_steps/total_steps:.1f}%)")

    if violating_steps > 0:
        violation_minutes = violating_steps * 5  # 5-minute steps
        print(f"  Total violation minutes     : {violation_minutes}")

    # Worst buses
    if "v_max_bus" in results.columns:
        from collections import Counter
        worst_buses = Counter(results["v_max_bus"].tolist())
        print(f"\nTop 5 Worst Buses (by max voltage):")
        for bus, count in worst_buses.most_common(5):
            max_v = results[results["v_max_bus"] == bus]["v_max"].max()
            print(f"  {bus}: {max_v:.4f} pu (max)")

    # Energy statistics
    total_pv_kwh = (results["pv_generation_kw"] * 5 / 60).sum()
    total_losses_kwh = (results["losses_kw"] * 5 / 60).sum()

    print(f"\nEnergy Statistics:")
    print(f"  Total PV generation  : {total_pv_kwh:.1f} kWh")
    print(f"  Total losses         : {total_losses_kwh:.1f} kWh")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    results = run_qsts(args.config, pv_scale=args.pv_scale or 1.0)

    # Export results
    cfg = load_config(args.config)
    out_dir_str = cfg.get("output", {}).get("qsts_dir", "results/baseline")
    out_dir = (_ROOT / out_dir_str).resolve()

    paths = export_qsts_results(results, out_dir)

    print(f"\nResults exported:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Print summary
    feeder_cfg_path = (_ROOT / cfg["feeder_config"]).resolve()
    feeder_cfg = load_config(feeder_cfg_path)
    v_limits = feeder_cfg.get("voltage_limits", {})
    v_min = v_limits.get("lower", 0.95)
    v_max = v_limits.get("upper", 1.05)

    print_summary(results, v_max)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating baseline plots...")
        plot_paths = create_baseline_plots(
            results,
            out_dir,
            v_min=v_min,
            v_max=v_max,
        )
        print(f"\nPlots generated:")
        for name, path in plot_paths.items():
            print(f"  {name}: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 24-hour QSTS baseline simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/study_mvp.yaml",
        help="Path to the YAML study config file.",
    )
    parser.add_argument(
        "--pv-scale",
        type=float,
        default=None,
        help="Scale factor for PV capacity (e.g., 2.0 = 2x default PV).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
