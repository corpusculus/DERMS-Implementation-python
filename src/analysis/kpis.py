"""
src/analysis/kpis.py
---------------------
Key Performance Indicator (KPI) calculations for QSTS simulation results.

Provides standardized metrics for:
- Voltage performance (violations, deviations)
- Control actions (reactive power, curtailment)
- System performance (losses, generation)
"""

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Voltage KPIs
# ---------------------------------------------------------------------------


def calculate_voltage_kpis(
    results: pd.DataFrame,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict[str, Any]:
    """Calculate voltage-related KPIs from QSTS results.

    Args:
        results: QSTS results DataFrame with columns v_min, v_max, v_mean,
                 violating_buses_count, overvoltage_buses_count, undervoltage_buses_count
        v_min: Lower voltage limit (pu)
        v_max: Upper voltage limit (pu)

    Returns:
        Dictionary with voltage KPIs:
        - feeder_violation_minutes: Number of minutes with any bus violating limits
        - bus_violation_minutes: Total violation minutes across all buses
        - max_voltage: Maximum voltage observed (pu)
        - min_voltage: Minimum voltage observed (pu)
        - buses_exceeding_limit: Count of unique buses that violated limits
        - avg_voltage_deviation: Mean absolute deviation from 1.0 pu
        - voltage_severity_score: Sum of squared violation magnitudes
    """
    required_cols = ["v_min", "v_max", "violating_buses_count"]
    missing = [c for c in required_cols if c not in results.columns]
    if missing:
        raise ValueError(f"Results missing required columns: {missing}")

    # Time step in minutes (assumed 5-minute based on simulation)
    time_step_minutes = 5

    # Feeder violation minutes (timesteps with any violation)
    violating_steps = (results["violating_buses_count"] > 0).sum()
    feeder_violation_minutes = int(violating_steps * time_step_minutes)

    # Bus violation minutes (sum across all buses and timesteps)
    if "violating_buses_count" in results.columns:
        bus_violation_minutes = int(results["violating_buses_count"].sum() * time_step_minutes)
    else:
        bus_violation_minutes = feeder_violation_minutes

    # Max/min voltages
    max_voltage = float(results["v_max"].max())
    min_voltage = float(results["v_min"].min())

    # Unique buses exceeding limit (requires per-bus data or count from peak)
    buses_exceeding_limit = 0
    if "overvoltage_buses_count" in results.columns:
        buses_exceeding_limit = int(results["overvoltage_buses_count"].max())
    if "undervoltage_buses_count" in results.columns:
        buses_exceeding_limit += int(results["undervoltage_buses_count"].max())

    # Average voltage deviation from nominal
    if "v_mean" in results.columns:
        avg_voltage_deviation = float(np.abs(results["v_mean"] - 1.0).mean())
    else:
        # Approximate from min/max
        avg_voltage_deviation = float(
            (np.abs(results["v_max"] - 1.0) + np.abs(results["v_min"] - 1.0)).mean() / 2
        )

    # Voltage severity score (sum of squared violations)
    # Penalizes larger violations more heavily
    overvoltage_mask = results["v_max"] > v_max
    undervoltage_mask = results["v_min"] < v_min

    severity = 0.0
    if overvoltage_mask.any():
        severity += ((results.loc[overvoltage_mask, "v_max"] - v_max) ** 2).sum()
    if undervoltage_mask.any():
        severity += ((results.loc[undervoltage_mask, "v_min"] - v_min) ** 2).sum()

    voltage_severity_score = float(severity)

    return {
        "feeder_violation_minutes": feeder_violation_minutes,
        "bus_violation_minutes": bus_violation_minutes,
        "max_voltage": max_voltage,
        "min_voltage": min_voltage,
        "buses_exceeding_limit": buses_exceeding_limit,
        "avg_voltage_deviation": avg_voltage_deviation,
        "voltage_severity_score": voltage_severity_score,
    }


# ---------------------------------------------------------------------------
# Control KPIs
# ---------------------------------------------------------------------------


def calculate_control_kpis(results: pd.DataFrame) -> dict[str, Any]:
    """Calculate control-related KPIs from QSTS results.

    Args:
        results: QSTS results DataFrame with columns total_q_dispatch_kvar,
                 total_p_curtailment_kw, pv_generation_kw, ders_controlled

    Returns:
        Dictionary with control KPIs:
        - total_reactive_energy_kvarh: Total reactive energy absorbed (kVARh)
        - peak_reactive_dispatch_kvar: Peak reactive power dispatch (kVAR)
        - active_control_timesteps: Number of timesteps with any control action
        - total_curtailed_energy_kwh: Total energy curtailed (kWh)
        - curtailed_energy_pct: Percentage of available PV energy curtailed
        - avg_q_dispatch_kvar: Average reactive power dispatch (kVAR)
        - avg_p_curtailment_kw: Average active power curtailment (kW)
    """
    time_step_hours = 5 / 60  # 5 minutes in hours

    kpis: dict[str, Any] = {
        "total_reactive_energy_kvarh": 0.0,
        "peak_reactive_dispatch_kvar": 0.0,
        "active_control_timesteps": 0,
        "total_curtailed_energy_kwh": 0.0,
        "curtailed_energy_pct": 0.0,
        "avg_q_dispatch_kvar": 0.0,
        "avg_p_curtailment_kw": 0.0,
    }

    # Reactive power KPIs
    if "total_q_dispatch_kvar" in results.columns:
        q_dispatch = results["total_q_dispatch_kvar"]
        kpis["total_reactive_energy_kvarh"] = float(q_dispatch.sum() * time_step_hours)
        kpis["peak_reactive_dispatch_kvar"] = float(q_dispatch.max())
        kpis["avg_q_dispatch_kvar"] = float(q_dispatch.mean())

    # Active control timesteps
    control_mask = pd.Series(False, index=results.index)
    if "total_q_dispatch_kvar" in results.columns:
        control_mask |= results["total_q_dispatch_kvar"] > 0
    if "total_p_curtailment_kw" in results.columns:
        control_mask |= results["total_p_curtailment_kw"] > 0
    if "ders_controlled" in results.columns:
        control_mask |= results["ders_controlled"] > 0

    kpis["active_control_timesteps"] = int(control_mask.sum())

    # Curtailment KPIs
    if "total_p_curtailment_kw" in results.columns:
        p_curtail = results["total_p_curtailment_kw"]
        kpis["total_curtailed_energy_kwh"] = float(p_curtail.sum() * time_step_hours)
        kpis["avg_p_curtailment_kw"] = float(p_curtail.mean())

    # Curtailment percentage
    if "pv_generation_kw" in results.columns and "total_p_curtailment_kw" in results.columns:
        total_pv_energy = results["pv_generation_kw"].sum() * time_step_hours
        total_curtailed = kpis["total_curtailed_energy_kwh"]

        # Add curtailed to get available
        available_pv_energy = total_pv_energy + total_curtailed
        if available_pv_energy > 0:
            kpis["curtailed_energy_pct"] = 100.0 * total_curtailed / available_pv_energy

    return kpis


# ---------------------------------------------------------------------------
# System KPIs
# ---------------------------------------------------------------------------


def calculate_system_kpis(results: pd.DataFrame) -> dict[str, Any]:
    """Calculate system-level KPIs from QSTS results.

    Args:
        results: QSTS results DataFrame with columns losses_kw, pv_generation_kw

    Returns:
        Dictionary with system KPIs:
        - total_losses_kwh: Total energy losses (kWh)
        - avg_losses_kw: Average power losses (kW)
        - total_pv_generation_kwh: Total PV energy generated (kWh)
        - avg_pv_generation_kw: Average PV generation (kW)
        - peak_pv_generation_kw: Peak PV generation (kW)
    """
    time_step_hours = 5 / 60  # 5 minutes in hours

    kpis: dict[str, Any] = {
        "total_losses_kwh": 0.0,
        "avg_losses_kw": 0.0,
        "total_pv_generation_kwh": 0.0,
        "avg_pv_generation_kw": 0.0,
        "peak_pv_generation_kw": 0.0,
    }

    # Loss KPIs
    if "losses_kw" in results.columns:
        losses = results["losses_kw"]
        kpis["total_losses_kwh"] = float(losses.sum() * time_step_hours)
        kpis["avg_losses_kw"] = float(losses.mean())

    # PV generation KPIs
    if "pv_generation_kw" in results.columns:
        pv_gen = results["pv_generation_kw"]
        kpis["total_pv_generation_kwh"] = float(pv_gen.sum() * time_step_hours)
        kpis["avg_pv_generation_kw"] = float(pv_gen.mean())
        kpis["peak_pv_generation_kw"] = float(pv_gen.max())

    return kpis


# ---------------------------------------------------------------------------
# Combined KPI Calculation
# ---------------------------------------------------------------------------


def calculate_all_kpis(
    results: pd.DataFrame,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict[str, Any]:
    """Calculate all KPIs from QSTS results.

    Args:
        results: QSTS results DataFrame
        v_min: Lower voltage limit (pu)
        v_max: Upper voltage limit (pu)

    Returns:
        Dictionary with all KPIs from voltage, control, and system categories
    """
    kpis: dict[str, Any] = {}

    kpis.update(calculate_voltage_kpis(results, v_min, v_max))
    kpis.update(calculate_control_kpis(results))
    kpis.update(calculate_system_kpis(results))

    return kpis


# ---------------------------------------------------------------------------
# KPI Comparison
# ---------------------------------------------------------------------------


def compare_kpi_dicts(
    baseline_kpis: dict[str, Any],
    controlled_kpis: dict[str, Any],
) -> dict[str, Any]:
    """Calculate improvement ratios between baseline and controlled KPIs.

    Args:
        baseline_kpis: KPIs from baseline simulation
        controlled_kpis: KPIs from controlled simulation

    Returns:
        Dictionary with KPI comparisons:
        - For each KPI, provides the percentage improvement
        - Negative values indicate worse performance
    """
    comparison: dict[str, Any] = {}

    for key in baseline_kpis:
        if key not in controlled_kpis:
            continue

        baseline_val = baseline_kpis[key]
        controlled_val = controlled_kpis[key]

        # Skip non-numeric values
        if not isinstance(baseline_val, (int, float)) or not isinstance(controlled_val, (int, float)):
            continue

        # Calculate percentage improvement
        if baseline_val != 0:
            improvement = 100 * (controlled_val - baseline_val) / abs(baseline_val)
        else:
            improvement = 0.0 if controlled_val == 0 else float("inf")

        comparison[key] = improvement

    return comparison


def format_kpis_for_display(kpis: dict[str, Any]) -> dict[str, str]:
    """Format KPI values for human-readable display.

    Args:
        kpis: Dictionary of KPIs

    Returns:
        Dictionary with formatted string values
    """
    formatted = {}

    for key, value in kpis.items():
        if isinstance(value, (int, float)):
            # Format based on key name
            if "pct" in key or "percentage" in key:
                formatted[key] = f"{value:.2f}%"
            elif "minutes" in key:
                formatted[key] = f"{int(value)} min"
            elif "voltage" in key and "pu" not in key:
                formatted[key] = f"{value:.4f} pu"
            elif "energy" in key or "kwh" in key.lower() or "kvarh" in key.lower():
                formatted[key] = f"{value:.2f}"
            elif "kw" in key.lower() or "kvar" in key.lower():
                formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = str(value)

    return formatted


# ---------------------------------------------------------------------------
# Battery KPIs
# ---------------------------------------------------------------------------


def calculate_battery_kpis(
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate battery-related KPIs from QSTS results.

    Args:
        results: QSTS results DataFrame with columns for battery state
                 (total_battery_power_kw, battery_soc_mean, etc.)

    Returns:
        Dictionary with battery KPIs:
        - energy_throughput_kwh: Total energy charged/discharged (kWh)
        - energy_charged_kwh: Total energy charged (kWh)
        - energy_discharged_kwh: Total energy discharged (kWh)
        - soc_profile_min: Minimum SOC observed (0-1)
        - soc_profile_max: Maximum SOC observed (0-1)
        - soc_final: Final SOC at end of simulation (0-1)
        - cycles_equivalent: Equivalent full cycles (throughput / capacity)
        - round_trip_efficiency: Actual efficiency achieved
        - peak_charge_kw: Maximum charging power (kW)
        - peak_discharge_kw: Maximum discharging power (kW)
        - battery_utilization_pct: Percentage of time battery was active
    """
    time_step_hours = 5 / 60  # 5 minutes in hours

    kpis: dict[str, Any] = {
        "energy_throughput_kwh": 0.0,
        "energy_charged_kwh": 0.0,
        "energy_discharged_kwh": 0.0,
        "soc_profile_min": 0.5,
        "soc_profile_max": 0.5,
        "soc_final": 0.5,
        "cycles_equivalent": 0.0,
        "round_trip_efficiency": 0.0,
        "peak_charge_kw": 0.0,
        "peak_discharge_kw": 0.0,
        "battery_utilization_pct": 0.0,
    }

    # Battery power column (negative = charge, positive = discharge)
    battery_power_col = "total_battery_power_kw"
    battery_soc_col = "battery_soc_mean"

    if battery_power_col in results.columns:
        battery_power = results[battery_power_col]

        # Separate charging and discharging
        charging = battery_power[battery_power < 0].abs()
        discharging = battery_power[battery_power > 0]

        # Energy throughput
        kpis["energy_charged_kwh"] = float(charging.sum() * time_step_hours)
        kpis["energy_discharged_kwh"] = float(discharging.sum() * time_step_hours)
        kpis["energy_throughput_kwh"] = (
            kpis["energy_charged_kwh"] + kpis["energy_discharged_kwh"]
        )

        # Peak power
        kpis["peak_charge_kw"] = float(-battery_power.min()) if len(charging) > 0 else 0.0
        kpis["peak_discharge_kw"] = float(battery_power.max()) if len(discharging) > 0 else 0.0

        # Round-trip efficiency (discharged / charged)
        if kpis["energy_charged_kwh"] > 0:
            kpis["round_trip_efficiency"] = (
                kpis["energy_discharged_kwh"] / kpis["energy_charged_kwh"]
            )

        # Utilization (percentage of timesteps with non-zero power)
        active_timesteps = (battery_power != 0).sum()
        kpis["battery_utilization_pct"] = 100.0 * active_timesteps / len(battery_power)

    # SOC profile
    if battery_soc_col in results.columns:
        kpis["soc_profile_min"] = float(results[battery_soc_col].min())
        kpis["soc_profile_max"] = float(results[battery_soc_col].max())
        kpis["soc_final"] = float(results[battery_soc_col].iloc[-1])

    # Equivalent cycles (requires capacity from elsewhere or assume default)
    # This would typically be passed as a parameter or looked up from config
    # For now, compute cycles per kWh of throughput
    kpis["cycles_equivalent"] = kpis["energy_throughput_kwh"]  # Cycles * capacity

    return kpis


def calculate_battery_curtailment_reduction(
    baseline_results: pd.DataFrame,
    battery_results: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate curtailment reduction due to battery storage.

    Args:
        baseline_results: QSTS results without battery
        battery_results: QSTS results with battery

    Returns:
        Dictionary with curtailment reduction KPIs:
        - curtailment_reduction_kwh: Energy saved from curtailment (kWh)
        - curtailment_reduction_pct: Percentage reduction in curtailment
        - pv_energy_utilized_kwh: Additional PV energy utilized (kWh)
    """
    time_step_hours = 5 / 60

    kpis: dict[str, Any] = {
        "curtailment_reduction_kwh": 0.0,
        "curtailment_reduction_pct": 0.0,
        "pv_energy_utilized_kwh": 0.0,
    }

    # Get curtailment from both scenarios
    baseline_p_curtail = baseline_results.get("total_p_curtailment_kw", pd.Series(0))
    battery_p_curtail = battery_results.get("total_p_curtailment_kw", pd.Series(0))

    baseline_curtailment_kwh = baseline_p_curtail.sum() * time_step_hours
    battery_curtailment_kwh = battery_p_curtail.sum() * time_step_hours

    kpis["curtailment_reduction_kwh"] = max(0, baseline_curtailment_kwh - battery_curtailment_kwh)

    if baseline_curtailment_kwh > 0:
        kpis["curtailment_reduction_pct"] = 100.0 * (
            kpis["curtailment_reduction_kwh"] / baseline_curtailment_kwh
        )

    # Additional PV utilized = reduced curtailment
    kpis["pv_energy_utilized_kwh"] = kpis["curtailment_reduction_kwh"]

    return kpis
