"""Interactive Plotly dashboard for DERMS QSTS results.

Creates HTML dashboard with interactive plots for voltage analysis,
control dispatch, and KPI visualization.
"""

import json
import pathlib
import base64
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder


def plot_voltage_envelope_plotly(
    results: pd.DataFrame,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> go.Figure:
    """Create an interactive voltage envelope plot.

    Args:
        results: QSTS results DataFrame with time_h, v_min, v_max, v_mean
        v_min: Lower voltage limit for shading
        v_max: Upper voltage limit for shading

    Returns:
        Plotly Figure object with interactive voltage envelope
    """
    fig = go.Figure()

    # Add limit bands
    fig.add_hrect(
        y0=v_max, y1=1.2,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Overvoltage",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=0.8, y1=v_min,
        fillcolor="orange", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Undervoltage",
        annotation_position="bottom left",
    )

    # Add voltage traces
    fig.add_trace(go.Scatter(
        x=results["time_h"],
        y=results["v_max"],
        mode="lines",
        name="Max Voltage",
        line=dict(color="#e74c3c", width=1),
        hovertemplate="<b>%{x:.2f} h</b><br>Max: %{y:.4f} pu<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=results["time_h"],
        y=results["v_mean"],
        mode="lines",
        name="Mean Voltage",
        line=dict(color="#3498db", width=2),
        hovertemplate="<b>%{x:.2f} h</b><br>Mean: %{y:.4f} pu<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=results["time_h"],
        y=results["v_min"],
        mode="lines",
        name="Min Voltage",
        line=dict(color="#f39c12", width=1),
        fill="tonexty",
        fillcolor="rgba(52, 152, 219, 0.1)",
        hovertemplate="<b>%{x:.2f} h</b><br>Min: %{y:.4f} pu<extra></extra>",
    ))

    # Add limit lines
    fig.add_hline(
        y=v_max, line_dash="dash", line_color="red",
        annotation_text=f"V_max = {v_max} pu"
    )
    fig.add_hline(
        y=v_min, line_dash="dash", line_color="orange",
        annotation_text=f"V_min = {v_min} pu"
    )

    fig.update_layout(
        title="Voltage Envelope - 24 Hour QSTS",
        xaxis_title="Time (hours)",
        yaxis_title="Voltage (pu)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    return fig


def plot_voltage_heatmap(
    results: pd.DataFrame,
    bus_subset: list[str] | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> go.Figure:
    """Create an interactive voltage heatmap (bus vs time).

    Args:
        results: QSTS results DataFrame with per-bus voltage columns
        bus_subset: Optional list of buses to include (all if None)
        v_min: Lower voltage limit for color scaling
        v_max: Upper voltage limit for color scaling

    Returns:
        Plotly Figure object with interactive heatmap
    """
    def _build_extrema_fallback_matrix() -> tuple[list[str], np.ndarray] | None:
        """Build sparse bus-vs-time matrix from v_min_bus/v_max_bus columns."""
        required_cols = {"time_h", "v_min_bus", "v_max_bus", "v_min", "v_max"}
        if not required_cols.issubset(results.columns):
            return None

        extrema_buses = pd.concat([
            results["v_min_bus"].astype(str),
            results["v_max_bus"].astype(str),
        ]).dropna()
        bus_names = sorted({b.strip() for b in extrema_buses if b and b.lower() != "nan"})
        if not bus_names:
            return None

        if bus_subset:
            bus_names = [b for b in bus_names if b in bus_subset]
            if not bus_names:
                return None

        bus_index = {bus: idx for idx, bus in enumerate(bus_names)}
        voltage_matrix = np.full((len(bus_names), len(results)), np.nan, dtype=float)

        for t_idx, row in results.reset_index(drop=True).iterrows():
            min_bus = str(row.get("v_min_bus", "")).strip()
            max_bus = str(row.get("v_max_bus", "")).strip()
            v_min_row = row.get("v_min", np.nan)
            v_max_row = row.get("v_max", np.nan)

            if min_bus in bus_index and pd.notna(v_min_row):
                voltage_matrix[bus_index[min_bus], t_idx] = float(v_min_row)
            if max_bus in bus_index and pd.notna(v_max_row):
                existing = voltage_matrix[bus_index[max_bus], t_idx]
                voltage_matrix[bus_index[max_bus], t_idx] = (
                    float(v_max_row) if np.isnan(existing) else max(existing, float(v_max_row))
                )

        return bus_names, voltage_matrix

    # Extract per-bus voltage columns (columns starting with "overv_")
    voltage_cols = [c for c in results.columns if c.startswith("overv_")]
    bus_names: list[str] = []
    voltage_matrix: np.ndarray | None = None
    title = "Voltage Heatmap - Bus vs Time"
    subtitle_note: str | None = None

    if voltage_cols:
        bus_names = [c.replace("overv_", "") for c in voltage_cols]
        if bus_subset:
            keep_cols = [c for c, b in zip(voltage_cols, bus_names) if b in bus_subset]
            bus_names = [b for b in bus_names if b in bus_subset]
            voltage_cols = keep_cols

        if voltage_cols:
            voltage_matrix = results[voltage_cols].T.values

        # If logging is too sparse (or all-NaN), use extrema fallback.
        use_fallback = False
        if voltage_matrix is None or voltage_matrix.size == 0:
            use_fallback = True
        else:
            finite_mask = np.isfinite(voltage_matrix)
            finite_ratio = float(finite_mask.sum()) / float(voltage_matrix.size)
            if finite_ratio < 0.08:
                use_fallback = True

        if use_fallback:
            fallback = _build_extrema_fallback_matrix()
            if fallback is not None:
                bus_names, voltage_matrix = fallback
                title = "Voltage Heatmap - Critical Buses (Fallback)"
                subtitle_note = (
                    "Derived from v_min_bus/v_max_bus because full per-bus traces are unavailable."
                )

    else:
        fallback = _build_extrema_fallback_matrix()
        if fallback is not None:
            bus_names, voltage_matrix = fallback
            title = "Voltage Heatmap - Critical Buses (Fallback)"
            subtitle_note = (
                "Derived from v_min_bus/v_max_bus because full per-bus traces are unavailable."
            )

    if voltage_matrix is None or len(bus_names) == 0:
        # No per-bus data and no extrema fallback available.
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "No per-bus voltage data available.<br>"
                "Enable detailed voltage logging to show full bus heatmaps."
            ),
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Voltage Heatmap - No Data",
            xaxis_title="Time (hours)",
            yaxis_title="Bus",
            height=300,
        )
        return fig

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=voltage_matrix,
        x=results["time_h"],
        y=bus_names,
        colorscale=[
            [0.0, "#3498db"],     # Blue for undervoltage
            [0.5, "#2ecc71"],     # Green for nominal
            [0.95, "#f39c12"],    # Orange for high
            [1.0, "#e74c3c"],     # Red for overvoltage
        ],
        zmin=v_min - 0.05,
        zmax=v_max + 0.02,
        colorbar=dict(title="Voltage (pu)"),
        hovertemplate="<b>Bus %{y}</b><br>Time: %{x:.2f} h<br>Voltage: %{z:.4f} pu<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (hours)",
        yaxis_title="Bus",
        height=max(300, len(bus_names) * 20),
        template="plotly_white",
        margin=dict(l=80, r=50, t=50, b=50),
    )

    if subtitle_note:
        fig.add_annotation(
            text=subtitle_note,
            xref="paper",
            yref="paper",
            x=0,
            y=1.14,
            showarrow=False,
            font=dict(size=11, color="gray"),
            align="left",
        )

    return fig


def plot_q_dispatch_plotly(
    results: pd.DataFrame,
) -> go.Figure:
    """Create an interactive reactive power dispatch plot.

    Args:
        results: QSTS results DataFrame with time_h, total_q_dispatch_kvar

    Returns:
        Plotly Figure object with interactive Q dispatch timeline
    """
    if "total_q_dispatch_kvar" not in results.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No reactive power dispatch data available.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(title="Reactive Power Dispatch - No Data")
        return fig

    fig = go.Figure()

    # Add Q dispatch trace
    fig.add_trace(go.Scatter(
        x=results["time_h"],
        y=results["total_q_dispatch_kvar"],
        mode="lines",
        name="Q Dispatch",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy",
        fillcolor="rgba(155, 89, 182, 0.2)",
        hovertemplate="<b>%{x:.2f} h</b><br>Q: %{y:.1f} kVAR<extra></extra>",
    ))

    # Add zero line
    fig.add_hline(y=0, line_color="gray", line_width=1)

    fig.update_layout(
        title="Reactive Power Dispatch Timeline",
        xaxis_title="Time (hours)",
        yaxis_title="Q Absorption (kVAR)",
        template="plotly_white",
        height=300,
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="x unified",
    )

    return fig


def plot_p_curtailment_plotly(
    results: pd.DataFrame,
) -> go.Figure:
    """Create an interactive active power curtailment plot.

    Args:
        results: QSTS results DataFrame with time_h, total_p_curtailment_kw, pv_generation_kw

    Returns:
        Plotly Figure object with interactive P curtailment timeline
    """
    fig = go.Figure()

    has_curtail = "total_p_curtailment_kw" in results.columns
    has_pv = "pv_generation_kw" in results.columns

    if has_pv:
        # Add PV generation trace
        fig.add_trace(go.Scatter(
            x=results["time_h"],
            y=results["pv_generation_kw"],
            mode="lines",
            name="PV Generation",
            line=dict(color="#f39c12", width=2),
            hovertemplate="<b>%{x:.2f} h</b><br>PV: %{y:.1f} kW<extra></extra>",
        ))

    if has_curtail:
        # Add curtailment trace
        fig.add_trace(go.Scatter(
            x=results["time_h"],
            y=results["total_p_curtailment_kw"],
            mode="lines",
            name="P Curtailment",
            line=dict(color="#e74c3c", width=2),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.3)",
            hovertemplate="<b>%{x:.2f} h</b><br>Curtail: %{y:.1f} kW<extra></extra>",
        ))

    if not has_curtail and not has_pv:
        fig.add_annotation(
            text="No power curtailment data available.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )

    fig.update_layout(
        title="Active Power Curtailment Timeline",
        xaxis_title="Time (hours)",
        yaxis_title="Power (kW)",
        template="plotly_white",
        height=300,
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    return fig


def plot_battery_power_plotly(
    results: pd.DataFrame,
) -> go.Figure:
    """Create an interactive battery power plot.

    Args:
        results: QSTS results DataFrame with battery metrics

    Returns:
        Plotly Figure object with battery power timeline
    """
    if "total_battery_power_kw" not in results.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No battery data available.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(title="Battery Power - No Data")
        return fig

    fig = go.Figure()

    # Add battery power trace
    # Positive = discharge, Negative = charge
    colors = ["#e74c3c" if p >= 0 else "#3498db"
              for p in results["total_battery_power_kw"]]

    fig.add_trace(go.Scatter(
        x=results["time_h"],
        y=results["total_battery_power_kw"],
        mode="lines",
        name="Battery Power",
        line=dict(color="#9b59b6", width=2),
        hovertemplate="<b>%{x:.2f} h</b><br>Power: %{y:.1f} kW<br>%{text}",
        text=["Discharging" if p >= 0 else "Charging"
              for p in results["total_battery_power_kw"]],
    ))

    # Add zero line
    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dash")

    # Add shaded regions
    fig.add_hrect(
        y0=0, y1=max(results["total_battery_power_kw"].max(), 1) * 1.1,
        fillcolor="#e74c3c", opacity=0.05,
        layer="below", line_width=0,
        annotation_text="Discharging",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=min(results["total_battery_power_kw"].min(), -1) * 1.1, y1=0,
        fillcolor="#3498db", opacity=0.05,
        layer="below", line_width=0,
        annotation_text="Charging",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title="Battery Power Timeline",
        xaxis_title="Time (hours)",
        yaxis_title="Power (kW) - (+) Discharge / (-) Charge",
        template="plotly_white",
        height=300,
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="x unified",
    )

    return fig


def plot_battery_soc_plotly(
    results: pd.DataFrame,
) -> go.Figure:
    """Create an interactive battery SOC plot.

    Args:
        results: QSTS results DataFrame with battery_soc_mean

    Returns:
        Plotly Figure object with battery SOC timeline
    """
    if "battery_soc_mean" not in results.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No battery SOC data available.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(title="Battery SOC - No Data")
        return fig

    fig = go.Figure()

    # Add SOC trace
    fig.add_trace(go.Scatter(
        x=results["time_h"],
        y=results["battery_soc_mean"],
        mode="lines",
        name="SOC",
        line=dict(color="#2ecc71", width=2),
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.2)",
        hovertemplate="<b>%{x:.2f} h</b><br>SOC: %{y:.1%}<extra></extra>",
    ))

    # Add limit lines
    fig.add_hline(
        y=0.9, line_dash="dash", line_color="green",
        annotation_text="SOC_max = 90%"
    )
    fig.add_hline(
        y=0.1, line_dash="dash", line_color="orange",
        annotation_text="SOC_min = 10%"
    )

    fig.update_layout(
        title="Battery State of Charge",
        xaxis_title="Time (hours)",
        yaxis_title="SOC (0-1)",
        template="plotly_white",
        height=300,
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="x unified",
    )

    return fig


def create_kpi_cards(
    baseline_kpis: dict[str, Any] | None = None,
    controlled_kpis: dict[str, Any] | None = None,
    battery_kpis: dict[str, Any] | None = None,
) -> str:
    """Create HTML KPI summary cards.

    Args:
        baseline_kpis: KPIs from baseline simulation
        controlled_kpis: KPIs from controlled simulation
        battery_kpis: Battery-specific KPIs

    Returns:
        HTML string with KPI cards
    """
    cards = []

    # Voltage KPIs
    if controlled_kpis:
        v_max = controlled_kpis.get("max_voltage", 0)
        v_min = controlled_kpis.get("min_voltage", 0)
        violation_min = controlled_kpis.get("feeder_violation_minutes", 0)

        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Voltage Range</div>
            <div class="kpi-value">{v_max:.3f} / {v_min:.3f} pu</div>
            <div class="kpi-subtitle">Max / Min</div>
        </div>
        """)

        # Build violation card content outside the f-string
        violation_value = "" if violation_min == 0 else f"{violation_min} min"
        violation_subtitle = "No violations" if violation_min == 0 else "Issues detected"

        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Violation Time</div>
            <div class="kpi-value">{violation_value}</div>
            <div class="kpi-subtitle">{violation_subtitle}</div>
        </div>
        """)

    # Control KPIs
    if controlled_kpis:
        curtailed = controlled_kpis.get("total_curtailed_energy_kwh", 0)
        reactive = controlled_kpis.get("total_reactive_energy_kvarh", 0)

        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Energy Curtailed</div>
            <div class="kpi-value">{curtailed:.1f} kWh</div>
            <div class="kpi-subtitle">Lost PV energy</div>
        </div>
        """)

        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Reactive Support</div>
            <div class="kpi-value">{reactive:.0f} kVARh</div>
            <div class="kpi-subtitle">Total Q absorption</div>
        </div>
        """)

    # Battery KPIs
    if battery_kpis:
        throughput = battery_kpis.get("energy_throughput_kwh", 0)
        cycles = battery_kpis.get("cycles_equivalent", 0)
        soc_final = battery_kpis.get("soc_final", 0.5)

        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Battery Throughput</div>
            <div class="kpi-value">{throughput:.1f} kWh</div>
            <div class="kpi-subtitle">Total energy cycled</div>
        </div>
        """)

        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Equivalent Cycles</div>
            <div class="kpi-value">{cycles:.2f}</div>
            <div class="kpi-subtitle">Throughput / Capacity</div>
        </div>
        """)

        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Final SOC</div>
            <div class="kpi-value">{soc_final:.0%}</div>
            <div class="kpi-subtitle">End of simulation</div>
        </div>
        """)

    return "\n".join(cards)


def _load_results_bundle(results_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Load all available mode results from a results directory."""
    results_bundle: dict[str, pd.DataFrame] = {}

    for csv_file in results_dir.glob("*.csv"):
        file_name = csv_file.name.lower()
        if "baseline" in file_name:
            results_bundle["baseline"] = pd.read_csv(csv_file)
        elif "heuristic" in file_name:
            results_bundle["heuristic"] = pd.read_csv(csv_file)
        elif "optimization" in file_name:
            results_bundle["optimization"] = pd.read_csv(csv_file)
        elif "battery" in file_name:
            results_bundle["battery"] = pd.read_csv(csv_file)

    if results_bundle:
        return results_bundle

    for root in [results_dir, results_dir.parent]:
        mode_paths = {
            "baseline": root / "baseline" / "qsts_baseline.csv",
            "heuristic": root / "heuristic" / "qsts_baseline.csv",
            "optimization": root / "optimization" / "qsts_baseline.csv",
            "battery": root / "battery" / "qsts_baseline.csv",
        }
        for mode, path in mode_paths.items():
            if mode not in results_bundle and path.exists():
                results_bundle[mode] = pd.read_csv(path)

    return results_bundle


def _plotly_to_plain_json(value: Any) -> Any:
    """Convert Plotly JSON payloads into browser-friendly plain JSON values."""
    if isinstance(value, dict):
        if {"dtype", "bdata"}.issubset(value.keys()):
            raw = base64.b64decode(value["bdata"])
            array = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
            shape = value.get("shape")
            if shape is not None:
                try:
                    if isinstance(shape, str):
                        normalized = shape.strip().replace("(", "").replace(")", "")
                        dims = tuple(int(part.strip()) for part in normalized.split(",") if part.strip())
                    elif isinstance(shape, (list, tuple)):
                        dims = tuple(int(part) for part in shape)
                    else:
                        dims = ()
                    if dims:
                        array = array.reshape(dims)
                except Exception:
                    # Fall back to flat list if shape metadata is malformed.
                    pass
            return array.tolist()
        return {key: _plotly_to_plain_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plotly_to_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_hosting_capacity_summary(results_dir: pathlib.Path) -> dict[str, Any] | None:
    """Load the hosting-capacity summary when present."""
    candidate_paths = [
        results_dir / "comparison" / "hosting_capacity" / "hosting_capacity_summary.json",
        results_dir / "hosting_capacity" / "hosting_capacity_summary.json",
    ]

    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text())

    return None


def _get_hosting_capacity_root(results_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the hosting-capacity results root when present."""
    candidate_paths = [
        results_dir,
        results_dir / "hosting_capacity",
    ]
    for path in candidate_paths:
        if (path / "baseline" / "hosting_capacity" / "sweep_summary.csv").exists():
            return path
        if (path / "baseline" / "sweep_summary.csv").exists():
            return path
    return None


def _load_stress_case_bundle(
    results_dir: pathlib.Path,
) -> tuple[dict[str, pd.DataFrame], float | None]:
    """Load the highest common hosting-capacity scale across modes."""
    hosting_root = _get_hosting_capacity_root(results_dir)
    if hosting_root is None:
        return {}, None

    summaries: dict[str, pd.DataFrame] = {}
    common_scales: set[float] | None = None
    for mode in ["baseline", "heuristic", "optimization"]:
        summary_path = hosting_root / mode / "hosting_capacity" / "sweep_summary.csv"
        if not summary_path.exists():
            summary_path = hosting_root / mode / "sweep_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        summaries[mode] = df
        scales = {float(scale) for scale in df["pv_scale"].tolist()}
        common_scales = scales if common_scales is None else common_scales & scales

    if not common_scales:
        return {}, None

    selected_scale = max(common_scales)
    bundle: dict[str, pd.DataFrame] = {}
    for mode in ["baseline", "heuristic", "optimization"]:
        csv_path = (
            hosting_root
            / mode
            / "hosting_capacity"
            / f"pv_scale_{selected_scale:.2f}"
            / "qsts_baseline.csv"
        )
        if not csv_path.exists():
            csv_path = hosting_root / mode / f"pv_scale_{selected_scale:.2f}" / "qsts_baseline.csv"
        if csv_path.exists():
            bundle[mode] = pd.read_csv(csv_path)

    return bundle, selected_scale


def _format_mode_label(mode: str) -> str:
    """Format a mode key for user-facing display."""
    return mode.replace("_", " ").title()


def _build_mode_summary_table(mode_kpis: dict[str, dict[str, Any]]) -> str:
    """Build a compact KPI comparison table for all modes."""
    metric_rows = [
        ("feeder_violation_minutes", "Violation Minutes", "{:.0f}"),
        ("max_voltage", "Max Voltage (pu)", "{:.4f}"),
        ("min_voltage", "Min Voltage (pu)", "{:.4f}"),
        ("total_reactive_energy_kvarh", "Reactive Energy (kVARh)", "{:.1f}"),
        ("total_curtailed_energy_kwh", "Curtailment (kWh)", "{:.1f}"),
        ("total_losses_kwh", "Losses (kWh)", "{:.1f}"),
    ]

    modes = list(mode_kpis.keys())
    rows = [
        "<div class='summary-table-wrap'>",
        "<table class='summary-table'>",
        "<thead><tr><th>Metric</th>"
        + "".join(f"<th>{_format_mode_label(mode)}</th>" for mode in modes)
        + "</tr></thead>",
        "<tbody>",
    ]

    for key, label, fmt in metric_rows:
        row = [f"<tr><td>{label}</td>"]
        for mode in modes:
            row.append(f"<td>{fmt.format(mode_kpis[mode].get(key, 0.0))}</td>")
        row.append("</tr>")
        rows.append("".join(row))

    rows.extend(["</tbody>", "</table>", "</div>"])
    return "\n".join(rows)


def _build_hosting_capacity_cards(hosting_summary: dict[str, Any] | None) -> str:
    """Build overview KPI cards from hosting-capacity outputs."""
    if not hosting_summary:
        return """
        <div class="kpi-card">
            <div class="kpi-title">Hosting Capacity</div>
            <div class="kpi-value">N/A</div>
            <div class="kpi-subtitle">Run hosting-capacity study</div>
        </div>
        """

    cards: list[str] = []
    for mode in ["baseline", "heuristic", "optimization"]:
        if mode not in hosting_summary:
            continue
        capacity = hosting_summary[mode].get("hosting_capacity", 0.0)
        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">{_format_mode_label(mode)} Hosting</div>
            <div class="kpi-value">{capacity:.2f}x</div>
            <div class="kpi-subtitle">Max safe PV scale</div>
        </div>
        """)

    optimization_gain = hosting_summary.get("improvements", {}).get("optimization_improvement_ratio")
    if optimization_gain is not None:
        cards.append(f"""
        <div class="kpi-card">
            <div class="kpi-title">Optimization Gain</div>
            <div class="kpi-value">{optimization_gain:.2f}x</div>
            <div class="kpi-subtitle">Vs baseline hosting capacity</div>
        </div>
        """)

    return "\n".join(cards)


def plot_voltage_comparison_plotly(
    results_by_mode: dict[str, pd.DataFrame],
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> go.Figure:
    """Create an interactive multi-mode voltage comparison chart."""
    fig = go.Figure()
    colors = {
        "baseline": "#34495e",
        "heuristic": "#f39c12",
        "optimization": "#27ae60",
        "battery": "#8e44ad",
    }

    for mode, results in results_by_mode.items():
        color = colors.get(mode, "#3498db")
        label = _format_mode_label(mode)
        fig.add_trace(go.Scatter(
            x=results["time_h"],
            y=results["v_max"],
            mode="lines",
            name=f"{label} Max",
            line=dict(color=color, width=2),
            hovertemplate="<b>%{x:.2f} h</b><br>Vmax: %{y:.4f} pu<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=results["time_h"],
            y=results["v_min"],
            mode="lines",
            name=f"{label} Min",
            line=dict(color=color, width=1, dash="dot"),
            hovertemplate="<b>%{x:.2f} h</b><br>Vmin: %{y:.4f} pu<extra></extra>",
        ))

    fig.add_hline(y=v_max, line_dash="dash", line_color="red", annotation_text=f"Vmax = {v_max:.2f}")
    fig.add_hline(y=v_min, line_dash="dash", line_color="orange", annotation_text=f"Vmin = {v_min:.2f}")
    fig.update_layout(
        xaxis_title="Time (hours)",
        yaxis_title="Voltage (pu)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=30, t=88, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12),
        ),
    )
    return fig


def plot_hosting_capacity_plotly(hosting_summary: dict[str, Any] | None) -> go.Figure:
    """Create a hosting-capacity comparison bar chart."""
    fig = go.Figure()

    if not hosting_summary:
        fig.add_annotation(
            text="No hosting-capacity summary available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(
            title="Hosting Capacity Comparison",
            template="plotly_white",
            height=320,
        )
        return fig

    modes = [mode for mode in ["baseline", "heuristic", "optimization"] if mode in hosting_summary]
    capacities = [hosting_summary[mode].get("hosting_capacity", 0.0) for mode in modes]
    colors = ["#34495e", "#f39c12", "#27ae60"][:len(modes)]

    fig.add_trace(go.Bar(
        x=[_format_mode_label(mode) for mode in modes],
        y=capacities,
        marker_color=colors,
        text=[f"{capacity:.2f}x" for capacity in capacities],
        textposition="outside",
        textfont=dict(color="#f2f4fb", size=13, family="'Space Grotesk', sans-serif"),
        cliponaxis=False,
        hovertemplate="%{x}<br>Hosting capacity: %{y:.2f}x<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Mode",
        yaxis_title="Max safe PV scale",
        template="plotly_white",
        height=320,
        margin=dict(l=60, r=30, t=44, b=50),
        yaxis=dict(range=[0, max(capacities + [1.0]) * 1.32]),
    )
    return fig


def create_dashboard(
    results_dir: pathlib.Path,
    output_path: pathlib.Path | None = None,
) -> str:
    """Create an interactive Plotly dashboard HTML file.

    Loads CSV results from baseline, heuristic, optimization runs.
    Creates interactive plots for voltage, control, and KPIs.

    Args:
        results_dir: Directory containing result CSV files
        output_path: Optional output HTML path (defaults to results_dir/dashboard.html)

    Returns:
        Path to generated HTML file
    """
    results_dir = pathlib.Path(results_dir)

    nominal_results_bundle = _load_results_bundle(results_dir)
    stress_results_bundle, stress_scale = _load_stress_case_bundle(results_dir)
    results_bundle = dict(nominal_results_bundle)
    if stress_results_bundle:
        # Prefer stressed hosting-capacity runs for the main control modes,
        # but keep any directly packaged extras such as battery results.
        results_bundle.update(stress_results_bundle)

    if not results_bundle:
        raise ValueError(f"No result CSV files found in {results_dir}")

    available_modes = [
        mode for mode in ["baseline", "heuristic", "optimization", "battery"]
        if mode in results_bundle
    ]
    default_mode = "optimization" if "optimization" in results_bundle else available_modes[0]

    # Get voltage limits from results or use defaults
    v_min = 0.95
    v_max = 1.05

    # Calculate KPIs
    from src.analysis.kpis import calculate_all_kpis, calculate_battery_kpis

    mode_kpis = {
        mode: calculate_all_kpis(results_bundle[mode], v_min=v_min, v_max=v_max)
        for mode in available_modes
    }
    battery_kpis = (
        calculate_battery_kpis(results_bundle["battery"])
        if "battery" in results_bundle and not results_bundle["battery"].empty
        else None
    )
    hosting_summary = _load_hosting_capacity_summary(results_dir)
    comparison_modes = {
        mode: results_bundle[mode]
        for mode in ["baseline", "heuristic", "optimization"]
        if mode in results_bundle
    }
    comparison_voltage_plot = plot_voltage_comparison_plotly(
        comparison_modes or {default_mode: results_bundle[default_mode]},
        v_min=v_min,
        v_max=v_max,
    )
    hosting_capacity_plot = plot_hosting_capacity_plotly(hosting_summary)

    baseline_kpi = mode_kpis.get("baseline", {})
    heuristic_kpi = mode_kpis.get("heuristic", {})
    optimization_kpi = mode_kpis.get("optimization", mode_kpis.get(default_mode, {}))
    battery_mode_kpi = mode_kpis.get("battery", {})
    baseline_violation = float(baseline_kpi.get("feeder_violation_minutes", 0.0) or 0.0)
    optimization_violation = float(optimization_kpi.get("feeder_violation_minutes", 0.0) or 0.0)
    if baseline_violation > 0:
        violation_reduction = max(
            0.0, min(100.0, (baseline_violation - optimization_violation) / baseline_violation * 100.0)
        )
    else:
        violation_reduction = 100.0 if optimization_violation <= 0 else 0.0

    optimization_hosting_gain = (
        hosting_summary.get("improvements", {}).get("optimization_improvement_ratio")
        if hosting_summary
        else None
    )
    headline_hosting_gain = (
        f"{optimization_hosting_gain:.2f}x"
        if optimization_hosting_gain is not None
        else "N/A"
    )
    headline_scale = f"{stress_scale:.1f}x PV" if stress_scale is not None else "Nominal Day"
    hero_copy = (
        f"At {headline_scale}, baseline recorded {baseline_violation:.0f} violation minutes. "
        f"Optimization finished at {optimization_violation:.0f} minutes with "
        f"{optimization_kpi.get('total_curtailed_energy_kwh', 0.0):.1f} kWh curtailed."
    )
    if "battery" in available_modes and battery_kpis is not None:
        hero_copy += (
            f" The battery run cycled {battery_kpis.get('energy_throughput_kwh', 0.0):.1f} kWh "
            f"and ended at {battery_kpis.get('soc_final', 0.5):.0%} SOC."
        )

    results_snapshot_cards: list[str] = []
    for mode in ["baseline", "heuristic", "optimization"]:
        if mode not in mode_kpis:
            continue
        kpis = mode_kpis[mode]
        results_snapshot_cards.append(f"""
        <article class="result-chip">
            <div class="result-mode">{_format_mode_label(mode)}</div>
            <div class="result-line"><span>Violation time</span><strong>{kpis.get("feeder_violation_minutes", 0.0):.0f} min</strong></div>
            <div class="result-line"><span>Max voltage</span><strong>{kpis.get("max_voltage", 0.0):.4f} pu</strong></div>
            <div class="result-line"><span>Curtailment</span><strong>{kpis.get("total_curtailed_energy_kwh", 0.0):.1f} kWh</strong></div>
        </article>
        """)
    if "battery" in available_modes and battery_kpis is not None:
        results_snapshot_cards.append(f"""
        <article class="result-chip">
            <div class="result-mode">Battery</div>
            <div class="result-line"><span>Violation time</span><strong>{battery_mode_kpi.get("feeder_violation_minutes", 0.0):.0f} min</strong></div>
            <div class="result-line"><span>Throughput</span><strong>{battery_kpis.get("energy_throughput_kwh", 0.0):.1f} kWh</strong></div>
            <div class="result-line"><span>Final SOC</span><strong>{battery_kpis.get("soc_final", 0.5):.0%}</strong></div>
        </article>
        """)
    results_snapshot_html = "\n".join(results_snapshot_cards)

    def _apply_dark_plot_theme(fig: go.Figure, height: int) -> go.Figure:
        """Apply a dark, minimal visual theme to Plotly figures."""
        fig.update_layout(
            template=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#15161a",
            font=dict(color="#d9dbe3", family="'Manrope', sans-serif"),
            margin=dict(l=54, r=24, t=48, b=44),
            height=height,
            hoverlabel=dict(
                bgcolor="rgba(12, 13, 18, 0.96)",
                bordercolor="rgba(255, 255, 255, 0.20)",
                font=dict(color="#eef1fa", size=12, family="'Manrope', sans-serif"),
            ),
            legend=dict(
                bgcolor="rgba(12, 13, 18, 0.78)",
                bordercolor="rgba(255, 255, 255, 0.22)",
                borderwidth=1,
                font=dict(color="#f0f2f8", size=12, family="'Manrope', sans-serif"),
                itemsizing="constant",
                itemwidth=42,
                tracegroupgap=8,
            ),
        )
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.20)",
            tickfont=dict(color="#c6cad7", size=11),
            title_font=dict(color="#b9bdc9"),
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.20)",
            tickfont=dict(color="#c6cad7", size=11),
            title_font=dict(color="#b9bdc9"),
        )
        for annotation in fig.layout.annotations or []:
            annotation.font = dict(color="#eceff7", size=11, family="'Manrope', sans-serif")
            annotation.bgcolor = "rgba(12, 13, 18, 0.82)"
            annotation.bordercolor = "rgba(255, 255, 255, 0.18)"
            annotation.borderwidth = 1
        for trace in fig.data:
            if getattr(trace, "type", "") == "heatmap" and getattr(trace, "colorbar", None) is not None:
                trace.colorbar.title = dict(text="Voltage (pu)", font=dict(color="#eceff7", size=12))
                trace.colorbar.tickfont = dict(color="#dde1ec", size=11)
                trace.colorbar.outlinecolor = "rgba(255,255,255,0.25)"
                trace.colorbar.bgcolor = "rgba(12,13,18,0.78)"
        return fig

    comparison_voltage_plot = _apply_dark_plot_theme(comparison_voltage_plot, 340)
    hosting_capacity_plot = _apply_dark_plot_theme(hosting_capacity_plot, 320)
    hosting_capacity_plot.update_layout(showlegend=False)

    mode_cards: dict[str, str] = {}
    mode_figures: dict[str, dict[str, Any]] = {}
    for mode in available_modes:
        results = results_bundle[mode]
        mode_cards[mode] = create_kpi_cards(
            controlled_kpis=mode_kpis[mode],
            battery_kpis=battery_kpis if mode == "battery" else None,
        )

        voltage_plot = _apply_dark_plot_theme(
            plot_voltage_envelope_plotly(results, v_min=v_min, v_max=v_max),
            320,
        )
        heatmap_plot = _apply_dark_plot_theme(
            plot_voltage_heatmap(results, v_min=v_min, v_max=v_max),
            max(300, min(500, 260 + len([c for c in results.columns if c.startswith("overv_")]) * 12)),
        )
        q_plot = _apply_dark_plot_theme(plot_q_dispatch_plotly(results), 280)
        p_plot = _apply_dark_plot_theme(plot_p_curtailment_plotly(results), 280)
        battery_power_plot = _apply_dark_plot_theme(plot_battery_power_plotly(results), 280)
        battery_soc_plot = _apply_dark_plot_theme(plot_battery_soc_plotly(results), 280)

        mode_figures[mode] = {
            "voltage": _plotly_to_plain_json(voltage_plot.to_plotly_json()),
            "heatmap": _plotly_to_plain_json(heatmap_plot.to_plotly_json()),
            "q": _plotly_to_plain_json(q_plot.to_plotly_json()),
            "p": _plotly_to_plain_json(p_plot.to_plotly_json()),
            "battery_power": _plotly_to_plain_json(battery_power_plot.to_plotly_json()),
            "battery_soc": _plotly_to_plain_json(battery_soc_plot.to_plotly_json()),
        }

    overview_figures = {
        "comparison_voltage": _plotly_to_plain_json(comparison_voltage_plot.to_plotly_json()),
        "hosting_capacity": _plotly_to_plain_json(hosting_capacity_plot.to_plotly_json()),
    }

    mode_cards_json = json.dumps(mode_cards)
    mode_figures_json = json.dumps(mode_figures, cls=PlotlyJSONEncoder)
    overview_figures_json = json.dumps(overview_figures, cls=PlotlyJSONEncoder)
    has_battery_mode = "battery" in available_modes

    # Create HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>DERMS Results Dashboard</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">',
        """
        <style>
            :root {
                --bg: #0d0e11;
                --card: #14151a;
                --card-soft: #171920;
                --line: rgba(255, 255, 255, 0.08);
                --text: #e9ebf2;
                --text-soft: #9fa5b5;
                --accent: #ff7a3d;
                --accent-soft: #ffb088;
                --radius: 22px;
                --shadow: 0 18px 36px rgba(0, 0, 0, 0.32);
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                color: var(--text);
                font-family: "Manrope", sans-serif;
                background:
                    radial-gradient(1200px 600px at 16% -20%, rgba(255, 122, 61, 0.25), transparent 70%),
                    radial-gradient(1000px 500px at 95% -10%, rgba(255, 168, 114, 0.15), transparent 70%),
                    linear-gradient(180deg, #0b0c10 0%, #0f1014 100%);
            }

            .page {
                width: min(1480px, 94vw);
                margin: 20px auto 36px;
            }

            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(12, minmax(0, 1fr));
                gap: 14px;
            }

            .panel {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: linear-gradient(160deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.00)), var(--card);
                box-shadow: var(--shadow);
                overflow: hidden;
                position: relative;
                animation: rise 500ms ease both;
            }

            .panel::after {
                content: "";
                position: absolute;
                inset: 0;
                pointer-events: none;
                background: linear-gradient(135deg, rgba(255, 122, 61, 0.08), transparent 40%);
                opacity: 0.8;
            }

            @keyframes rise {
                from {
                    opacity: 0;
                    transform: translateY(14px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .hero {
                grid-column: span 6;
                padding: 28px 28px 24px;
                background:
                    radial-gradient(circle at 40% 35%, rgba(255, 122, 61, 0.18), transparent 60%),
                    radial-gradient(circle at 75% 80%, rgba(255, 168, 114, 0.10), transparent 65%),
                    var(--card);
            }

            .hero-eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--text-soft);
                font-size: 11px;
                font-weight: 700;
            }

            .hero h1 {
                margin: 10px 0 12px;
                font-family: "Space Grotesk", sans-serif;
                font-size: clamp(28px, 3.4vw, 52px);
                line-height: 1.05;
                max-width: 13ch;
            }

            .hero p {
                margin: 0;
                max-width: 62ch;
                color: #c7ccda;
                line-height: 1.5;
                font-size: 14px;
            }

            .hero-badges {
                margin-top: 18px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            .badge {
                border-radius: 999px;
                border: 1px solid rgba(255, 122, 61, 0.3);
                background: rgba(255, 122, 61, 0.12);
                color: var(--accent-soft);
                padding: 7px 12px;
                font-size: 12px;
                font-weight: 700;
            }

            .stat-stack {
                grid-column: span 3;
                display: grid;
                grid-template-rows: repeat(2, minmax(136px, 1fr));
                gap: 14px;
            }

            .stat-card {
                padding: 18px 20px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .stat-title {
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-size: 11px;
                color: var(--text-soft);
                font-weight: 700;
            }

            .stat-value {
                font-family: "Space Grotesk", sans-serif;
                font-size: clamp(32px, 4vw, 58px);
                line-height: 1;
                color: var(--accent);
            }

            .stat-sub {
                margin-top: 8px;
                color: #bcc2d1;
                font-size: 13px;
            }

            .card {
                grid-column: span 3;
                padding: 18px 20px;
            }

            .card-title {
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 11px;
                color: var(--text-soft);
                font-weight: 700;
                margin-bottom: 10px;
            }

            .card-value {
                font-family: "Space Grotesk", sans-serif;
                font-size: 46px;
                line-height: 1;
                color: var(--accent);
                margin-bottom: 4px;
            }

            .card-copy {
                color: #c2c8d7;
                font-size: 13px;
                line-height: 1.45;
            }

            .modes {
                grid-column: span 12;
                padding: 14px 18px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }

            .modes-copy {
                color: var(--text-soft);
                font-size: 13px;
            }

            .mode-toggle {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            .mode-button {
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.03);
                color: #dce0ea;
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 700;
                cursor: pointer;
                transition: all 200ms ease;
            }

            .mode-button:hover {
                border-color: rgba(255, 122, 61, 0.5);
                transform: translateY(-1px);
            }

            .mode-button.active {
                background: rgba(255, 122, 61, 0.2);
                border-color: rgba(255, 122, 61, 0.75);
                color: #ffd3bd;
            }

            .results-strip {
                grid-column: span 12;
                padding: 18px;
            }

            .results-title {
                font-size: 14px;
                font-weight: 700;
                color: #dbe0ec;
                margin: 0 0 12px 4px;
            }

            .results-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
            }

            .result-chip {
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 16px;
                min-width: 0;
            }

            .result-mode {
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 11px;
                color: var(--accent-soft);
                font-weight: 700;
                margin-bottom: 10px;
            }

            .result-line {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 12px;
                color: #c5cad8;
                font-size: 12px;
                padding: 6px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }

            .result-line:last-child {
                border-bottom: none;
                padding-bottom: 0;
            }

            .result-line strong {
                color: #f3f5fb;
                font-size: 13px;
                font-weight: 700;
                text-align: right;
            }

            .kpi-wrap {
                grid-column: span 12;
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
            }

            .kpi-wrap .kpi-card {
                border: 1px solid var(--line);
                border-radius: var(--radius);
                background: var(--card-soft);
                padding: 18px;
                box-shadow: var(--shadow);
            }

            .kpi-wrap .kpi-title {
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 11px;
                color: var(--text-soft);
                font-weight: 700;
                margin-bottom: 8px;
            }

            .kpi-wrap .kpi-value {
                color: var(--text);
                font-family: "Space Grotesk", sans-serif;
                font-size: 26px;
                line-height: 1.1;
            }

            .kpi-wrap .kpi-subtitle {
                margin-top: 6px;
                color: #aab0be;
                font-size: 12px;
            }

            .plot-card {
                grid-column: span 6;
                padding: 14px 14px 6px;
            }

            .plot-wide {
                grid-column: span 8;
                padding: 14px 14px 6px;
            }

            .plot-narrow {
                grid-column: span 4;
                padding: 14px 14px 6px;
            }

            .plot-title {
                font-size: 14px;
                font-weight: 700;
                color: #dbe0ec;
                margin: 0 0 10px 4px;
            }

            .plot-subtitle {
                margin: -6px 0 10px 4px;
                color: var(--text-soft);
                font-size: 12px;
            }

            .plot-container {
                min-height: 220px;
            }

            .summary {
                grid-column: span 12;
                padding: 16px;
            }

            .summary-table-wrap {
                overflow-x: auto;
            }

            .summary-table {
                width: 100%;
                border-collapse: collapse;
                color: #d9ddeb;
                font-size: 13px;
            }

            .summary-table th,
            .summary-table td {
                padding: 10px 10px;
                text-align: left;
                border-bottom: 1px solid var(--line);
                white-space: nowrap;
            }

            .summary-table th {
                color: var(--text-soft);
                text-transform: uppercase;
                letter-spacing: 0.07em;
                font-size: 10px;
                font-weight: 700;
            }

            .summary-table td:first-child {
                color: #f0f2f8;
                font-weight: 600;
            }

            .summary-table tr:last-child td {
                border-bottom: none;
            }

            .legacy-hook {
                display: none;
            }

            @media (max-width: 1200px) {
                .hero {
                    grid-column: span 12;
                }
                .stat-stack, .card {
                    grid-column: span 6;
                }
                .plot-card, .plot-wide, .plot-narrow {
                    grid-column: span 12;
                }
                .kpi-wrap {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .results-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 680px) {
                .page {
                    width: min(1480px, 96vw);
                    margin: 10px auto 24px;
                }
                .dashboard-grid {
                    gap: 10px;
                }
                .stat-stack, .card {
                    grid-column: span 12;
                }
                .hero {
                    padding: 22px 18px 20px;
                }
                .card,
                .stat-card {
                    padding: 16px;
                }
                .kpi-wrap {
                    grid-template-columns: 1fr;
                }
                .results-grid {
                    grid-template-columns: 1fr;
                }
                .modes {
                    padding: 12px;
                }
            }
        </style>
        """,
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
        "</head>",
        "<body>",
        "<!-- DERMS QSTS Dashboard -->",
        "<div class='legacy-hook'>DERMS QSTS Dashboard</div>",
        "<div class='page'>",
        "<div class='dashboard-grid'>",
        f"""
        <section class="panel hero">
            <div class="hero-eyebrow">DERMS Portfolio Case Study</div>
            <h1>Volt-VAR Control for High-PV Feeders</h1>
            <p>{hero_copy}</p>
            <div class="hero-badges">
                <span class="badge">24h QSTS</span>
                <span class="badge">5-min control steps</span>
                <span class="badge">IEEE 13/123 feeder flow</span>
            </div>
        </section>
        """,
        f"""
        <section class="panel stat-stack">
            <article class="panel stat-card">
                <div class="stat-title">Violation Reduction</div>
                <div>
                    <div class="stat-value">{violation_reduction:.0f}%</div>
                    <div class="stat-sub">Optimization vs baseline voltage violations</div>
                </div>
            </article>
            <article class="panel stat-card">
                <div class="stat-title">Scenario Focus</div>
                <div>
                    <div class="stat-value" style="font-size: 42px;">{headline_scale}</div>
                    <div class="stat-sub">Current dashboard context</div>
                </div>
            </article>
        </section>
        """,
        f"""
        <section class="panel card">
            <div class="card-title">Hosting Capacity Gain</div>
            <div class="card-value">{headline_hosting_gain}</div>
            <div class="card-copy">Optimization hosting capacity ratio compared with baseline.</div>
        </section>
        """,
        f"""
        <section class="panel card">
            <div class="card-title">Voltage Envelope</div>
            <div class="card-value">{optimization_kpi.get('max_voltage', 0.0):.3f}</div>
            <div class="card-copy">Peak per-unit voltage in the selected optimization run.</div>
        </section>
        """,
        f"""
        <section class="panel results-strip">
            <h3 class="results-title">Study Results</h3>
            <div class="results-grid">
                {results_snapshot_html}
            </div>
        </section>
        """,
        '<section class="panel modes">',
        '<div class="modes-copy">Switch scenario mode to inspect controller behavior over the same 24-hour period.</div>',
        '<div class="mode-toggle">',
    ]

    for mode in available_modes:
        active_class = " active" if mode == default_mode else ""
        html_parts.append(
            f'<button class="mode-button{active_class}" data-mode="{mode}" onclick="showMode(\'{mode}\')">{_format_mode_label(mode)}</button>'
        )

    html_parts.extend([
        "</div>",
        "</section>",
        '<section class="kpi-wrap" id="mode-kpis"></section>',
        '<section class="panel plot-wide"><h3 class="plot-title">Multi-Mode Voltage Comparison</h3><p class="plot-subtitle">Baseline, heuristic, and optimization voltage envelopes over the same time axis.</p><div class="plot-container" id="comparison-voltage-plot"></div></section>',
        '<section class="panel plot-narrow"><h3 class="plot-title">Hosting Capacity</h3><p class="plot-subtitle">Maximum safe PV scale under project assumptions.</p><div class="plot-container" id="hosting-capacity-plot"></div></section>',
        '<section class="panel plot-card"><h3 class="plot-title">Voltage Envelope</h3><div class="plot-container" id="voltage-plot"></div></section>',
        '<section class="panel plot-card"><h3 class="plot-title">Bus Voltage Heatmap</h3><div class="plot-container" id="heatmap-plot"></div></section>',
        '<section class="panel plot-card"><h3 class="plot-title">Reactive Power Dispatch</h3><div class="plot-container" id="q-plot"></div></section>',
        '<section class="panel plot-card"><h3 class="plot-title">Active Power Curtailment</h3><div class="plot-container" id="p-plot"></div></section>',
    ])

    if has_battery_mode:
        html_parts.extend([
            '<section class="panel plot-card"><h3 class="plot-title">Battery Power</h3><div class="plot-container" id="battery-power-plot"></div></section>',
            '<section class="panel plot-card"><h3 class="plot-title">Battery SOC</h3><div class="plot-container" id="battery-soc-plot"></div></section>',
        ])

    html_parts.extend([
        '<section class="panel summary">',
        '<h3 class="plot-title">Mode Comparison Metrics</h3>',
        _build_mode_summary_table(mode_kpis),
        "</section>",
        "</div>",
        "</div>",
    ])

    html_parts.append(f"""
    <script>
        const MODE_CARDS = {mode_cards_json};
        const MODE_FIGURES = {mode_figures_json};
        const OVERVIEW_FIGURES = {overview_figures_json};
        const HAS_BATTERY_MODE = {str(has_battery_mode).lower()};

        function renderPlot(targetId, figure) {{
            if (!figure || typeof Plotly === "undefined") {{
                return;
            }}
            const target = document.getElementById(targetId);
            if (!target) {{
                return;
            }}
            Plotly.react(target, figure.data || [], figure.layout || {{}}, {{
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ["lasso2d", "select2d"],
            }});
        }}

        function renderOverview() {{
            renderPlot("comparison-voltage-plot", OVERVIEW_FIGURES.comparison_voltage);
            renderPlot("hosting-capacity-plot", OVERVIEW_FIGURES.hosting_capacity);
        }}

        function renderMode(mode) {{
            const kpiContainer = document.getElementById("mode-kpis");
            if (kpiContainer) {{
                kpiContainer.innerHTML = MODE_CARDS[mode] || "";
            }}

            const figures = MODE_FIGURES[mode] || {{}};
            renderPlot("voltage-plot", figures.voltage);
            renderPlot("heatmap-plot", figures.heatmap);
            renderPlot("q-plot", figures.q);
            renderPlot("p-plot", figures.p);

            if (HAS_BATTERY_MODE) {{
                renderPlot("battery-power-plot", figures.battery_power);
                renderPlot("battery-soc-plot", figures.battery_soc);
            }}
        }}

        function showMode(mode) {{
            document.querySelectorAll(".mode-button").forEach((button) => {{
                button.classList.toggle("active", button.dataset.mode === mode);
            }});
            renderMode(mode);
        }}

        window.addEventListener("load", () => {{
            renderOverview();
            renderMode("{default_mode}");
        }});

        let resizeTimer;
        window.addEventListener("resize", () => {{
            window.clearTimeout(resizeTimer);
            resizeTimer = window.setTimeout(() => {{
                renderOverview();
                const activeButton = document.querySelector(".mode-button.active");
                const activeMode = activeButton ? activeButton.dataset.mode : "{default_mode}";
                renderMode(activeMode);
            }}, 180);
        }});
    </script>
    """)

    html_parts.extend(["</body>", "</html>"])

    # Write to file
    if output_path is None:
        output_path = results_dir / "dashboard.html"

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    return str(output_path)
