"""Tests for Phase 5 orchestration and QSTS cleanup behavior."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd


def test_run_qsts_ignores_opendss_reset_failures(monkeypatch, tmp_path, capsys):
    """OpenDSS reset warnings should not mask successful QSTS results."""
    hosting_capacity_stub = ModuleType("src.analysis.hosting_capacity")
    for name in [
        "run_pv_sweep",
        "find_hosting_capacity",
        "find_hosting_capacity_interpolated",
        "find_hosting_capacity_binary",
        "compare_hosting_capacity",
        "plot_hosting_capacity_comparison",
        "plot_sweep_results",
        "plot_voltage_vs_pv_scale",
    ]:
        setattr(hosting_capacity_stub, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "src.analysis.hosting_capacity", hosting_capacity_stub)

    import src.sim.run_qsts as run_qsts_module

    config_path = tmp_path / "config" / "study.yaml"

    def fake_load_config(path):
        path = Path(path)
        if path.name == "study.yaml":
            return {
                "feeder_config": "config/feeder.yaml",
                "profiles": {
                    "load_profile": "data/load.csv",
                    "pv_profile": "data/pv.csv",
                },
                "controller": {"mode": "baseline"},
            }
        if path.name == "feeder.yaml":
            return {
                "simulation": {"time_step_minutes": 5},
                "voltage_limits": {"lower": 0.95, "upper": 1.05},
                "feeder": {
                    "name": "ieee13",
                    "master_file": "feeder_models/ieee13/IEEE13Nodeckt.dss",
                },
            }
        raise AssertionError(f"Unexpected config path: {path}")

    monkeypatch.setattr(run_qsts_module, "load_config", fake_load_config)
    monkeypatch.setattr(run_qsts_module, "load_load_profile", lambda *_args: {0: 1.0})
    monkeypatch.setattr(run_qsts_module, "load_pv_profile", lambda *_args: {0: 0.5})
    monkeypatch.setattr(run_qsts_module, "load_feeder", lambda *_args: None)
    monkeypatch.setattr(run_qsts_module, "get_original_load_totals", lambda: {"kw": 100.0, "kvar": 50.0})
    monkeypatch.setattr(run_qsts_module, "place_pv_list", lambda *_args, **_kwargs: ["pv_001"])
    monkeypatch.setattr(run_qsts_module, "get_total_pv_capacity_kw", lambda: 50.0)
    monkeypatch.setattr(run_qsts_module, "update_loads_from_base", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qsts_module, "update_pv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_qsts_module, "solve_power_flow", lambda: None)
    monkeypatch.setattr(run_qsts_module, "get_bus_voltages", lambda: {"bus1": 1.0})
    fake_dss = SimpleNamespace(
        Solution=SimpleNamespace(Solve=lambda: None),
        Circuit=SimpleNamespace(Losses=lambda: (1000.0, 0.0)),
        run_command=lambda command: (_ for _ in ()).throw(RuntimeError("reset failed")) if command == "Clear" else "",
    )
    monkeypatch.setattr(run_qsts_module, "dss", fake_dss)

    results = run_qsts_module.run_qsts(config_path)

    assert len(results) == 1
    assert results.loc[0, "v_max"] == 1.0

    output = capsys.readouterr().out
    assert "Warning: Failed to reset OpenDSS state: reset failed" in output


def test_phase5_uses_existing_results_for_comparison_and_plots(monkeypatch, tmp_path):
    """Phase 5 should complete comparison and plotting when results exist."""
    from src.analysis import run_phase5_analysis as phase5

    for mode in ["baseline", "heuristic", "optimization"]:
        mode_dir = tmp_path / mode
        mode_dir.mkdir()
        pd.DataFrame(
            {
                "step": [0],
                "time_h": [0.0],
                "v_min": [0.99],
                "v_max": [1.01],
                "violating_buses_count": [0],
                "overvoltage_buses_count": [0],
                "undervoltage_buses_count": [0],
                "losses_kw": [1.0],
                "pv_generation_kw": [10.0],
                "total_q_dispatch_kvar": [0.0],
                "total_p_curtailment_kw": [0.0],
                "ders_controlled": [0],
            }
        ).to_csv(mode_dir / "qsts_baseline.csv", index=False)

    def fake_load_config(path):
        path = Path(path)
        return {"output": {"qsts_dir": str(tmp_path / path.stem.replace("study_", ""))}}

    monkeypatch.setattr(phase5, "load_config", fake_load_config)
    monkeypatch.setattr(
        phase5,
        "generate_comparison_report",
        lambda *_args, **_kwargs: {"comparison_csv": tmp_path / "comparison.csv"},
    )
    monkeypatch.setattr(
        phase5,
        "create_baseline_plots",
        lambda *_args, **_kwargs: {"voltage_envelope": tmp_path / "baseline_plot.png"},
    )
    monkeypatch.setattr(
        phase5,
        "create_comparison_plots",
        lambda *_args, **_kwargs: {"voltage_comparison": tmp_path / "comparison_plot.png"},
    )

    results = phase5.run_full_phase5_analysis(
        baseline_config="config/study_baseline.yaml",
        heuristic_config="config/study_heuristic.yaml",
        optimization_config="config/study_optimization.yaml",
        output_dir=tmp_path / "phase5",
        run_hosting_capacity=False,
        skip_simulations=True,
    )

    assert results["comparison"]["status"] == "success"
    assert results["plots"]["status"] == "success"


def test_phase5_reports_missing_results_clearly(monkeypatch, tmp_path):
    """Phase 5 should emit a direct missing-results error instead of KeyError noise."""
    from src.analysis import run_phase5_analysis as phase5

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    pd.DataFrame(
        {
            "step": [0],
            "time_h": [0.0],
            "v_min": [0.99],
            "v_max": [1.01],
            "violating_buses_count": [0],
            "overvoltage_buses_count": [0],
            "undervoltage_buses_count": [0],
            "losses_kw": [1.0],
            "pv_generation_kw": [10.0],
            "total_q_dispatch_kvar": [0.0],
            "total_p_curtailment_kw": [0.0],
            "ders_controlled": [0],
        }
    ).to_csv(baseline_dir / "qsts_baseline.csv", index=False)

    def fake_load_config(path):
        path = Path(path)
        mapping = {
            "study_baseline.yaml": baseline_dir,
            "study_heuristic.yaml": tmp_path / "heuristic_missing",
            "study_optimization.yaml": tmp_path / "optimization_missing",
        }
        return {"output": {"qsts_dir": str(mapping[path.name])}}

    monkeypatch.setattr(phase5, "load_config", fake_load_config)

    results = phase5.run_full_phase5_analysis(
        baseline_config="config/study_baseline.yaml",
        heuristic_config="config/study_heuristic.yaml",
        optimization_config="config/study_optimization.yaml",
        output_dir=tmp_path / "phase5",
        run_hosting_capacity=False,
        skip_simulations=True,
    )

    assert results["comparison"]["status"] == "failed"
    assert results["plots"]["status"] == "failed"
    assert "Missing simulation results for: heuristic, optimization" in results["comparison"]["error"]
    assert "Missing simulation results for: heuristic, optimization" in results["plots"]["error"]
