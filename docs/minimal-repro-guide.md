# Minimal Repro Guide

Short procedure for running the smallest reproducible DERMS study in this repository. Each step names the starting files, command, outputs, and what the result means.

## Prerequisites

- Python 3.11
- OpenDSS available through `OpenDSSDirect.py`
- Either Conda or pip for dependency installation

Install with Conda:

```bash
conda env create -f environment.yml
conda activate derms-mvp
```

Install with pip:

```bash
pip install -r requirements.txt
```

## Starting Data

- Feeder config: `config/feeder_dev_ieee13.yaml`
- Baseline study config: `config/study_mvp.yaml`
- Heuristic study config: `config/study_heuristic.yaml`
- Optimization study config: `config/study_optimization.yaml`
- Load profile: `data/load_profiles/load_24h.csv`
- PV profile: `data/pv_profiles/pv_clear_sky_24h.csv`
- DER placement: `data/der_configs/ders_ieee13.csv`
- Feeder master file used by the development config: `feeder_models/ieee13/IEEE13Nodeckt.dss`

## Step 1: Snapshot Validation

Inputs:

- `config/feeder_dev_ieee13.yaml`
- `feeder_models/ieee13/IEEE13Nodeckt.dss`

Command:

```bash
python -m src.sim.run_snapshot --config config/feeder_dev_ieee13.yaml
```

Expected outputs:

- `results/baseline/bus_voltages.csv`
- `results/baseline/bus_voltages.png`

Interpretation:

This confirms the feeder loads, solves, and exports voltage results before running any 24-hour study.

## Step 2: Baseline 24-Hour QSTS

Inputs:

- `config/study_mvp.yaml`
- `config/feeder_dev_ieee13.yaml`
- `data/load_profiles/load_24h.csv`
- `data/pv_profiles/pv_clear_sky_24h.csv`
- `data/der_configs/ders_ieee13.csv`

Command:

```bash
python -m src.sim.run_qsts --config config/study_mvp.yaml
```

Expected outputs:

- `results/baseline/qsts_baseline.csv`
- `results/baseline/qsts_summary.csv`
- `results/baseline/qsts_violations.csv` when the run includes voltage violations
- `results/baseline/plot_voltage_envelope.png`
- `results/baseline/plot_voltage_histogram.png`
- `results/baseline/plot_pv_vs_voltage.png`
- `results/baseline/plot_violation_timeline.png`

Interpretation:

This is the uncontrolled reference case used for every later comparison.

## Step 3: Controlled Run

Inputs:

- `config/study_heuristic.yaml` or `config/study_optimization.yaml`
- `config/feeder_dev_ieee13.yaml`
- `data/load_profiles/load_24h.csv`
- `data/pv_profiles/pv_clear_sky_24h.csv`
- `data/der_configs/ders_ieee13.csv`

Command:

```bash
python -m src.sim.run_qsts --config config/study_heuristic.yaml
```

Or:

```bash
python -m src.sim.run_qsts --config config/study_optimization.yaml
```

Expected outputs for heuristic:

- `results/heuristic/qsts_baseline.csv`
- `results/heuristic/qsts_summary.csv`
- `results/heuristic/commands.csv`
- `results/heuristic/command_summary.csv`
- plot files in `results/heuristic/`

Expected outputs for optimization:

- `results/optimization/qsts_baseline.csv`
- `results/optimization/qsts_summary.csv`
- `results/optimization/commands.csv`
- `results/optimization/command_summary.csv`
- plot files in `results/optimization/`

Interpretation:

The controlled run should show fewer voltage violations than the baseline. `commands.csv` is a per-DER simulation audit log with requested setpoints, Q/P-specific apply status, before/after DER state, and before/after local voltage. `command_summary.csv` gives showcase-level command KPIs such as commands sent, failures, out-of-range commands, controlled DER count, reactive energy, and curtailed energy.

## Step 4: Optional Full Analysis

Inputs:

- `config/study_mvp.yaml`
- `config/study_heuristic.yaml`
- `config/study_optimization.yaml`

Command:

```bash
python -m src.analysis.run_phase5_analysis \
    --baseline-config config/study_mvp.yaml \
    --heuristic-config config/study_heuristic.yaml \
    --optimization-config config/study_optimization.yaml \
    --output results/phase5
```

Expected outputs:

- `results/phase5/`
- comparison reports and plots
- packaged per-mode CSV files
- `dashboard.html`

Interpretation:

This packages the advanced analysis artifacts after the minimal workflow is already working.
