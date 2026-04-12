# DERMS MVP

Python and OpenDSS-based simulation of a simplified Distributed Energy Resource Management System (DERMS) for feeder voltage control under high photovoltaic penetration. The project runs quasi-static time-series studies, applies centralized DER dispatch, and exports analysis artifacts for comparison across baseline, heuristic, optimization, and battery-supported control modes.

This repository is a research and portfolio demonstration, not a production DERMS implementation. It focuses on reproducible studies, transparent inputs, and inspectable outputs rather than market integration, protocol interoperability, or field deployment.

## What Is Included

- Config-driven feeder studies in `config/`
- Input profiles and DER placement files in `data/`
- Included IEEE 13-bus and IEEE 123-bus feeder assets in `feeder_models/`
- Python simulation, control, and analysis modules in `src/`
- Pytest coverage in `tests/`

If you publish this repository externally, keep attribution to IEEE test feeder sources and confirm you are comfortable redistributing the committed feeder assets.

## Stack

- Python 3.11
- OpenDSS via `OpenDSSDirect.py`
- `numpy`, `pandas`, `scipy`
- `cvxpy`
- `matplotlib`, `plotly`
- `pytest`

## Architecture

The workflow is: load a feeder and study config, read 24-hour load and PV profiles, place DERs from CSV, run 5-minute OpenDSS power flows across the day, optionally compute DER setpoints through heuristic or optimization control, and export voltages, summaries, command logs, and comparison plots.

## Minimal Quickstart

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate derms-mvp
```

Or:

```bash
pip install -r requirements.txt
```

### 2. Validate the feeder with a single snapshot

```bash
python -m src.sim.run_snapshot --config config/feeder_dev_ieee13.yaml
```

Expected outputs:

- `results/baseline/bus_voltages.csv`
- `results/baseline/bus_voltages.png`

### 3. Run the baseline 24-hour study

```bash
python -m src.sim.run_qsts --config config/study_mvp.yaml
```

Expected outputs:

- `results/baseline/qsts_baseline.csv`
- `results/baseline/qsts_summary.csv`
- `results/baseline/qsts_violations.csv` when violations occur
- plots such as `results/baseline/plot_voltage_envelope.png`

### 4. Run a controlled study

```bash
python -m src.sim.run_qsts --config config/study_heuristic.yaml
python -m src.sim.run_qsts --config config/study_optimization.yaml
```

Expected outputs:

- `results/heuristic/`
- `results/optimization/`

The controlled runs should reduce overvoltage duration relative to the baseline and log the reactive and active power actions used to do it.

## Reproducibility Guide

For the short step-by-step version with starting files, commands, and expected result locations, see [docs/minimal-repro-guide.md](docs/minimal-repro-guide.md).

## Implemented Capabilities

- Snapshot feeder validation
- 24-hour QSTS baseline runs
- CSV-driven PV and DER placement
- Heuristic Volt-VAR and curtailment control
- Optimization-based DER dispatch with heuristic fallback
- Battery-integrated study mode
- KPI aggregation, plots, hosting-capacity analysis, and dashboard packaging

## Additional Docs

- Public release checklist: [docs/public_release_checklist.md](docs/public_release_checklist.md)
- Assumptions and definitions: [docs/assumptions.md](docs/assumptions.md)
- Executive summary: [docs/executive_summary.md](docs/executive_summary.md)

## Advanced Analysis

To run the full packaged analysis:

```bash
python -m src.analysis.run_phase5_analysis \
    --baseline-config config/study_mvp.yaml \
    --heuristic-config config/study_heuristic.yaml \
    --optimization-config config/study_optimization.yaml \
    --output results/phase5
```

Optional battery-integrated package:

```bash
python -m src.analysis.run_phase5_analysis \
    --baseline-config config/study_mvp.yaml \
    --heuristic-config config/study_heuristic.yaml \
    --optimization-config config/study_optimization.yaml \
    --battery-config config/study_battery.yaml \
    --output results/phase5
```

## Test Suite

```bash
pytest tests/ -v
```

Some tests require the Python dependencies and OpenDSS bindings to be installed.
