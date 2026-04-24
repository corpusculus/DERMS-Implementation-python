# Results Directory

Simulation outputs are organized by control mode:

- `baseline/`: no DERMS control
- `heuristic/`: threshold-based DER control
- `optimization/`: optimization-based DER control

Each mode folder contains:

- `qsts_baseline.csv`: 24-hour time-series simulation output
- `qsts_summary.csv`: mode-level KPI summary
- `qsts_violations.csv`: voltage violation rows
- `plots/`: plots for that mode or comparison against baseline
- `hosting_capacity/`: PV hosting-capacity sweep data for that mode

Controlled modes also include DER command logs:

- `commands.csv`: per-timestep DER setpoint audit log
- `command_summary.csv`: per-DER command summary

Shared cross-mode outputs live in:

- `comparison/`: KPI tables, hosting-capacity comparison plots, and `analysis_summary.json`
- `dashboard.html`: interactive dashboard
- `docs/`: generated supporting documentation
