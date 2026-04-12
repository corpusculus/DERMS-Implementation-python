# Heuristic Volt-VAR Controller Design

**Date:** 2026-03-17
**Phase:** Phase 3 - Controller v0
**Status:** Approved

## Overview

Implement a centralized heuristic Volt-VAR controller that prioritizes reactive power absorption before active power curtailment to mitigate voltage violations on high-PV distribution feeders.

## Architecture

```
QSTS Loop (per timestep):
┌─────────────────────────────────────────────────────────────┐
│ 1. Update load/PV from profiles                             │
│ 2. Solve initial power flow                                 │
│ 3. Read voltages and DER states                             │
│ 4. ┌─────────────────────────────────────────────────────┐  │
│    │  HeuristicController.compute_commands()             │  │
│    │  - Check all bus voltages vs thresholds             │  │
│    │  - Calculate Q commands (equal dispatch)            │  │
│    │  - If needed, calculate P curtailment (proportional)│  │
│    └─────────────────────────────────────────────────────┘  │
│ 5. Apply commands via DERInterface                          │
│ 6. Re-solve power flow                                      │
│ 7. Log commands + post-control voltages                     │
└─────────────────────────────────────────────────────────────┘
```

**Key design decision:** The controller is stateless — it reads current voltages, computes commands, and returns them. No persistent state between timesteps.

## Control Logic

### Voltage Thresholds (from config)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| VAR activation | 1.03 pu | Start reactive absorption |
| Curtailment | 1.05 pu | Allow active curtailment |
| Deadband | ±0.005 pu | No action within 1.0 ± 0.005 |

### Algorithm

1. **Find max voltage** across all buses
2. **Calculate severity:** `severity = max(0, (max_voltage - 1.03) / 0.02)` clipped to [0, 1]
3. **Deadband check:** No action if `|max_voltage - 1.0| < 0.005`
4. **VAR dispatch** (if max_voltage > 1.03):
   - All enabled DERs absorb equally: `q_command = -severity × q_max`
   - Each DER limited by individual `q_max_kvar`
5. **Curtailment** (if max_voltage > 1.05 after VAR):
   - Each DER curtails: `p_curtail = severity × p_avail × curtailment_factor`
   - Curtailment factor starts at 0.1, ramps if voltage persists
6. **Return:** `q_commands`, `p_curtail`, `reason`

### Dispatch Strategy

- **VAR:** Equal dispatch to all enabled DERs (fair, simple)
- **Curtailment:** Proportional across all DERs (spreads burden fairly)

## File Structure

### New Files

```
src/control/heuristic_controller.py  # HeuristicController class
src/analysis/kpis.py                 # KPI calculation and comparison
tests/test_heuristic_controller.py   # Unit tests
```

### Modified Files

```
src/sim/run_qsts.py       # Integrate controller into QSTS loop
src/analysis/plots.py     # Add comparative plots
```

## Interface

```python
class HeuristicController:
    def __init__(self, thresholds: dict):
        """Initialize with voltage thresholds from config."""
        self.q_activation_pu = thresholds["q_activation_pu"]
        self.curtailment_pu = thresholds["curtailment_pu"]
        self.deadband_pu = thresholds["deadband_pu"]

    def compute_commands(
        self,
        bus_voltages: dict[str, float],  # bus_name -> pu_voltage
        ders: DERContainer
    ) -> dict:
        """Compute Q and P commands for all DERs.

        Returns:
            {
                "q_commands": {der_id: q_kvar, ...},
                "p_curtail": {der_id: p_kw, ...},
                "reason": "var_dispatch" | "curtailment" | "no_action",
                "max_voltage": float,
                "severity": float
            }
        """
```

## Metrics

### New Metrics (beyond baseline)

- Total VAR dispatched per timestep
- Number of DERs actively absorbing VARs
- Total curtailed energy (kWh)
- Curtailment as % of available PV energy
- Pre-control vs post-control voltage comparison

### Output Files

```
results/heuristic/qsts_results.csv    # Same format as baseline
results/heuristic/commands.csv        # CommandLogger output
results/heuristic/kpi_comparison.csv  # Baseline vs heuristic
```

## Error Handling

### Guardrails (inherited from DERInterface)

- Q outside `[q_min, q_max]` → clamped, logged as "out_of_range"
- P curtailment < 0 or > p_avail → rejected, logged as "failed"
- Missing DER buses → skipped, logged as "failed"

### Heuristic-Specific

- No DERs enabled → return "no_action"
- All DERs at Q_max → proceed to curtailment logic
- OpenDSS solve fails → log error, continue with previous state
- Voltage NaN/inf → skip control, log warning

## Testing

### Unit Tests

- Deadband behavior (no action within ±0.005 pu)
- VAR activation at exactly 1.03 pu
- Curtailment activation at 1.05 pu
- Equal dispatch across DERs
- Proportional curtailment
- Edge cases (no DERs, all at Q_max)

### Integration Test

- Run 24-hour on IEEE 13-bus
- Verify no errors
- Verify violations reduced vs baseline
- Verify reactive power before curtailment

## Definition of Done (from TODO.md)

- [x] Heuristic controller runs end-to-end on dev feeder
- [x] Reduces voltage violations vs baseline
- [x] Curtailment low relative to total PV energy
- [x] Logs show reactive power used before curtailment

## Implementation Tasks

1. Create `src/control/heuristic_controller.py`
2. Modify `src/sim/run_qsts.py` for controller integration
3. Enhance `src/analysis/plots.py` with comparative plots
4. Create `src/analysis/kpis.py`
5. Update configs for controller mode support
6. Create `tests/test_heuristic_controller.py`
7. Run end-to-end validation on IEEE 13-bus
8. Generate all plots and KPIs
