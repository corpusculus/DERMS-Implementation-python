"""Control layer for DERMS simulation.

This module provides:
- DER data model (dataclass with type hints)
- OpenDSS interface for reading/writing DER state
- Command logging with audit trail
- Battery storage models and control
"""

from src.control.der_models import DER, DERContainer
from src.control.der_interface import (
    SetpointApplyResult,
    load_ders_from_csv,
    read_der_state,
    apply_q_setpoint,
    apply_p_curtailment,
    apply_setpoints,
)
from src.control.command_log import CommandLogger, summarize_command_log
from src.control.heuristic_controller import (
    HeuristicController,
    HeuristicConfig,
)
from src.control.sensitivity_estimator import (
    SensitivityEstimator,
    SensitivityConfig,
)
from src.control.optimization_controller import (
    OptimizationController,
    OptimizationConfig,
)
from src.control.battery_models import (
    Battery,
    BatteryContainer,
    BatteryConfig,
)
from src.control.battery_controller import (
    BatteryController,
    BatteryControlConfig,
)

__all__ = [
    "DER",
    "DERContainer",
    "SetpointApplyResult",
    "load_ders_from_csv",
    "read_der_state",
    "apply_q_setpoint",
    "apply_p_curtailment",
    "apply_setpoints",
    "CommandLogger",
    "summarize_command_log",
    "HeuristicController",
    "HeuristicConfig",
    "SensitivityEstimator",
    "SensitivityConfig",
    "OptimizationController",
    "OptimizationConfig",
    "Battery",
    "BatteryContainer",
    "BatteryConfig",
    "BatteryController",
    "BatteryControlConfig",
]
