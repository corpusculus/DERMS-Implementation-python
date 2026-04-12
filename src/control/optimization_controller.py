"""Optimization-based Volt-VAR controller with P curtailment.

This module provides a convex optimization-based controller that:
1. Uses voltage sensitivity matrices (M_Q, M_P) to predict voltage impact
2. Solves a convex optimization problem to minimize violations and curtailment
3. Falls back to heuristic control if optimization fails
4. Prioritizes reactive power before active power curtailment
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    CVXPY_AVAILABLE = False

from src.control.der_models import DER, DERContainer
from src.control.sensitivity_estimator import (
    SensitivityEstimator,
    SensitivityConfig,
    SensitivityResult,
)
from src.control.heuristic_controller import (
    HeuristicController,
    HeuristicConfig,
)


@dataclass
class OptimizationConfig:
    """Configuration for the optimization controller."""

    # Objective function weights
    alpha_violation: float = 1000.0
    # Voltage violation penalty weight

    beta_q_effort: float = 1.0
    # Reactive power effort penalty weight

    gamma_p_curtail: float = 100.0
    # Active power curtailment penalty weight

    # Voltage limits
    v_min: float = 0.95
    # Lower voltage limit (pu)

    v_max: float = 1.05
    # Upper voltage limit (pu)

    # Solver settings
    solver: str = "ECOS"
    # CVXPY solver: ECOS (fast), OSQP (robust), SCS (accurate)

    max_iterations: int = 1000
    # Maximum solver iterations

    tolerance: float = 1e-6
    # Solver tolerance

    # Fallback settings
    enable_fallback: bool = True
    # Enable heuristic fallback if optimization fails

    # Sensitivity config
    sensitivity_config: SensitivityConfig = field(default_factory=SensitivityConfig)
    # Configuration for sensitivity estimation


class OptimizationController:
    """Convex optimization-based Volt-VAR controller with P curtailment.

    Solves a convex optimization problem at each timestep to find the optimal
    reactive power and curtailment setpoints that minimize voltage violations
    while penalizing control effort and energy curtailment.

    Optimization formulation:
        minimize    alpha * ||s||^2 + beta * ||Q||^2 + gamma * ||P_curt||^2
        subject to  V_pred + s >= v_min
                    V_pred + s <= v_max
                    Q_min <= Q <= Q_max
                    0 <= P_curt <= P_avail

    Where:
        V_pred = V0 + M_Q @ (Q - Q0) + M_P @ P_curt
        s = Voltage slack variable for soft constraints

    Falls back to heuristic control if:
    - CVXPY is not available
    - Solver fails (exception, infeasible, timeout)
    """

    def __init__(
        self,
        config: OptimizationConfig,
        der_container: DERContainer,
        heuristic_controller: HeuristicController | None = None,
    ):
        """Initialize the optimization controller.

        Args:
            config: OptimizationConfig with weights and solver settings.
            der_container: DERContainer with all DERs.
            heuristic_controller: Optional heuristic controller for fallback.
                If None and enable_fallback=True, creates a default one.
        """
        self.config = config
        self.der_container = der_container
        self.sensitivity_estimator = SensitivityEstimator(config.sensitivity_config)

        # Set up fallback controller
        if config.enable_fallback:
            if heuristic_controller is None:
                # Create default heuristic controller
                heuristic_config = HeuristicConfig(
                    q_activation_pu=1.03,
                    curtailment_pu=1.05,
                    deadband_pu=0.005,
                    v_lower_limit=config.v_min,
                )
                self.heuristic_controller = HeuristicController(heuristic_config, der_container)
            else:
                self.heuristic_controller = heuristic_controller
        else:
            self.heuristic_controller = None

        # Status tracking
        self._last_status: str = "N/A"
        self._cache_hit: bool = False

    def compute_commands(
        self,
        bus_voltages: dict[str, float],
        previous_commands: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute optimal Q and P commands for all DERs.

        Args:
            bus_voltages: Current bus voltages from OpenDSS (pu).
            previous_commands: Previous timestep commands as {der_id: (q, p)}
                for ramp limiting (not used in optimization, but tracked).

        Returns:
            (q_commands, p_commands) dictionaries mapping DER ID to setpoints:
                - q_commands: DER ID -> Q setpoint (kVAR), negative = absorption
                - p_commands: DER ID -> P curtailment (kW), 0 = no curtailment
        """
        if not CVXPY_AVAILABLE:
            self._last_status = "CVXPY not available"
            return self._fallback_or_zero(bus_voltages)

        max_v = max(bus_voltages.values(), default=1.0)
        min_v = min(bus_voltages.values(), default=1.0)
        if max_v <= self.config.v_max and min_v >= self.config.v_min:
            self._last_status = "no action needed"
            return {}, {}

        # Get enabled DERs
        enabled_ders = list(self.der_container.enabled())
        if not enabled_ders:
            return {}, {}

        buses = list(bus_voltages.keys())

        # Compute sensitivities
        try:
            M_Q, M_P = self.sensitivity_estimator.compute_sensitivities(
                self.der_container, buses, bus_voltages
            )
            self._cache_hit = self.sensitivity_estimator.cache_hit
        except RuntimeError as e:
            self._last_status = f"Sensitivity failed: {e}"
            return self._fallback_or_zero(bus_voltages)

        # Build and solve optimization problem
        try:
            q_commands, p_commands = self._solve_optimization(
                enabled_ders, buses, bus_voltages, M_Q, M_P
            )
            self._last_status = "optimal"
            return q_commands, p_commands

        except Exception as e:
            self._last_status = f"Optimization failed: {e}"
            return self._fallback_or_zero(bus_voltages)

    def _solve_optimization(
        self,
        ders: list[DER],
        buses: list[str],
        voltages: dict[str, float],
        M_Q: dict[str, dict[str, float]],
        M_P: dict[str, dict[str, float]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Solve the convex optimization problem.

        Args:
            ders: List of enabled DERs.
            buses: List of bus names.
            voltages: Current bus voltages.
            M_Q: Q sensitivity matrix [der_id][bus].
            M_P: P sensitivity matrix [der_id][bus].

        Returns:
            (q_commands, p_commands) optimal setpoints.
        """
        n_ders = len(ders)
        n_buses = len(buses)

        if n_ders == 0 or n_buses == 0:
            return {}, {}

        # Create DER ID to index mapping
        der_ids = [der.id for der in ders]
        der_index = {der_id: i for i, der_id in enumerate(der_ids)}
        bus_index = {bus: i for i, bus in enumerate(buses)}

        # Build sensitivity matrices as numpy arrays
        M_Q_mat = np.zeros((n_buses, n_ders))
        M_P_mat = np.zeros((n_buses, n_ders))

        for der_id, sensitivities in M_Q.items():
            if der_id in der_index:
                i = der_index[der_id]
                for bus, sens in sensitivities.items():
                    if bus in bus_index:
                        j = bus_index[bus]
                        M_Q_mat[j, i] = sens

        for der_id, sensitivities in M_P.items():
            if der_id in der_index:
                i = der_index[der_id]
                for bus, sens in sensitivities.items():
                    if bus in bus_index:
                        j = bus_index[bus]
                        M_P_mat[j, i] = sens

        # Current operating point
        V0 = np.array([voltages.get(bus, 1.0) for bus in buses])
        Q0 = np.array([der.q_kvar for der in ders])
        P_avail = np.array([der.p_avail_kw for der in ders])
        Q_max = np.array([der.q_max_kvar for der in ders])
        Q_min = np.array([der.q_min_kvar for der in ders])

        # Decision variables
        q = cp.Variable(n_ders)  # Reactive power setpoints
        p_curt = cp.Variable(n_ders)  # Active power curtailment
        s = cp.Variable(n_buses)  # Voltage slack (for soft constraints)

        # Predicted voltage (linearized)
        V_pred = V0 + M_Q_mat @ (q - Q0) + M_P_mat @ p_curt

        # Objective
        violation_penalty = self.config.alpha_violation * cp.sum_squares(s)
        q_effort_penalty = self.config.beta_q_effort * cp.sum_squares(q / (Q_max + 1e-6))
        p_curtail_penalty = self.config.gamma_p_curtail * cp.sum_squares(p_curt / (P_avail + 1e-6))

        objective = cp.Minimize(
            violation_penalty + q_effort_penalty + p_curtail_penalty
        )

        # Constraints
        constraints = [
            # Q limits
            q >= Q_min,
            q <= Q_max,
            # P curtailment limits
            p_curt >= 0,
            p_curt <= P_avail,
            # Voltage limits (soft constraints via slack)
            V_pred + s >= self.config.v_min,
            V_pred - s <= self.config.v_max,
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(
                solver=getattr(cp, self.config.solver, cp.ECOS),
                max_iters=self.config.max_iterations,
                eps=self.config.tolerance,
            )
        except Exception:
            # Try fallback solver if primary fails
            problem.solve(solver=cp.SCS)

        # Check solution status
        if problem.status in ["optimal", "optimal_inaccurate"]:
            q_values = q.value if q.value is not None else np.zeros(n_ders)
            p_values = p_curt.value if p_curt.value is not None else np.zeros(n_ders)

            # Convert to command dictionaries
            q_commands: dict[str, float] = {}
            p_commands: dict[str, float] = {}

            for i, der_id in enumerate(der_ids):
                q_cmd = float(q_values[i])
                p_cmd = float(p_values[i])

                # Only include non-zero commands
                if abs(q_cmd) > 1e-6:
                    q_commands[der_id] = q_cmd
                if p_cmd > 1e-6:
                    p_commands[der_id] = p_cmd

            return q_commands, p_commands

        else:
            # Infeasible or solver error - raise to trigger fallback
            raise ValueError(f"Solver status: {problem.status}")

    def _fallback_or_zero(
        self,
        bus_voltages: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Use heuristic fallback or return zero commands.

        Args:
            bus_voltages: Current bus voltages.

        Returns:
            (q_commands, p_commands) from fallback or empty dicts.
        """
        if self.heuristic_controller is not None:
            self._last_status = f"{self._last_status} (fallback)"
            return self.heuristic_controller.compute_commands(bus_voltages)
        return {}, {}

    @property
    def last_status(self) -> str:
        """Last optimization status."""
        return self._last_status

    @property
    def cache_hit(self) -> bool:
        """Whether last sensitivity computation hit the cache."""
        return self._cache_hit
