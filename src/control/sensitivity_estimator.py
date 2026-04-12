"""Voltage sensitivity estimation for optimization-based DER control.

This module provides voltage sensitivity computation using finite-difference
perturbation. It calculates how voltage at each bus responds to changes in
reactive power (M_Q) and active power (M_P) at each DER.

The sensitivity matrices are used by the optimization controller to predict
the voltage impact of control actions and solve for optimal setpoints.
"""

from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np

try:
    import opendssdirect as dss
    OPENDSS_AVAILABLE = True
except ImportError:  # pragma: no cover
    dss = None  # type: ignore[assignment]
    OPENDSS_AVAILABLE = False

from src.control.der_models import DER, DERContainer
from src.sim.opendss_interface import _require_opendss


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity estimation."""

    q_perturbation_pct: float = 0.10
    # Perturb Q by 10% of Q_max as default

    p_perturbation_pct: float = 0.05
    # Perturb P by 5% of P_avail as default

    min_perturbation: float = 1.0
    # Minimum perturbation in kVAR/kW

    cache_sensitivities: bool = True
    # Cache sensitivity results for reuse

    cache_valid_minutes: int = 30
    # How long cached sensitivities remain valid (minutes)


@dataclass
class SensitivityResult:
    """Result of sensitivity computation."""

    M_Q: dict[str, dict[str, float]]
    # M_Q[der_id][bus] = dV_bus/dQ_der

    M_P: dict[str, dict[str, float]]
    # M_P[der_id][bus] = dV_bus/dP_der

    timestamp: float
    # When sensitivities were computed (epoch time)

    v0: dict[str, float]
    # Base voltages when sensitivities were computed

    q0: dict[str, float]
    # Base Q setpoints when sensitivities were computed

    p0: dict[str, float]
    # Base P dispatch when sensitivities were computed


class SensitivityEstimator:
    """Computes voltage sensitivity matrices via finite-difference perturbation.

    The sensitivity matrices quantify how voltage at each bus responds to
    changes in DER output:
    - M_Q[der_id][bus] = dV_bus / dQ_der (voltage change per kVAR)
    - M_P[der_id][bus] = dV_bus / dP_der (voltage change per kW)

    These are computed by perturbing each DER's output and measuring the
    resulting voltage changes at all buses.
    """

    def __init__(self, config: SensitivityConfig | None = None):
        """Initialize the sensitivity estimator.

        Args:
            config: Optional sensitivity configuration. Uses defaults if None.
        """
        self.config = config or SensitivityConfig()
        self._cached_result: SensitivityResult | None = None

    def compute_sensitivities(
        self,
        der_container: DERContainer,
        buses: list[str],
        voltages: dict[str, float],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Compute voltage sensitivity matrices M_Q and M_P.

        Args:
            der_container: Container with all DERs.
            buses: List of bus names to compute sensitivities for.
            voltages: Current bus voltages (pu) at operating point.

        Returns:
            (M_Q, M_P) where:
                M_Q[der_id][bus] = dV_bus/dQ_der
                M_P[der_id][bus] = dV_bus/dP_der
        """
        # Check cache validity if enabled
        if self.config.cache_sensitivities and self._cached_result is not None:
            if self._is_cache_valid(voltages):
                return self._cached_result.M_Q, self._cached_result.M_P

        _require_opendss()

        # Get enabled DERs
        enabled_ders = [der for der in der_container.enabled()]
        if not enabled_ders:
            # No DERs - return empty matrices
            return {}, {}

        # Store base operating point
        v0 = voltages.copy()
        q0: dict[str, float] = {}
        p0: dict[str, float] = {}

        for der in enabled_ders:
            q0[der.id] = der.q_kvar
            p0[der.id] = der.p_dispatch_kw

        # Initialize sensitivity matrices
        M_Q: dict[str, dict[str, float]] = {}
        M_P: dict[str, dict[str, float]] = {}

        # Compute sensitivities for each DER
        for der in enabled_ders:
            # Compute Q sensitivity
            M_Q[der.id] = self._perturb_and_measure_q(
                der, buses, v0
            )

            # Restore state after Q perturbation
            self._restore_der_state(der, q0[der.id], p0[der.id])

            # Compute P sensitivity
            M_P[der.id] = self._perturb_and_measure_p(
                der, buses, v0
            )

            # Restore state after P perturbation
            self._restore_der_state(der, q0[der.id], p0[der.id])

        # Cache the result
        if self.config.cache_sensitivities:
            self._cached_result = SensitivityResult(
                M_Q=M_Q,
                M_P=M_P,
                timestamp=time(),
                v0=v0,
                q0=q0,
                p0=p0,
            )

        return M_Q, M_P

    def _perturb_and_measure_q(
        self,
        der: DER,
        buses: list[str],
        v0: dict[str, float],
    ) -> dict[str, float]:
        """Perturb Q and measure voltage changes.

        Args:
            der: DER to perturb.
            buses: Buses to measure voltages at.
            v0: Base voltages before perturbation.

        Returns:
            Dict mapping bus -> dV/dQ (voltage sensitivity to Q).
        """
        try:
            if not dss.Circuit.SetActiveElement(f"PVSystem.{der.id}"):
                # DER not found - zero sensitivity
                return {bus: 0.0 for bus in buses}
        except Exception:
            # OpenDSS error (e.g., no circuit loaded) - zero sensitivity
            return {bus: 0.0 for bus in buses}

        # Store original Q
        q_original = der.q_kvar

        # Compute perturbation magnitude
        q_max = der.q_max_kvar
        delta_q = max(
            self.config.min_perturbation,
            self.config.q_perturbation_pct * q_max,
        )

        # Apply perturbation (absorb reactive power = negative Q)
        q_perturbed = q_original - delta_q

        # Clamp to capability curve
        q_perturbed = max(der.q_min_kvar, min(der.q_max_kvar, q_perturbed))

        # Apply to OpenDSS
        pf = self._q_to_pf(der.p_dispatch_kw, q_perturbed, der.s_kva_rated)
        dss.run_command(f"PVSystem.{der.id}.pf={pf:.6f}")

        # Solve power flow
        dss.Solution.Solve()

        # Measure voltage changes
        sensitivities: dict[str, float] = {}
        actual_delta_q = q_perturbed - q_original

        for bus in buses:
            dss.Circuit.SetActiveBus(bus)
            pu_mags = dss.Bus.puVmagAngle()[::2]
            v_perturbed = float(sum(pu_mags) / len(pu_mags)) if pu_mags else 1.0

            # dV/dQ = (V_perturbed - V0) / delta_Q
            if abs(actual_delta_q) > 1e-6:
                sensitivities[bus] = (v_perturbed - v0.get(bus, 1.0)) / actual_delta_q
            else:
                sensitivities[bus] = 0.0

        return sensitivities

    def _perturb_and_measure_p(
        self,
        der: DER,
        buses: list[str],
        v0: dict[str, float],
    ) -> dict[str, float]:
        """Perturb P (curtailment) and measure voltage changes.

        Args:
            der: DER to perturb.
            buses: Buses to measure voltages at.
            v0: Base voltages before perturbation.

        Returns:
            Dict mapping bus -> dV/dP (voltage sensitivity to P).
        """
        try:
            if not dss.Circuit.SetActiveElement(f"PVSystem.{der.id}"):
                # DER not found - zero sensitivity
                return {bus: 0.0 for bus in buses}
        except Exception:
            # OpenDSS error (e.g., no circuit loaded) - zero sensitivity
            return {bus: 0.0 for bus in buses}

        # Store original P
        p_original = der.p_dispatch_kw

        # Compute perturbation magnitude (curtail = reduce P)
        delta_p = max(
            self.config.min_perturbation,
            self.config.p_perturbation_pct * der.p_avail_kw,
        )

        # Apply perturbation (curtail = reduce P)
        p_perturbed = max(0.0, p_original - delta_p)

        # Apply to OpenDSS
        dss.run_command(f"PVSystem.{der.id}.Pmpp={p_perturbed:.3f}")

        # Solve power flow
        dss.Solution.Solve()

        # Measure voltage changes
        sensitivities: dict[str, float] = {}
        # Sensitivity is with respect to positive curtailment amount, not the
        # signed change in real-power dispatch.
        actual_delta_p = p_original - p_perturbed

        for bus in buses:
            dss.Circuit.SetActiveBus(bus)
            pu_mags = dss.Bus.puVmagAngle()[::2]
            v_perturbed = float(sum(pu_mags) / len(pu_mags)) if pu_mags else 1.0

            # dV/dP = (V_perturbed - V0) / delta_P
            if abs(actual_delta_p) > 1e-6:
                sensitivities[bus] = (v_perturbed - v0.get(bus, 1.0)) / actual_delta_p
            else:
                sensitivities[bus] = 0.0

        return sensitivities

    def _restore_der_state(self, der: DER, q_kvar: float, p_kw: float) -> None:
        """Restore DER to original state.

        Args:
            der: DER to restore.
            q_kvar: Original Q setpoint.
            p_kw: Original P dispatch.
        """
        try:
            if not dss.Circuit.SetActiveElement(f"PVSystem.{der.id}"):
                return
        except Exception:
            # OpenDSS error - skip restoration
            return

        # Restore P
        dss.run_command(f"PVSystem.{der.id}.Pmpp={p_kw:.3f}")

        # Restore Q via power factor
        pf = self._q_to_pf(p_kw, q_kvar, der.s_kva_rated)
        dss.run_command(f"PVSystem.{der.id}.pf={pf:.6f}")

    def _q_to_pf(self, p_kw: float, q_kvar: float, s_kva: float) -> float:
        """Convert P and Q to power factor for OpenDSS.

        Args:
            p_kw: Active power (kW).
            q_kvar: Reactive power (kVAR).
            s_kva: Apparent power rating (kVA).

        Returns:
            Power factor (positive = injection/lagging, negative = absorption/leading).
        """
        s = (p_kw**2 + q_kvar**2) ** 0.5
        if s == 0:
            return 1.0

        pf = p_kw / s
        # Negative pf means absorbing reactive power
        if q_kvar < 0:
            pf = -pf

        return pf

    def _is_cache_valid(self, current_voltages: dict[str, float]) -> bool:
        """Check if cached sensitivities are still valid.

        Cache is valid if:
        1. Cache exists and hasn't expired
        2. Operating point (voltages) hasn't changed significantly

        Args:
            current_voltages: Current bus voltages.

        Returns:
            True if cache is valid, False otherwise.
        """
        if self._cached_result is None:
            return False

        # Check time validity
        age_seconds = time() - self._cached_result.timestamp
        if age_seconds > self.config.cache_valid_minutes * 60:
            return False

        # Check operating point validity
        # Voltages shouldn't have changed more than 0.01 pu
        v0 = self._cached_result.v0
        for bus, v_current in current_voltages.items():
            v_base = v0.get(bus, 1.0)
            if abs(v_current - v_base) > 0.01:
                return False

        return True

    def clear_cache(self) -> None:
        """Clear the cached sensitivity result."""
        self._cached_result = None

    @property
    def cache_hit(self) -> bool:
        """Whether the last computation hit the cache."""
        # This is set during compute_sensitivities
        return self._cached_result is not None
