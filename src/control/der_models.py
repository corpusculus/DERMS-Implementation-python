"""DER data model for the DERMS control layer.

Defines the core data structures representing Distributed Energy Resources
with their capabilities, state, and constraints.
"""

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass
class DER:
    """Controllable Distributed Energy Resource.

    Follows IEEE 1547 convention: positive Q = injection (lagging),
    negative Q = absorption (leading).
    """

    id: str
    bus: str
    phases: int
    p_kw_rated: float
    s_kva_rated: float
    control_enabled: bool = True

    # Runtime state (populated from OpenDSS)
    p_avail_kw: float = 0.0
    p_dispatch_kw: float = 0.0
    q_kvar: float = 0.0
    v_local_pu: float = 1.0

    @property
    def q_max_kvar(self) -> float:
        """Maximum reactive power available at current dispatch level."""
        headroom = max(0.0, self.s_kva_rated**2 - self.p_dispatch_kw**2)
        return sqrt(headroom)

    @property
    def q_min_kvar(self) -> float:
        """Minimum reactive power (absorption limit)."""
        return -self.q_max_kvar

    def can_provide_q(self, q_kvar: float) -> bool:
        """Check if Q request is within inverter capability.

        Args:
            q_kvar: Requested reactive power in kVAR.

        Returns:
            True if the request is within [q_min, q_max], False otherwise.
        """
        return self.q_min_kvar <= q_kvar <= self.q_max_kvar


@dataclass
class DERContainer:
    """Container for all DERs in a simulation.

    Provides convenient lookup and filtering methods for DER collections.
    """

    ders: list[DER] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of DERs in the container."""
        return len(self.ders)

    def __getitem__(self, der_id: str) -> DER:
        """Get a DER by its ID.

        Args:
            der_id: The unique identifier of the DER.

        Returns:
            The DER with the given ID.

        Raises:
            KeyError: If no DER with the given ID exists.
        """
        for der in self.ders:
            if der.id == der_id:
                return der
        raise KeyError(f"DER not found: {der_id}")

    def __contains__(self, der_id: str) -> bool:
        """Check if a DER ID exists in the container.

        Args:
            der_id: The unique identifier to check.

        Returns:
            True if a DER with the given ID exists, False otherwise.
        """
        return any(der.id == der_id for der in self.ders)

    def by_bus(self, bus: str) -> list[DER]:
        """Get all DERs connected to a specific bus.

        Args:
            bus: The bus name to filter by.

        Returns:
            List of DERs connected to the specified bus.
        """
        return [der for der in self.ders if der.bus == bus]

    def enabled(self) -> list[DER]:
        """Get only control-enabled DERs.

        Returns:
            List of DERs with control_enabled=True.
        """
        return [der for der in self.ders if der.control_enabled]

    def enabled_ids(self) -> list[str]:
        """Get IDs of enabled DERs.

        Returns:
            List of DER IDs with control_enabled=True.
        """
        return [der.id for der in self.ders if der.control_enabled]
