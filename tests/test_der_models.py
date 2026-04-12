"""Unit tests for DER data model.

Tests the DER dataclass and DERContainer without requiring OpenDSS.
"""

import pytest

from src.control.der_models import DER, DERContainer


# ---------------------------------------------------------------------------
# DER dataclass tests
# ---------------------------------------------------------------------------


def test_der_creation():
    """DER should be creatable with required parameters."""
    der = DER(
        id="pv_001_675",
        bus="675",
        phases=1,
        p_kw_rated=100.0,
        s_kva_rated=120.0,
    )
    assert der.id == "pv_001_675"
    assert der.bus == "675"
    assert der.phases == 1
    assert der.p_kw_rated == 100.0
    assert der.s_kva_rated == 120.0


def test_der_default_values():
    """DER should have sensible defaults for optional parameters."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    assert der.control_enabled is True
    assert der.p_avail_kw == 0.0
    assert der.p_dispatch_kw == 0.0
    assert der.q_kvar == 0.0
    assert der.v_local_pu == 1.0


def test_q_max_at_zero_dispatch():
    """At zero active power, Qmax should equal the rated kVA."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    der.p_dispatch_kw = 0.0
    assert der.q_max_kvar == 120.0


def test_q_max_at_full_active_power():
    """At full active power (P=S), Qmax should be zero."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    der.p_dispatch_kw = 120.0  # At kVA rating
    assert der.q_max_kvar == 0.0


def test_q_max_at_partial_dispatch():
    """Qmax should follow S² = P² + Q² curve."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    der.p_dispatch_kw = 72.0  # P = 0.6 * S
    # Q = sqrt(120² - 72²) = sqrt(14400 - 5184) = sqrt(9216) = 96
    assert abs(der.q_max_kvar - 96.0) < 0.01


def test_q_min_is_negative_q_max():
    """Qmin should be the negative of Qmax (symmetric absorption)."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    der.p_dispatch_kw = 50.0
    assert der.q_min_kvar == -der.q_max_kvar


def test_can_provide_q_within_limits():
    """can_provide_q should return True for Q within [q_min, q_max]."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    der.p_dispatch_kw = 50.0
    assert der.can_provide_q(0.0) is True
    assert der.can_provide_q(der.q_max_kvar) is True
    assert der.can_provide_q(der.q_min_kvar) is True


def test_can_provide_q_outside_limits():
    """can_provide_q should return False for Q outside [q_min, q_max]."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    der.p_dispatch_kw = 50.0
    assert der.can_provide_q(der.q_max_kvar + 1.0) is False
    assert der.can_provide_q(der.q_min_kvar - 1.0) is False


def test_control_enabled_false_affects_operations():
    """DER with control_enabled=False should be filterable."""
    der_enabled = DER(
        id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0, control_enabled=True
    )
    der_disabled = DER(
        id="pv_002", bus="676", phases=1, p_kw_rated=100.0, s_kva_rated=120.0, control_enabled=False
    )
    assert der_enabled.control_enabled is True
    assert der_disabled.control_enabled is False


# ---------------------------------------------------------------------------
# DERContainer tests
# ---------------------------------------------------------------------------


def test_container_creation():
    """DERContainer should be creatable with an empty list."""
    container = DERContainer()
    assert len(container) == 0


def test_container_with_ders():
    """DERContainer should store and count DERs."""
    ders = [
        DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0),
        DER(id="pv_002", bus="676", phases=1, p_kw_rated=80.0, s_kva_rated=100.0),
    ]
    container = DERContainer(ders)
    assert len(container) == 2


def test_container_getitem_by_id():
    """Container should allow accessing DERs by ID."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    container = DERContainer([der])
    retrieved = container["pv_001"]
    assert retrieved is der


def test_container_getitem_raises_keyerror():
    """Accessing non-existent DER ID should raise KeyError."""
    container = DERContainer()
    with pytest.raises(KeyError, match="DER not found"):
        _ = container["nonexistent"]


def test_container_contains():
    """Container should support 'in' operator for ID lookup."""
    der = DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0)
    container = DERContainer([der])
    assert "pv_001" in container
    assert "pv_002" not in container


def test_container_by_bus():
    """Container should filter DERs by bus."""
    ders = [
        DER(id="pv_001", bus="675", phases=1, p_kw_rated=100.0, s_kva_rated=120.0),
        DER(id="pv_002", bus="675", phases=1, p_kw_rated=80.0, s_kva_rated=100.0),
        DER(id="pv_003", bus="680", phases=1, p_kw_rated=100.0, s_kva_rated=120.0),
    ]
    container = DERContainer(ders)
    bus_675_ders = container.by_bus("675")
    assert len(bus_675_ders) == 2
    assert all(d.bus == "675" for d in bus_675_ders)


def test_container_enabled():
    """Container should filter to only control-enabled DERs."""
    ders = [
        DER(
            id="pv_001",
            bus="675",
            phases=1,
            p_kw_rated=100.0,
            s_kva_rated=120.0,
            control_enabled=True,
        ),
        DER(
            id="pv_002",
            bus="676",
            phases=1,
            p_kw_rated=100.0,
            s_kva_rated=120.0,
            control_enabled=False,
        ),
        DER(
            id="pv_003",
            bus="680",
            phases=1,
            p_kw_rated=100.0,
            s_kva_rated=120.0,
            control_enabled=True,
        ),
    ]
    container = DERContainer(ders)
    enabled = container.enabled()
    assert len(enabled) == 2
    assert all(d.control_enabled for d in enabled)
    assert all(d.id in ["pv_001", "pv_003"] for d in enabled)
