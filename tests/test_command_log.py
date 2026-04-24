"""Tests for DER command audit logging."""

from datetime import datetime
from math import isclose

import pandas as pd

from src.control.command_log import CommandLogger, summarize_command_log
from src.control.der_interface import SetpointApplyResult
from src.control.der_models import DER, DERContainer


def test_command_logger_records_rich_audit_fields(tmp_path):
    """CommandLogger should preserve simulation context and before/after state."""
    der = DER(
        id="pv_001_675",
        bus="675",
        phases=1,
        p_kw_rated=100.0,
        s_kva_rated=120.0,
        p_avail_kw=80.0,
        p_dispatch_kw=80.0,
        q_kvar=0.0,
    )
    container = DERContainer([der])
    result = SetpointApplyResult(status="applied", q_status="applied")

    logger = CommandLogger(tmp_path / "commands.csv")
    logger.log_batch(
        container=container,
        q_commands={der.id: -12.0},
        p_commands=None,
        results={der.id: result},
        timestamp=datetime(2026, 4, 24, 12, 0),
        voltages_before={"675": 1.061},
        voltages_after={"675": 1.048},
        step=144,
        time_min=720,
        time_h=12.0,
        controller_mode="heuristic",
        controller_status="active",
        states_before={
            der.id: {
                "p_avail_kw": 80.0,
                "p_dispatch_kw": 80.0,
                "q_kvar": 0.0,
            }
        },
        states_after={
            der.id: {
                "p_dispatch_kw": 79.5,
                "q_kvar": -11.8,
            }
        },
    )

    df = pd.read_csv(logger.save())
    row = df.iloc[0]

    assert row["step"] == 144
    assert row["controller_mode"] == "heuristic"
    assert row["q_status"] == "applied"
    assert row["p_status"] == "not_requested"
    assert row["p_dispatch_before_kw"] == 80.0
    assert row["p_dispatch_after_kw"] == 79.5
    assert row["q_after_kvar"] == -11.8
    assert row["v_delta_pu"] < 0


def test_command_logger_keeps_unknown_der_failures(tmp_path):
    """Unknown DER commands should remain visible in the audit log."""
    logger = CommandLogger(tmp_path / "commands.csv")
    logger.log_batch(
        container=DERContainer([]),
        q_commands={"missing_der": 5.0},
        p_commands=None,
        results={
            "missing_der": SetpointApplyResult(
                status="failed",
                q_status="failed",
                message="DER not found in container",
            )
        },
        timestamp=datetime(2026, 4, 24, 12, 0),
    )

    df = pd.read_csv(logger.save())

    assert len(df) == 1
    assert df.loc[0, "der_id"] == "missing_der"
    assert df.loc[0, "status"] == "failed"
    assert df.loc[0, "reason"] == "DER not found in container"


def test_summarize_command_log():
    """Command summaries should expose showcase-level control metrics."""
    df = pd.DataFrame(
        {
            "der_id": ["pv_1", "pv_1", "pv_2"],
            "status": ["applied", "partial", "out_of_range"],
            "q_status": ["applied", "applied", "out_of_range"],
            "p_status": ["not_requested", "failed", "not_requested"],
            "q_commanded_kvar": [-10.0, -20.0, -30.0],
            "p_curtail_commanded_kw": [0.0, 5.0, 0.0],
            "v_delta_pu": [-0.01, -0.02, 0.0],
        }
    )

    summary = summarize_command_log(df, time_step_minutes=5)

    assert summary["commands_sent"] == 3
    assert summary["commands_applied"] == 1
    assert summary["commands_partial"] == 1
    assert summary["commands_failed"] == 1
    assert summary["commands_out_of_range"] == 1
    assert summary["unique_ders_controlled"] == 2
    assert summary["top_controlled_der"] == "pv_1"
    assert isclose(summary["total_curtailed_energy_kwh"], 5.0 * 5 / 60)
