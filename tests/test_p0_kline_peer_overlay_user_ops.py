from scripts.audit_p0_kline_peer_overlay_user_ops import (
    assert_no_future_fields,
    overlay_position,
    overlay_status,
)


def test_overlay_position_policies() -> None:
    assert overlay_position(0.50, policy="risk_cap_10", opp_active=False, risk_hard=True) == 0.10
    assert overlay_position(0.50, policy="risk_cap_0", opp_active=False, risk_hard=True) == 0.0
    assert overlay_position(0.05, policy="opp_floor_20", opp_active=True, risk_hard=False) == 0.20
    assert overlay_position(0.35, policy="opp_floor_20", opp_active=True, risk_hard=False) == 0.35
    assert overlay_position(0.50, policy="guarded_floor30_cap10", opp_active=True, risk_hard=True) == 0.10
    assert overlay_position(0.05, policy="opp_floor_20_nonrisk", opp_active=True, risk_hard=False, ds_risk_action=True) == 0.05
    assert overlay_position(0.00, policy="opp_floor_30_active_only", opp_active=True, risk_hard=False) == 0.00
    assert overlay_position(0.10, policy="opp_floor_30_active_only", opp_active=True, risk_hard=False) == 0.30


def test_overlay_status_requires_multi_source_block_evidence() -> None:
    row = {
        "policy": "guarded_floor20_cap10",
        "delta_cash_avg": 0.1,
        "changed_rows": 10,
        "raised_negative": 1,
        "lowered_positive": 2,
        "blocks": 3,
        "source_families": 2,
    }
    assert overlay_status(row) == "observe_candidate_needs_block_check"
    weak = dict(row, blocks=1)
    assert overlay_status(weak) == "diagnostic_positive_not_promoted"
    bad = dict(row, delta_cash_avg=-0.1)
    assert overlay_status(bad) == "rejected_overlay"


def test_agent_preview_rejects_future_fields() -> None:
    assert_no_future_fields({"date": "2026-01-01", "opp_score": 0.8})
    try:
        assert_no_future_fields({"return_20d": 1.2})
    except ValueError as exc:
        assert "future/result" in str(exc)
    else:
        raise AssertionError("expected future/result field rejection")
