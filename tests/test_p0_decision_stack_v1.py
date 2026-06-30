from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_decision_stack_v1 import (
    apply_policy,
    assert_no_future_fields,
    build_agent_preview_rows,
    evaluate_policy,
    panel_stability,
    promotion_status,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-09",
                "code": "000001",
                "time_block": "H2026_1",
                "return_20d": 5.0,
                "opp_active": True,
                "opp_strong": True,
                "kline_active": True,
                "risk_review_queue": False,
                "risk_queue_high_hard_counter": False,
                "kline_hard_risk": False,
                "opp_score": 0.08,
                "opp_threshold": 0.02,
                "opp_quantile_in_date": 1.0,
                "kline_opp_score": 0.55,
                "kline_opp_threshold": 0.40,
                "kline_risk_score": 0.20,
                "kline_risk_threshold": 0.60,
                "risk_review_cap_pct": 0.10,
                "confirmation_count": 2,
                "tool_threshold_context": "ok",
            },
            {
                "date": "2026-01-09",
                "code": "000002",
                "time_block": "H2026_1",
                "return_20d": -8.0,
                "opp_active": True,
                "opp_strong": True,
                "kline_active": True,
                "risk_review_queue": True,
                "risk_queue_high_hard_counter": True,
                "kline_hard_risk": False,
                "opp_score": 0.07,
                "opp_threshold": 0.02,
                "opp_quantile_in_date": 0.9,
                "kline_opp_score": 0.50,
                "kline_opp_threshold": 0.40,
                "kline_risk_score": 0.30,
                "kline_risk_threshold": 0.60,
                "risk_review_cap_pct": 0.10,
                "confirmation_count": 2,
                "tool_threshold_context": "risk",
            },
        ]
    )


def test_branch_stack_never_raises_risk_queue_rows() -> None:
    out = apply_policy(_base_frame(), "branch_stack_v1")

    assert list(out["target_position"]) == [0.70, 0.0]
    assert list(out["operation_hint"]) == ["trial_buy_or_add_if_user_confirms", "avoid_or_reduce"]


def test_agent_preview_rejects_future_fields_and_writes_safe_rows() -> None:
    with pytest.raises(ValueError, match="future/result field leaked"):
        assert_no_future_fields({"date": "2026-01-09", "return_20d": 1.0})

    scored = apply_policy(_base_frame(), "opp_kline_confirm_no_raise")
    metrics = evaluate_policy(scored, frequency="weekly_friday", target_block="H2026_1", policy_name="opp_kline_confirm_no_raise")
    rows = build_agent_preview_rows(scored, metrics)

    assert rows[0]["tool_id"] == "p0_decision_stack_v1"
    assert "return_20d" not in rows[0]
    assert rows[0]["operation_hint"] == "trial_buy_or_add_if_user_confirms"


def test_promotion_status_separates_green_yellow_and_reject() -> None:
    green = {
        "policy_name": "branch_stack_v1",
        "h2026_active_pos": 0.62,
        "h2026_active_avg": 1.2,
        "h2026_active_rate": 0.12,
        "prior_active_avg_delta_hit_rate": 0.75,
        "h2026_delta_active_avg_vs_opp": 0.4,
        "h2026_delta_strategy_avg_vs_opp": 0.1,
    }
    yellow = {**green, "h2026_active_pos": 0.52, "prior_active_avg_delta_hit_rate": 0.50}
    reject = {**green, "h2026_active_pos": 0.45, "h2026_active_avg": -0.1}
    baseline = {**green, "policy_name": "hold_all_baseline"}

    assert promotion_status(green) == "green_candidate_for_ds_ablation"
    assert promotion_status(yellow) == "yellow_candidate_needs_fresh_panel"
    assert promotion_status(reject) == "reject_or_reference_only"
    assert promotion_status(baseline) == "baseline_gray_reference"


def test_panel_stability_honors_panel_seed_count() -> None:
    scored = apply_policy(_base_frame(), "opp_kline_confirm_no_raise")
    metrics = evaluate_policy(scored, frequency="weekly_friday", target_block="H2026_1", policy_name="opp_kline_confirm_no_raise")

    rows = panel_stability(scored, metrics, panel_size=1, panel_seeds=5)

    assert len(rows) == 5
    assert {row["panel_seed"] for row in rows} == {0, 1, 2, 3, 4}
