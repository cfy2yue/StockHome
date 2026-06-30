from __future__ import annotations

import pandas as pd

from scripts.audit_p0_user_operation_error_patterns import (
    add_error_and_channel_flags,
    summarize_flags,
)


def test_soft_gap_without_hard_risk_is_not_hard_counter() -> None:
    frame = pd.DataFrame(
        [
            {
                "data_missing_flags": "news_missing_rate=1; financial_no_event_in_window",
                "final_agent_reasoning_summary": "同行弱但无硬反证，保留小仓试探。",
                "user_operation_suggestion": "试探买入/持有",
                "research_grade": "放入观察",
                "return_20d": 8.0,
                "target_position": 0.25,
                "target_cash20": 2.0,
                "buy_like_action": True,
                "risk_action": False,
                "target_active": True,
            }
        ]
    )

    audited = add_error_and_channel_flags(frame, min_large_gain=5.0, min_large_loss=-5.0)

    assert bool(audited.loc[0, "news_missing_or_empty"])
    assert bool(audited.loc[0, "financial_missing_or_no_event"])
    assert bool(audited.loc[0, "peer_weak_or_lagging"])
    assert bool(audited.loc[0, "soft_gap_without_hard_risk"])
    assert not bool(audited.loc[0, "explicit_or_financial_hard_risk"])
    assert audited.loc[0, "error_type"] == "successful_large_gain_buy"


def test_hard_risk_flag_keeps_confirmation_policy() -> None:
    frame = pd.DataFrame(
        [
            {
                "data_missing_flags": "",
                "final_agent_reasoning_summary": "新闻风险高，财报质量风险，建议减仓/卖出复核。",
                "user_operation_suggestion": "减仓/卖出复核",
                "research_grade": "放入观察",
                "return_20d": 10.0,
                "target_position": 0.0,
                "target_cash20": 0.238095,
                "buy_like_action": False,
                "risk_action": True,
                "target_active": False,
            }
        ]
    )

    audited = add_error_and_channel_flags(frame, min_large_gain=5.0, min_large_loss=-5.0)
    flags = summarize_flags(audited)

    assert bool(audited.loc[0, "explicit_or_financial_hard_risk"])
    assert audited.loc[0, "error_type"] == "risk_false_veto_large_gain"
    hard = flags[flags["flag"].eq("explicit_or_financial_hard_risk")].iloc[0]
    assert hard["risk_false_veto_large_gain_rows"] == 1
    assert hard["policy_hint"] == "second_check_required_no_blind_zero_no_raise"
