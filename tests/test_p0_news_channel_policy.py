from __future__ import annotations

import pandas as pd

from scripts.audit_p0_news_channel_policy_v1 import (
    add_news_flags,
    summarize_news_combinations,
    summarize_news_flags,
)


def _base_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "code": "000001",
                "data_missing_flags": "news_semantic_questionnaire缺失",
                "final_agent_reasoning_summary": "财报空窗，同行弱，无硬反证，低仓位试探",
                "user_operation_suggestion": "试探买入",
                "research_grade": "放入观察",
                "return_20d": 10.0,
                "target_cash20": 2.0,
                "target_position": 0.2,
                "target_active": True,
                "buy_like_action": True,
                "risk_action": False,
                "large_loss": False,
                "successful_large_gain_buy": True,
                "successful_buy": True,
                "false_positive_buy": False,
                "large_loss_buy": False,
                "missed_large_gain": False,
                "risk_false_veto_large_gain": False,
                "explicit_or_financial_hard_risk": False,
                "financial_missing_or_no_event": True,
                "peer_weak_or_lagging": True,
                "chip_overhang_or_trapped": False,
            },
            {
                "code": "000002",
                "data_missing_flags": "",
                "final_agent_reasoning_summary": "明确负面新闻，监管处罚，禁止加仓",
                "user_operation_suggestion": "减仓/卖出复核",
                "research_grade": "暂时剔除",
                "return_20d": -6.0,
                "target_cash20": 0.2,
                "target_position": 0.0,
                "target_active": False,
                "buy_like_action": False,
                "risk_action": True,
                "large_loss": True,
                "successful_large_gain_buy": False,
                "successful_buy": False,
                "false_positive_buy": False,
                "large_loss_buy": False,
                "missed_large_gain": False,
                "risk_false_veto_large_gain": False,
                "explicit_or_financial_hard_risk": True,
                "financial_missing_or_no_event": False,
                "peer_weak_or_lagging": False,
                "chip_overhang_or_trapped": False,
            },
            {
                "code": "000003",
                "data_missing_flags": "",
                "final_agent_reasoning_summary": "新闻正面催化，公告利好，但仍需量化和同行确认",
                "user_operation_suggestion": "试探买入",
                "research_grade": "继续深挖",
                "return_20d": 3.0,
                "target_cash20": 1.0,
                "target_position": 0.2,
                "target_active": True,
                "buy_like_action": True,
                "risk_action": False,
                "large_loss": False,
                "successful_large_gain_buy": False,
                "successful_buy": True,
                "false_positive_buy": False,
                "large_loss_buy": False,
                "missed_large_gain": False,
                "risk_false_veto_large_gain": False,
                "explicit_or_financial_hard_risk": False,
                "financial_missing_or_no_event": False,
                "peer_weak_or_lagging": False,
                "chip_overhang_or_trapped": False,
            },
        ]
    )
    return frame


def test_news_missing_is_soft_gap_not_hard_warning() -> None:
    audited = add_news_flags(_base_frame())

    row = audited[audited["code"].eq("000001")].iloc[0]

    assert bool(row["news_missing_no_hard_warning"])
    assert not bool(row["news_hard_warning"])
    assert bool(row["news_soft_gap_with_peer_or_financial"])


def test_news_policy_summary_keeps_missing_as_uncertainty_cap() -> None:
    audited = add_news_flags(_base_frame())
    summary = summarize_news_flags(audited)
    combos = summarize_news_combinations(audited)

    missing = summary[summary["news_flag"].eq("news_missing_no_hard_warning")].iloc[0]
    opportunity = summary[summary["news_flag"].eq("news_opportunity_clean")].iloc[0]
    soft_cluster = combos[combos["news_combo"].eq("missing_news_plus_financial_and_peer")].iloc[0]

    assert missing["policy_hint"] == "uncertainty_cap_not_sell_signal"
    assert opportunity["policy_hint"] == "positive_context_requires_quant_and_peer_confirmation"
    assert soft_cluster["policy_hint"] == "cap_position_but_do_not_auto_zero"
