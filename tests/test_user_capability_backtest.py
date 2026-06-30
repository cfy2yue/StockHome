from __future__ import annotations

import pandas as pd

from scripts.run_user_capability_backtest import (
    aggregate_user_summary,
    operation_decision,
    select_candidate_shortlist,
)


def test_operation_decision_outputs_clear_buy_or_wait_actions() -> None:
    buy = operation_decision(
        {
            "rev_chip_score_quantile": 0.95,
            "agent_policy_score": 0.75,
            "news_warning_score": 0.1,
            "news_opportunity_score": 0.5,
            "financial_quality_risk_score": 0.1,
            "tushare_industry_positive_breadth_20d": 0.7,
            "tushare_industry_relative_return_20d": 1.0,
            "lower_support": 0.25,
        },
        previous_position=0.0,
    )
    assert buy["operation_action"] == "买入"
    assert buy["target_position"] >= 0.6

    wait = operation_decision(
        {
            "rev_chip_score_quantile": 0.6,
            "agent_policy_score": 0.3,
            "news_warning_score": 0.8,
            "financial_quality_risk_score": 0.8,
            "tushare_industry_positive_breadth_20d": 0.2,
            "tushare_industry_relative_return_20d": -6.0,
        },
        previous_position=0.0,
    )
    assert wait["operation_action"] == "等待不买"
    assert wait["target_position"] == 0.0

    sell = operation_decision(
        {
            "rev_chip_score_quantile": 0.5,
            "agent_policy_score": 0.2,
            "news_warning_score": 0.8,
            "financial_quality_risk_score": 0.8,
        },
        previous_position=0.7,
    )
    assert sell["operation_action"] == "卖出/不买"


def test_select_candidate_shortlist_is_industry_diversified() -> None:
    rows = []
    for idx in range(20):
        rows.append(
            {
                "code": f"{idx:06d}",
                "industry_for_selection": "A" if idx < 10 else f"I{idx}",
                "candidate_selector_score": 100 - idx,
                "return_20d": 1.0,
            }
        )
    selected = select_candidate_shortlist(pd.DataFrame(rows), pool_size=200)

    assert len(selected) == 12
    assert selected["industry_for_selection"].value_counts().max() <= 2
    assert selected["selection_rank"].tolist() == list(range(1, 13))


def test_aggregate_user_summary_reports_panel_std() -> None:
    frame = pd.DataFrame(
        [
            {
                "task_mode": "single_stock_watch",
                "panel_id": "p1",
                "period": "H2026",
                "decision_frequency": "weekly_friday",
                "decision_count": 10,
                "unique_stocks": 5,
                "avg_target_position": 0.2,
                "active_decision_rate": 0.4,
                "active_decision_count": 4,
                "active_strategy_positive_20d_rate": 0.5,
                "active_strategy_avg_return_20d": 0.8,
                "active_hold_positive_20d_rate": 0.4,
                "active_hold_avg_return_20d": 0.1,
                "active_excess_avg_return_vs_hold": 0.7,
                "strategy_positive_20d_rate": 0.6,
                "strategy_avg_return_20d": 1.0,
                "strategy_std_return_20d": 2.0,
                "strategy_loss_gt5_rate": 0.1,
                "hold_positive_20d_rate": 0.5,
                "hold_avg_return_20d": 0.0,
                "excess_avg_return_vs_hold": 1.0,
                "capital_100k_mean_after_20d": 101000,
            },
            {
                "task_mode": "single_stock_watch",
                "panel_id": "p2",
                "period": "H2026",
                "decision_frequency": "weekly_friday",
                "decision_count": 20,
                "unique_stocks": 7,
                "avg_target_position": 0.4,
                "active_decision_rate": 0.8,
                "active_decision_count": 16,
                "active_strategy_positive_20d_rate": 0.75,
                "active_strategy_avg_return_20d": 2.2,
                "active_hold_positive_20d_rate": 0.55,
                "active_hold_avg_return_20d": 0.5,
                "active_excess_avg_return_vs_hold": 1.7,
                "strategy_positive_20d_rate": 0.8,
                "strategy_avg_return_20d": 3.0,
                "strategy_std_return_20d": 4.0,
                "strategy_loss_gt5_rate": 0.2,
                "hold_positive_20d_rate": 0.6,
                "hold_avg_return_20d": 1.0,
                "excess_avg_return_vs_hold": 2.0,
                "capital_100k_mean_after_20d": 103000,
            },
        ]
    )

    out = aggregate_user_summary(frame, ["task_mode", "period", "decision_frequency"])

    assert out.iloc[0]["panels"] == 2
    assert "strategy_positive_20d_rate_std" in out.columns
    assert out.iloc[0]["strategy_avg_return_20d"] == 2.0
