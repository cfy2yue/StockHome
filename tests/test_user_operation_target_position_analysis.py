from __future__ import annotations

import pandas as pd

from scripts.analyze_user_operation_target_position_results import (
    BANK_RETURN_20D_PP,
    build_detail,
    build_pair_summary,
    build_variant_summary,
)


def test_target_position_summary_uses_user_position_not_sim_weight() -> None:
    cards = [
        {
            "agent_policy_version": "p",
            "variant": "full_agent",
            "decision_date": "2026-01-02",
            "code": "1",
            "name": "A",
            "valid_block": "H2026_1",
            "sample_panel_id": "p1",
            "user_operation_suggestion": "试探买入",
            "target_position": 0.25,
            "simulated_action": "增加研究暴露",
            "simulated_weight_change": 0.80,
            "research_grade": "继续深挖",
        }
    ]
    returns = pd.DataFrame({"date": ["2026-01-02"], "code": ["000001"], "return_20d": [10.0]})

    detail = build_detail(cards, returns, panel_label="panel")
    summary = build_variant_summary(detail, [])

    expected = 0.25 * 10.0 + 0.75 * BANK_RETURN_20D_PP
    assert round(float(summary.iloc[0]["target_cash_avg20"]), 6) == round(expected, 6)
    assert float(summary.iloc[0]["sim_cash_avg20"]) > float(summary.iloc[0]["target_cash_avg20"])
    assert summary.iloc[0]["buy_like_cards"] == 1


def test_pair_summary_reports_full_agent_target_position_lift() -> None:
    detail = pd.DataFrame(
        [
            {
                "variant": "full_agent",
                "decision_date": "2026-01-02",
                "code": "000001",
                "valid_block": "H2026_1",
                "sample_panel_id": "p1",
                "user_operation_suggestion": "试探买入",
                "target_position": 0.40,
                "return_20d": 6.0,
            },
            {
                "variant": "no_pps_q017",
                "decision_date": "2026-01-02",
                "code": "000001",
                "valid_block": "H2026_1",
                "sample_panel_id": "p1",
                "user_operation_suggestion": "等待不买",
                "target_position": 0.0,
                "return_20d": 6.0,
            },
        ]
    )

    pair = build_pair_summary(detail, controls=["no_pps_q017"])

    assert pair.iloc[0]["paired_rows"] == 1
    assert pair.iloc[0]["changed_rows"] == 1
    assert pair.iloc[0]["raised_positive"] == 1
    assert pair.iloc[0]["mean_delta_target_cash20"] > 0
