from __future__ import annotations

import pandas as pd

from scripts.run_local_kline_news_fin_peer_interactions import add_interaction_flags, evaluate_rules


def test_interaction_flags_identify_gaps_and_clean_context() -> None:
    frame = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "date": pd.to_datetime(["2025-01-03", "2025-01-03"]),
            "time_block": ["H2025_1", "H2025_1"],
            "return_20d": [2.0, -3.0],
            "kline_return_20d": [-12.0, -12.0],
            "peer_kline_group_positive_breadth_20d": [0.55, 0.20],
            "news_missing_rate": [0.2, 1.0],
            "news_warning_score": [0.1, 0.1],
            "financial_report_missing_rate": [0.0, 1.0],
            "financial_quality_risk_score": [0.2, 0.2],
            "triggered_skills": ["CORE_TREND_001", ""],
        }
    )

    out = add_interaction_flags(frame)

    assert out.loc[0, "kline_20d_pullback_flag"]
    assert out.loc[0, "clean_cross_channel_context_flag"]
    assert not out.loc[0, "major_confirmation_gap_flag"]
    assert out.loc[1, "major_confirmation_gap_flag"]
    assert out.loc[1, "peer_weak_flag"]


def test_evaluate_rules_keeps_baseline_and_kline_rule() -> None:
    rows = []
    for idx, block in enumerate(["H2023_1", "H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]):
        for code in ["000001", "000002"]:
            rows.append(
                {
                    "code": code,
                    "date": pd.Timestamp("2025-01-03") + pd.Timedelta(days=idx),
                    "time_block": block,
                    "return_20d": 3.0 if code == "000001" else -1.0,
                    "kline_return_20d": -12.0,
                    "kline_return_60d": -5.0,
                    "kline_volatility_ratio_20_60": 1.0,
                    "kline_trend_consistency_20d": 0.5,
                    "peer_kline_group_positive_breadth_20d": 0.6,
                    "news_missing_rate": 0.2,
                    "news_warning_score": 0.1,
                    "financial_report_missing_rate": 0.0,
                    "financial_quality_risk_score": 0.2,
                    "triggered_skills": "CORE_TREND_001",
                }
            )
    frame = add_interaction_flags(pd.DataFrame(rows))
    result, block_result = evaluate_rules(frame)

    assert "baseline_all" in set(result["rule_id"])
    assert "kline_20d_pullback_all" in set(result["rule_id"])
    assert set(block_result["time_block"]).issuperset({"H2025_2", "H2026_1"})
