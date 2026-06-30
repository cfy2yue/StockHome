from __future__ import annotations

import pandas as pd

from scripts.compare_single_stock_kline_thresholds import count_future_keys, threshold_verdict


def test_threshold_verdict_keeps_wider_when_narrow_loses_precision_and_recall() -> None:
    row = pd.Series(
        {
            "h2026_opp_delta_pos_top10": 0.006,
            "h2026_opp_delta_pos_top05": 0.002,
            "h2026_opp_delta_mean_top10": 0.50,
            "h2026_opp_delta_mean_top05": 0.10,
            "h2026_risk_recall_top10": 0.12,
            "h2026_risk_recall_top05": 0.06,
            "prior_opp_delta_pos_top10": 0.003,
            "prior_opp_delta_pos_top05": -0.004,
        }
    )
    assert threshold_verdict(row, "top10", "top05") == "keep_wider_threshold"


def test_threshold_verdict_accepts_narrow_only_when_precision_mean_and_recall_hold() -> None:
    row = pd.Series(
        {
            "h2026_opp_delta_pos_top10": 0.006,
            "h2026_opp_delta_pos_top05": 0.009,
            "h2026_opp_delta_mean_top10": 0.50,
            "h2026_opp_delta_mean_top05": 0.80,
            "h2026_risk_recall_top10": 0.12,
            "h2026_risk_recall_top05": 0.115,
            "prior_opp_delta_pos_top10": 0.003,
            "prior_opp_delta_pos_top05": 0.001,
        }
    )
    assert threshold_verdict(row, "top10", "top05") == "narrow_threshold_candidate"


def test_future_key_counter_is_exact_key_based() -> None:
    payload = {
        "safe_feature": "kline_return_20d is historical",
        "nested": [{"return_20d": 1.2}, {"prior_return_20d": -0.3}],
    }
    assert count_future_keys(payload) == 1
