from __future__ import annotations

import pandas as pd

from src.backtest.news_alerts import add_news_alert_features


def test_news_alert_features_split_warning_and_opportunity():
    rows = []
    for idx in range(10):
        rows.append(
            {
                "date": "2026-01-02",
                "code": f"{idx:06d}",
                "news_count_30d": 1,
                "news_risk_event_score_30d": 0,
                "news_conflict_intensity_30d": 0,
                "news_recency_weighted_materiality_30d": 0,
                "peer_group_news_risk_avg": 0,
                "peer_group_news_count_avg": 0,
                "peer_group_news_opportunity_avg": 0,
                "news_opportunity_event_score_30d": 0,
                "news_evidence_quality_score_30d": 0,
                "news_company_count_30d": 0,
                "news_top_event_materiality_30d": 0,
                "news_industry_policy_count_30d": 0,
                "news_price_policy_score_30d": 0,
                "news_macro_market_score_30d": 0,
            }
        )
    rows[0]["news_count_30d"] = 12
    rows[0]["news_opportunity_event_score_30d"] = 6
    rows[1]["news_risk_event_score_30d"] = 8
    rows[1]["news_conflict_intensity_30d"] = 3

    out = add_news_alert_features(pd.DataFrame(rows))

    opportunity = out[out["code"] == "000000"].iloc[0]
    warning = out[out["code"] == "000001"].iloc[0]

    assert opportunity["news_attention_spike_score_30d"] > 0
    assert opportunity["news_relative_attention_score_30d"] > 0
    assert opportunity["news_opportunity_alert_score_30d"] > 0
    assert opportunity["news_alert_label"] == "机会提醒"
    assert warning["news_warning_score_30d"] >= 1
    assert warning["news_alert_label"] == "风险预警"
