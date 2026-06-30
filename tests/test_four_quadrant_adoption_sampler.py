from __future__ import annotations

import pandas as pd

from scripts.build_four_quadrant_adoption_sampler import (
    financial_event_asof_mask,
    news_availability_group,
    peer_confirmation_group,
    summarize_quadrants,
)


def test_four_quadrant_channel_classifiers() -> None:
    frame = pd.DataFrame(
        [
            {
                "financial_report_join_status": "event_window_matched",
                "financial_report_event_count": 2,
                "financial_report_available_at": "2026-01-02 00:00:00",
                "tushare_industry_positive_breadth_20d": 0.6,
                "tushare_industry_relative_return_20d": 1.0,
                "tushare_area_positive_breadth_20d": 0.4,
                "tushare_area_relative_return_20d": -1.0,
                "news_count_30d": 3,
                "news_missing_rate": 0.0,
                "news_evidence_quality": 0.5,
            },
            {
                "financial_report_join_status": "code_not_in_feature_table",
                "financial_report_event_count": 0,
                "financial_report_available_at": "",
                "tushare_industry_positive_breadth_20d": 0.4,
                "tushare_industry_relative_return_20d": -2.0,
                "tushare_area_positive_breadth_20d": 0.3,
                "tushare_area_relative_return_20d": -1.0,
                "news_count_30d": 0,
                "news_missing_rate": 1.0,
                "news_evidence_quality": 0.0,
            },
        ]
    )

    assert financial_event_asof_mask(frame).tolist() == [True, False]
    assert peer_confirmation_group(frame).tolist() == ["peer_positive", "peer_negative_or_weak"]
    assert news_availability_group(frame).tolist() == ["news_available", "news_missing"]


def test_quadrant_summary_uses_candidate_rows_for_candidate_pool() -> None:
    detail = pd.DataFrame(
        [
            {
                "record_type": "candidate_pool",
                "valid_block": "H2026_1",
                "quadrant_id": "financial_asof__peer_positive__news_available",
                "candidate_rows": 0,
                "unique_stocks": 0,
                "unique_dates": 0,
                "_dual_mode_score": None,
                "financial_event_asof": True,
                "peer_positive": True,
                "news_available": True,
            },
            {
                "record_type": "selected_sample",
                "valid_block": "H2026_1",
                "quadrant_id": "financial_asof__peer_positive__news_missing",
                "date": "2026-01-02",
                "code": "000001",
                "_dual_mode_score": 1.0,
                "financial_event_asof": True,
                "peer_positive": True,
                "news_available": False,
            },
        ]
    )

    summary = summarize_quadrants(detail)

    empty_pool = summary[summary["record_type"].eq("candidate_pool")].iloc[0]
    selected = summary[summary["record_type"].eq("selected_sample")].iloc[0]
    assert empty_pool["rows"] == 0
    assert empty_pool["unique_stocks"] == 0
    assert selected["rows"] == 1
