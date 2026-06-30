from __future__ import annotations

import pytest

import pandas as pd

import scripts.audit_nonquant_positive_confirmation as audit_module
from scripts.audit_nonquant_positive_confirmation import (
    DEFAULT_JOINED_GT_CACHE_PATH,
    add_signal_flags,
    assert_no_future_fields,
    build_agent_rule_previews,
    evaluate_rules,
    aggregate_rule_metrics,
    load_candidate_frame,
    select_rule_rows,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "name": "A",
                "time_block": "H2026_1",
                "return_20d": 5.0,
                "pool_excess_20d": 2.0,
                "news_missing_rate": 0.0,
                "news_opportunity_score": 0.33,
                "news_warning_score": 0.0,
                "news_evidence_quality": 0.865,
                "official_confirmation_score": 1.0,
                "announcement_materiality_score": 0.7,
                "financial_report_join_status": "event_window_matched",
                "financial_report_event_count": 1,
                "financial_report_materiality_score": 0.8,
                "financial_quality_risk_score": 0.0,
                "financial_surprise_score": 0.2,
                "financial_disclosure_quality_score": 0.85,
                "tushare_industry_relative_return_20d": 2.0,
                "tushare_industry_positive_breadth_20d": 0.7,
                "tushare_area_relative_return_20d": 1.0,
                "tushare_area_positive_breadth_20d": 0.6,
                "triggered_skills": "PPS-Q-017",
                "book_score": 6.5,
                "counter_score": 5.5,
                "completeness_score": 8.0,
            },
            {
                "date": "2026-01-02",
                "code": "000002",
                "name": "B",
                "time_block": "H2026_1",
                "return_20d": -4.0,
                "pool_excess_20d": -2.0,
                "news_missing_rate": 1.0,
                "news_opportunity_score": 0.0,
                "news_warning_score": 0.66,
                "news_evidence_quality": 0.0,
                "official_confirmation_score": 0.0,
                "announcement_materiality_score": 0.0,
                "financial_report_join_status": "no_event_in_window",
                "financial_report_event_count": 0,
                "financial_quality_risk_score": 0.0,
                "tushare_industry_relative_return_20d": -2.0,
                "tushare_industry_positive_breadth_20d": 0.3,
                "tushare_area_relative_return_20d": -1.0,
                "tushare_area_positive_breadth_20d": 0.4,
                "triggered_skills": "",
                "book_score": 4.0,
                "counter_score": 8.0,
                "completeness_score": 8.0,
            },
        ]
    )


def test_signal_flags_capture_cross_channel_confirmation() -> None:
    flagged = add_signal_flags(_frame())

    first = flagged.iloc[0]
    second = flagged.iloc[1]
    assert first["news_high_quality_positive"]
    assert first["financial_event_quality_low_risk"]
    assert first["industry_peer_support"]
    assert first["bookskill_support_low_counter"]
    assert first["nonquant_confirmation_count"] == 4
    assert not second["news_high_quality_positive"]
    assert second["bookskill_counter_high"]


def test_select_rule_rows_uses_flags_and_min_count() -> None:
    flagged = add_signal_flags(_frame())
    selected = select_rule_rows(flagged, {"rule_id": "x", "required_flags": ["news_high_quality_positive"]})
    selected_min = select_rule_rows(flagged, {"rule_id": "y", "min_confirmation_count": 3})

    assert selected["code"].tolist() == ["000001"]
    assert selected_min["code"].tolist() == ["000001"]


def test_agent_rule_previews_do_not_contain_future_fields() -> None:
    flagged = add_signal_flags(_frame())
    metrics = evaluate_rules(flagged)
    aggregate = aggregate_rule_metrics(metrics)
    previews = build_agent_rule_previews(aggregate)

    assert previews
    text = str(previews)
    assert "return_20d" not in text
    assert "pool_excess_20d" not in text
    assert "gt_status" not in text
    for preview in previews:
        assert_no_future_fields(preview)


def test_assert_no_future_fields_rejects_future_key() -> None:
    with pytest.raises(ValueError):
        assert_no_future_fields({"return_20d": 1.0})


def test_default_candidate_loader_rebuilds_joined_cache_through_ground_truth_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    source_frame = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "1",
                "name": "A",
                "gt_status": "evaluated",
                "return_20d": 1.0,
            }
        ]
    )

    def fake_load_ground_truth(paths, **kwargs):
        calls["paths"] = list(paths)
        calls["kwargs"] = kwargs
        return source_frame.copy()

    def fake_ranker_details(frame, **kwargs):
        return {
            "score": pd.Series([1.0], index=frame.index),
            "score_quantile": pd.Series([1.0], index=frame.index),
        }

    monkeypatch.setattr(audit_module, "load_ground_truth", fake_load_ground_truth)
    monkeypatch.setattr(audit_module, "_portfolio_ranker_details", fake_ranker_details)

    loaded = load_candidate_frame(
        DEFAULT_JOINED_GT_CACHE_PATH,
        ground_truth_sources=[audit_module.ROOT / "dummy_ground_truth.csv"],
        high_ranker_quantile=0.8,
    )

    assert calls["paths"]
    assert calls["kwargs"]["kline_features_path"] == audit_module.DEFAULT_KLINE_FEATURES_PATH
    assert loaded["code"].tolist() == ["000001"]
