from __future__ import annotations

import pandas as pd

from scripts.analyze_news_financial_interactions import (
    add_interaction_columns,
    assert_no_future_plan_columns,
    build_rule_tables,
    build_sample_plan,
    load_rows,
)


def test_news_financial_interaction_plan_excludes_future_columns(tmp_path):
    path = tmp_path / "joined.csv"
    rows = []
    for index in range(80):
        rows.append(
            {
                "date": "2025-04-30",
                "code": f"{index + 1:06d}",
                "name": f"S{index}",
                "gt_status": "evaluated",
                "return_20d": 2.0 if index % 2 else -6.0,
                "prior_return_20d": 25.0 if index % 3 == 0 else -8.0,
                "rsi14": 76.0 if index % 3 == 0 else 45.0,
                "relative_strength_rank": 0.92 if index % 3 == 0 else 0.4,
                "news_count_30d": 3,
                "event_count": 3,
                "news_missing_rate": 0.0,
                "news_warning_score": 0.4 if index % 2 == 0 else 0.0,
                "news_opportunity_score": 0.5 if index % 2 == 1 else 0.0,
                "policy_background_score": 0.3,
                "official_confirmation_score": 0.9,
                "announcement_materiality_score": 0.7,
                "news_timestamp_quality": 0.7,
                "news_evidence_quality": 0.8,
                "news_negative_materiality_30d": 0.8 if index % 2 == 0 else 0.0,
                "news_positive_materiality_30d": 0.8 if index % 2 == 1 else 0.0,
                "news_conflict_intensity_30d": 0.0,
                "peer_group_news_count_avg": 3.0,
                "peer_relative_to_group_20d": -3.0 if index % 2 == 0 else 3.0,
                "peer_group_positive_breadth_20d": 0.4 if index % 2 == 0 else 0.7,
                "financial_report_event_count": 1,
                "financial_report_join_status": "event_window_matched",
                "financial_quality_risk_score": 0.7 if index % 2 == 0 else 0.1,
                "financial_surprise_score": -0.2 if index % 2 == 0 else 0.4,
                "financial_disclosure_quality_score": 0.8,
                "financial_report_event_types": "quarterly_metrics",
                "triggered_skills": "TEST-SKILL",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)

    frame = add_interaction_columns(load_rows(path))
    metrics, _ = build_rule_tables(frame)
    plan = build_sample_plan(frame, metrics, max_samples_per_rule=3, max_rows=12)

    assert not plan.empty
    assert "return_20d" not in plan.columns
    assert "gt_status" not in plan.columns
    assert {"candidate_rule", "date", "code", "reason_to_test"}.issubset(plan.columns)
    assert_no_future_plan_columns(plan)


def test_assert_no_future_plan_columns_rejects_result_fields():
    plan = pd.DataFrame([{"date": "2025-01-01", "code": "000001", "return_20d": 3.0}])

    try:
        assert_no_future_plan_columns(plan)
    except ValueError as exc:
        assert "future/result fields" in str(exc)
    else:
        raise AssertionError("expected future/result field rejection")
