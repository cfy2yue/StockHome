from __future__ import annotations

import pandas as pd

from scripts.analyze_news_financial_interactions import PLAN_COLUMNS, add_interaction_columns, assert_no_future_plan_columns
from scripts.build_balanced_news_ablation_sample_plan import (
    add_balanced_columns,
    build_balanced_sample_plan,
    build_micro48_plan,
    build_rule_metrics,
)


def _base_row(index: int, date: str) -> dict[str, object]:
    return {
        "date": date,
        "code": f"{index + 1:06d}",
        "name": f"S{index}",
        "return_20d": 2.0 if index % 2 else -3.0,
        "prior_return_20d": 8.0,
        "rsi14": 55.0,
        "relative_strength_rank": 0.75,
        "news_count_30d": 1,
        "event_count": 2,
        "news_missing_rate": 0.0,
        "news_warning_score": 0.0,
        "news_opportunity_score": 0.1,
        "policy_background_score": 0.2,
        "official_confirmation_score": 0.9,
        "announcement_materiality_score": 0.7,
        "news_timestamp_quality": 0.8,
        "news_evidence_quality": 0.8,
        "news_negative_materiality_30d": 0.0,
        "news_positive_materiality_30d": 0.0,
        "news_conflict_intensity_30d": 0.0,
        "peer_group_news_count_avg": 1.0,
        "peer_relative_to_group_20d": 1.0,
        "peer_group_positive_breadth_20d": 0.6,
        "financial_report_event_count": 1,
        "financial_report_join_status": "event_window_matched",
        "financial_quality_risk_score": 0.1,
        "financial_surprise_score": 0.2,
        "financial_disclosure_quality_score": 0.8,
        "financial_report_event_types": "quarterly_report",
        "triggered_skills": "",
        "data_gaps": "",
    }


def test_balanced_sample_plan_excludes_future_columns_and_builds_micro48() -> None:
    dates = ["2023-09-01", "2024-08-02", "2025-05-20", "2026-01-09"]
    rows = []
    for index in range(24):
        row = _base_row(index, dates[index % len(dates)])
        if index % 5 == 0:
            row["news_opportunity_score"] = 0.6
            row["peer_relative_to_group_20d"] = -2.0
            row["financial_report_join_status"] = ""
            row["financial_report_event_count"] = 0
        if index % 7 == 0:
            row["news_warning_score"] = 0.5
            row["news_negative_materiality_30d"] = 0.8
            row["peer_group_positive_breadth_20d"] = 0.3
        if index % 11 == 0:
            row["prior_return_20d"] = 38.0
            row["rsi14"] = 82.0
            row["news_missing_rate"] = 0.9
            row["event_count"] = 0
        rows.append(row)

    frame = add_balanced_columns(add_interaction_columns(pd.DataFrame(rows)))
    metrics = build_rule_metrics(frame)
    plan = build_balanced_sample_plan(frame, metrics, max_rows=12, max_per_rule=4, max_per_block=4)
    micro = build_micro48_plan(plan)

    assert not plan.empty
    assert set(micro["time_block"]) <= {"H2023_2", "H2024_2", "H2025_1", "H2026_1"}
    assert len(micro) <= 4
    assert "return_20d" not in plan.columns
    assert "gt_status" not in plan.columns
    assert list(plan.columns) == PLAN_COLUMNS
    assert_no_future_plan_columns(plan)
    assert_no_future_plan_columns(micro)


def test_build_micro48_plan_prefers_one_row_per_target_block() -> None:
    rows = []
    for block, date, rule in [
        ("H2023_2", "2023-09-01", "counter_news_opportunity_peer_weak_or_fin_missing_v1"),
        ("H2024_2", "2024-08-02", "potential_active_clean_context_v1"),
        ("H2025_1", "2025-05-20", "potential_active_news_financial_confirmed_v1"),
        ("H2026_1", "2026-01-09", "potential_active_news_financial_confirmed_v1"),
    ]:
        row = {column: "" for column in PLAN_COLUMNS}
        row.update({"candidate_rule": rule, "time_block": block, "date": date, "code": str(len(rows) + 1).zfill(6), "name": f"S{len(rows)}"})
        row["return_20d"] = 99.0
        rows.append(row)
    plan = pd.DataFrame(rows)

    micro = build_micro48_plan(plan)

    assert len(micro) == 4
    assert set(micro["time_block"]) == {"H2023_2", "H2024_2", "H2025_1", "H2026_1"}
    assert "return_20d" not in micro.columns
    assert_no_future_plan_columns(micro)
