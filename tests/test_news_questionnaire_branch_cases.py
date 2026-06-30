from __future__ import annotations

import pandas as pd

from scripts.audit_news_questionnaire_branch_cases import (
    add_branch_labels,
    assert_no_future_fields,
    assign_branch_tags,
    build_agent_previews,
    build_prior_policy_metrics,
    build_prior_similar_cases,
)


def test_assign_branch_tags_separates_explicit_negative_from_reversal_friction() -> None:
    explicit = assign_branch_tags(
        {
            "ds_news_risk_score": 0.9,
            "ds_news_self_regulatory_legal": -2.0,
            "ds_news_mainline_summary": "公司受到监管处罚并存在诉讼风险",
            "rev_chip_score_quantile": 0.92,
            "kline_return_20d": -12.0,
            "lower_support": 0.7,
        }
    )
    friction = assign_branch_tags(
        {
            "ds_news_risk_score": 0.55,
            "ds_news_uncertainty_score": 0.75,
            "ds_news_missing_or_conflict_notes": "股价异常波动，存在风险提示，但未见明确负面处罚",
            "rev_chip_score_quantile": 0.93,
            "kline_return_20d": -16.0,
            "kline_drawdown_60d": -24.0,
            "lower_support": 0.72,
            "upper_overhang": 0.3,
        }
    )

    assert explicit["primary_branch"] == "explicit_negative_event"
    assert "reversible_reversal_friction" in friction["tags"]
    assert friction["primary_branch"] == "reversible_reversal_friction"
    assert "do_not_hard_veto_from_news_risk_alone" in friction["branch_policy"]


def test_add_branch_labels_marks_routine_and_soft_gap() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-03",
                "code": "000001",
                "ds_news_official_support": 0.9,
                "ds_news_opportunity_score": 0.1,
                "ds_news_risk_score": 0.1,
                "ds_news_mainline_clarity": 0.2,
                "ds_news_decision_relevance": 0.2,
                "ds_news_repetition_lag": 0.8,
                "ds_news_source_coverage": 0.9,
            },
            {
                "date": "2025-01-03",
                "code": "000002",
                "ds_news_uncertainty_score": 0.9,
                "ds_news_risk_score": 0.1,
                "ds_news_opportunity_score": 0.1,
                "ds_news_source_coverage": 0.1,
            },
        ]
    )

    out = add_branch_labels(frame)

    assert out.loc[0, "primary_news_branch"] == "routine_official_low_signal"
    assert out.loc[1, "primary_news_branch"] == "soft_gap"


def test_agent_preview_omits_future_and_result_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2023-07-04",
                "code": "000001",
                "name": "A",
                "time_block": "H2023_2",
                "primary_news_branch": "soft_gap",
                "news_branch_tags": "soft_gap",
                "branch_policy": "soft_gap_confidence_discount",
                "branch_rationale": "coverage weak",
                "return_20d": 12.3,
                "pool_excess_20d": 1.2,
            },
            {
                "date": "2024-01-04",
                "code": "000002",
                "name": "B",
                "time_block": "H2024_1",
                "primary_news_branch": "soft_gap",
                "news_branch_tags": "soft_gap",
                "branch_policy": "soft_gap_confidence_discount",
                "branch_rationale": "coverage weak",
                "return_20d": -3.0,
                "pool_excess_20d": -1.0,
            },
        ]
    )
    prior = build_prior_policy_metrics(frame)
    similar = build_prior_similar_cases(frame)

    previews = build_agent_previews(frame, prior, similar, max_rows=10)
    for item in previews:
        assert_no_future_fields(item)
        rendered = str(item)
        assert "return_20d" not in rendered
        assert "pool_excess_20d" not in rendered
        assert "gt_status" not in rendered


def test_assert_no_future_fields_rejects_nested_future_key() -> None:
    try:
        assert_no_future_fields({"nested": [{"return_20d": 1.0}]})
    except ValueError as exc:
        assert "return_20d" in str(exc)
    else:
        raise AssertionError("future field was not rejected")
