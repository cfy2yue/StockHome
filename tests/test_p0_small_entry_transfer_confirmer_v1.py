from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_small_entry_transfer_confirmer_v1 import (
    add_derived_channel_features,
    apply_train_cohort,
    assert_no_future_fields,
    feature_sets_for,
    forbidden_field,
    prior_tail_train_validation,
    promotion_status,
)


def test_derived_channel_features_are_numeric_and_skill_specific() -> None:
    frame = pd.DataFrame(
        [
            {
                "news_opportunity_score": 0.7,
                "news_warning_score": 0.2,
                "peer_group_news_opportunity_avg": 0.3,
                "peer_group_news_risk_avg": 0.1,
                "news_missing_rate": 0.25,
                "news_evidence_quality": 0.8,
                "financial_disclosure_quality_score": 0.6,
                "financial_quality_risk_score": 0.1,
                "financial_surprise_score": 0.4,
                "peer_relative_to_group_20d": 1.0,
                "corr_peer_relative_return_20d": 2.0,
                "lower_support": 0.8,
                "upper_overhang": 0.3,
                "triggered_skills": "PPS-Q-017;DOW-B-004",
            }
        ]
    )
    out = add_derived_channel_features(frame)
    assert out.loc[0, "news_opportunity_minus_warning"] == pytest.approx(0.5)
    assert out.loc[0, "news_peer_opportunity_gap"] == pytest.approx(0.4)
    assert out.loc[0, "financial_quality_minus_risk"] == pytest.approx(0.5)
    assert out.loc[0, "peer_relative_strength_blend"] == pytest.approx(3.0)
    assert int(out.loc[0, "skill_PPS_Q_017_triggered"]) == 1
    assert int(out.loc[0, "skill_PPS_M_003_triggered"]) == 0


def test_feature_sets_exclude_future_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "target_position": 0.4,
                "opp_score": 0.7,
                "opp_threshold": 0.5,
                "news_warning_score": 0.1,
                "financial_quality_minus_risk": 0.2,
                "return_20d": 5.0,
                "gt_status": "ok",
            }
        ]
    )
    assert forbidden_field("return_20d")
    assert forbidden_field("gt_status")
    for feature_set in feature_sets_for(frame):
        assert "return_20d" not in feature_set.columns
        assert "gt_status" not in feature_set.columns


def test_prior_tail_train_validation_excludes_current_target() -> None:
    frame = pd.DataFrame(
        {
            "target_block": ["H2024_1"] * 50 + ["H2024_2"] * 50,
            "date": [f"2024-01-{idx % 28 + 1:02d}" for idx in range(100)],
            "code": [f"{idx:06d}" for idx in range(100)],
        }
    )
    train, validation, context = prior_tail_train_validation(frame, validation_fraction=0.2, min_validation_rows=10)
    assert len(train) == 80
    assert len(validation) == 20
    assert context.startswith("prior_tail_20pct")


def test_apply_train_cohort_filters_opportunity_context() -> None:
    frame = pd.DataFrame(
        {
            "target_position": [0.0, 0.12, 0.0],
            "opp_quantile_in_date": [0.1, 0.2, 0.75],
        }
    )
    out = apply_train_cohort(frame, "opportunity_context_rows")
    assert len(out) == 2


def test_promotion_requires_prior_support() -> None:
    sparse = {
        "variant": "all_scored_rows__stack_plus_all_channels__logistic_l2_c050__top50",
        "h2026_selected_rows": 100,
        "h2026_selected_rate": 0.4,
        "h2026_selected_pos20": 0.7,
        "h2026_selected_avg20": 5,
        "h2026_selected_loss_gt5": 0.1,
        "h2026_delta_pos": 0.08,
        "h2026_delta_avg": 2,
        "h2026_delta_loss": -0.02,
        "prior_blocks": 2,
        "prior_selected_rows_mean": 5,
        "prior_delta_pos_hit": 1.0,
        "prior_delta_avg_hit": 1.0,
    }
    assert promotion_status(sparse) == "observe_diagnostic_only"
    supported = {**sparse, "prior_selected_rows_mean": 50}
    assert promotion_status(supported) == "green_candidate_for_ds_confirmation"


def test_preview_future_guard() -> None:
    assert_no_future_fields({"safe_score": 0.1})
    with pytest.raises(ValueError):
        assert_no_future_fields({"future_return_20d": 1.0})
