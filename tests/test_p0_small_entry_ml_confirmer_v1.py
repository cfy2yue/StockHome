from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_small_entry_ml_confirmer_v1 import (
    assert_no_future_fields,
    feature_sets_for,
    forbidden_field,
    preview_num,
    promotion_status,
    prior_tail_train_validation,
    validation_threshold,
    write_jsonl,
)


def test_forbidden_field_blocks_future_and_label_names() -> None:
    assert forbidden_field("return_20d")
    assert forbidden_field("future_return_20d")
    assert forbidden_field("gt_pass")
    assert forbidden_field("single_stock_label")
    assert forbidden_field("cash_adjusted_return_20d")
    assert not forbidden_field("news_warning_score")
    assert not forbidden_field("kline_rsi14")


def test_feature_sets_do_not_include_future_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "opp_score": 0.8,
                "opp_threshold": 0.4,
                "opp_margin": 0.4,
                "kline_opp_margin": 0.2,
                "news_warning_score": 0.1,
                "financial_report_missing_rate": pd.NA,
                "corr_peer_relative_return_20d": 1.2,
                "return_20d": 5.0,
                "future_return_20d": 5.0,
            }
        ]
    )
    for feature_set in feature_sets_for(frame):
        assert "return_20d" not in feature_set.columns
        assert "future_return_20d" not in feature_set.columns
        assert all(not forbidden_field(col) for col in feature_set.columns)


def test_validation_threshold_uses_registered_quantile() -> None:
    threshold = validation_threshold(pd.Series([0.1, 0.2, 0.3, 0.4]), 0.5)
    assert threshold == pytest.approx(0.25)


def test_prior_tail_split_excludes_target_block() -> None:
    frame = pd.DataFrame(
        {
            "target_block": ["H2024_1"] * 10 + ["H2024_2"] * 10 + ["H2025_1"] * 10,
            "date": [f"2024-01-{idx:02d}" for idx in range(1, 11)]
            + [f"2024-07-{idx:02d}" for idx in range(1, 11)]
            + [f"2025-01-{idx:02d}" for idx in range(1, 11)],
            "code": [f"{idx:06d}" for idx in range(30)],
        }
    )
    train, validation, context = prior_tail_train_validation(
        frame, "H2025_1", validation_fraction=0.25, min_validation_rows=5
    )
    assert set(train["target_block"]).issubset({"H2024_1", "H2024_2"})
    assert set(validation["target_block"]).issubset({"H2024_1", "H2024_2"})
    assert "H2025_1" not in set(train["target_block"])
    assert "H2025_1" not in set(validation["target_block"])
    assert len(validation) == 5
    assert context.startswith("prior_tail_25pct")


def test_preview_jsonl_is_strict_and_null_safe(tmp_path) -> None:
    assert preview_num(float("nan")) is None
    assert preview_num("nan") is None

    path = tmp_path / "preview.jsonl"
    write_jsonl(path, pd.DataFrame([{"score": float("nan"), "nested": {"risk": pd.NA}}]))
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert '"score": null' in text
    assert '"risk": null' in text

    assert_no_future_fields({"score": 0.1})
    with pytest.raises(ValueError):
        assert_no_future_fields({"return_20d": 1.0})


def test_promotion_requires_non_sparse_prior_support() -> None:
    strong_h2026_sparse_prior = {
        "variant": "stack_margins_only__logistic_l1_c005__top40",
        "h2026_selected_rows": 82,
        "h2026_selected_rate": 0.52,
        "h2026_selected_pos20": 0.73,
        "h2026_selected_avg20": 7.1,
        "h2026_selected_loss_gt5": 0.08,
        "h2026_delta_pos": 0.18,
        "h2026_delta_avg": 3.6,
        "h2026_delta_loss": -0.12,
        "prior_blocks": 2,
        "prior_selected_rows_mean": 2,
        "prior_delta_pos_hit": 1.0,
        "prior_delta_avg_hit": 1.0,
    }
    assert promotion_status(strong_h2026_sparse_prior) == "observe_diagnostic_only"

    strong_h2026_supported_prior = {
        **strong_h2026_sparse_prior,
        "prior_selected_rows_mean": 30,
    }
    assert promotion_status(strong_h2026_supported_prior) == "green_candidate_for_ds_confirmation"
