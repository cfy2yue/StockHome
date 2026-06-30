from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_transfer_channel_confirm_v1 import (
    add_channel_flags,
    apply_channel_gates,
    gate_status,
    load_transfer_configs,
    write_jsonl,
)


def test_channel_flags_count_support_and_hard_counter() -> None:
    frame = pd.DataFrame(
        [
            {
                "triggered_skills": "PPS-Q-017",
                "book_score": 0.2,
                "news_count_30d": 3,
                "news_missing_rate": 0,
                "news_warning_score": 0.2,
                "news_opportunity_score": 0.7,
                "official_confirmation_score": 0,
                "announcement_materiality_score": 0,
                "financial_report_join_status": "no_event_in_window",
                "financial_report_missing_rate": 0,
                "financial_quality_risk_score": 0.1,
                "financial_surprise_score": 0.1,
                "peer_group_positive_breadth_20d": 0.6,
                "peer_relative_to_group_20d": -1,
                "tushare_industry_positive_breadth_20d": 0,
                "tushare_area_positive_breadth_20d": 0,
                "prior_return_20d": -2,
                "drawdown60": -4,
                "rsi14": 55,
                "lower_support": 0.2,
                "upper_overhang": 0.1,
            },
            {
                "triggered_skills": "PPS-M-003",
                "book_score": -0.8,
                "news_count_30d": 2,
                "news_missing_rate": 0,
                "news_warning_score": 0.8,
                "news_opportunity_score": 0.1,
                "official_confirmation_score": 0,
                "announcement_materiality_score": 0,
                "financial_report_join_status": "event_window_matched",
                "financial_report_missing_rate": 0,
                "financial_quality_risk_score": 0.8,
                "financial_surprise_score": -0.6,
                "peer_group_positive_breadth_20d": 0.2,
                "peer_relative_to_group_20d": -1,
                "tushare_industry_positive_breadth_20d": 0,
                "tushare_area_positive_breadth_20d": 0,
                "prior_return_20d": 8,
                "drawdown60": -2,
                "rsi14": 78,
                "lower_support": 0,
                "upper_overhang": 0.7,
            },
        ]
    )
    out = add_channel_flags(frame)
    assert int(out.loc[0, "channel_support_count"]) == 5
    assert int(out.loc[0, "channel_hard_counter_count"]) == 0
    assert bool(out.loc[0, "news_low_warning"])
    assert bool(out.loc[0, "financial_no_recent_event"])

    assert int(out.loc[1, "channel_support_count"]) == 0
    assert int(out.loc[1, "channel_hard_counter_count"]) == 3
    assert bool(out.loc[1, "news_high_warning"])
    assert bool(out.loc[1, "financial_high_risk_event"])


def test_channel_gates_are_fail_closed() -> None:
    base = add_channel_flags(
        pd.DataFrame(
            [
                {
                    "triggered_skills": "PPS-Q-017",
                    "book_score": 0,
                    "news_count_30d": 2,
                    "news_missing_rate": 0,
                    "news_warning_score": 0.1,
                    "news_opportunity_score": 0.6,
                    "financial_report_join_status": "no_event_in_window",
                    "financial_report_missing_rate": 0,
                    "peer_group_positive_breadth_20d": 0.7,
                    "rsi14": 50,
                    "lower_support": 0.3,
                },
                {
                    "triggered_skills": "PPS-Q-017",
                    "book_score": -0.7,
                    "news_count_30d": 2,
                    "news_missing_rate": 0,
                    "news_warning_score": 0.9,
                    "news_opportunity_score": 0.1,
                    "financial_report_join_status": "event_window_matched",
                    "financial_report_missing_rate": 0,
                    "financial_quality_risk_score": 0.8,
                    "financial_surprise_score": -0.8,
                    "rsi14": 80,
                    "lower_support": 0.3,
                },
            ]
        )
    )
    gates = apply_channel_gates(base)
    assert len(gates["transfer_only"]) == 2
    assert len(gates["no_hard_counter"]) == 1
    assert len(gates["support_min2_no_hard"]) == 1
    assert len(gates["news_financial_clean_no_hard"]) == 1
    assert len(gates["chip_support_no_overheat_no_hard"]) == 1


def test_gate_status_requires_prior_and_size_support() -> None:
    green = {
        "gate_id": "news_financial_clean_no_hard",
        "h2026_selected_rows": 60,
        "prior_blocks": 3,
        "prior_selected_rows_mean": 40,
        "prior_delta_pos_hit": 0.75,
        "prior_delta_avg_hit": 0.75,
        "h2026_selected_pos20": 0.72,
        "h2026_selected_avg20": 5.5,
        "h2026_selected_loss_gt5": 0.1,
        "h2026_delta_pos_vs_transfer": 0.04,
        "h2026_delta_avg_vs_transfer": 0.2,
    }
    assert gate_status(green) == "green_candidate_for_ds_confirmation"
    assert gate_status({**green, "gate_id": "transfer_only"}) == "transfer_reference"
    assert gate_status({**green, "h2026_selected_rows": 10}) == "reject_too_sparse"
    assert gate_status({**green, "prior_delta_pos_hit": 0.5}) == "yellow_candidate_needs_fresh_panel"


def test_preview_writer_rejects_future_fields(tmp_path) -> None:
    path = tmp_path / "preview.jsonl"
    write_jsonl(path, pd.DataFrame([{"safe_score": 0.1, "code": "000001"}]))
    assert path.read_text(encoding="utf-8").strip()
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "bad.jsonl", pd.DataFrame([{"return_20d": 1.0}]))


def test_load_transfer_configs_can_expand_status_regex(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "frequency": "every_2_weeks",
                "variant": "yellow_v",
                "cohort": "all_scored_rows",
                "feature_set": "stack_plus_all_channels",
                "model_name": "logistic_l1_c005",
                "confirm_quantile": 0.7,
                "promotion_status": "yellow_candidate_needs_fresh_panel",
                "rank_score": 2.0,
            },
            {
                "frequency": "every_2_weeks",
                "variant": "observe_v",
                "cohort": "all_scored_rows",
                "feature_set": "stack_plus_all_channels",
                "model_name": "logistic_l1_c005",
                "confirm_quantile": 0.5,
                "promotion_status": "observe_diagnostic_only",
                "rank_score": 3.0,
            },
        ]
    ).to_csv(path, index=False)

    default_configs = load_transfer_configs(path)
    assert [cfg.variant for cfg in default_configs] == ["yellow_v"]

    expanded_configs = load_transfer_configs(path, status_regex="yellow|observe", max_configs=1)
    assert [cfg.variant for cfg in expanded_configs] == ["observe_v"]
