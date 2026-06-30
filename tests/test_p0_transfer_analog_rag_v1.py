from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_transfer_analog_rag_v1 import (
    AnalogSpec,
    add_analog_features,
    analog_feature_columns,
    analog_status,
    apply_analog_gates,
)
from scripts.audit_p0_transfer_channel_confirm_v1 import write_jsonl


def test_analog_features_use_nearest_historical_cases() -> None:
    bank_rows = []
    for idx in range(8):
        bank_rows.append(
            {
                "code": f"0000{idx:02d}",
                "date": f"2025-01-{idx + 1:02d}",
                "_transfer_score": 0.8 + idx * 0.001,
                "channel_support_count": 3,
                "channel_hard_counter_count": 0,
                "news_warning_score": 0.1,
                "lower_support": 0.3,
                "return_20d": 6.0 + idx,
            }
        )
    for idx in range(8):
        bank_rows.append(
            {
                "code": f"0001{idx:02d}",
                "date": f"2025-02-{idx + 1:02d}",
                "_transfer_score": 0.2 + idx * 0.001,
                "channel_support_count": 0,
                "channel_hard_counter_count": 2,
                "news_warning_score": 0.9,
                "lower_support": 0.0,
                "return_20d": -7.0 - idx,
            }
        )
    bank = pd.DataFrame(bank_rows)
    target = pd.DataFrame(
        [
            {
                "code": "300001",
                "date": "2026-01-01",
                "_transfer_score": 0.82,
                "channel_support_count": 3,
                "channel_hard_counter_count": 0,
                "news_warning_score": 0.1,
                "lower_support": 0.32,
            },
            {
                "code": "300002",
                "date": "2026-01-01",
                "_transfer_score": 0.21,
                "channel_support_count": 0,
                "channel_hard_counter_count": 2,
                "news_warning_score": 0.92,
                "lower_support": 0.0,
            },
        ]
    )
    out = add_analog_features(target, bank, spec=AnalogSpec("unit", top_k=5, min_neighbors=5))
    assert out.loc[0, "analog_pos_rate"] == pytest.approx(1.0)
    assert out.loc[1, "analog_pos_rate"] == pytest.approx(0.0)
    assert "2025-" in out.loc[0, "analog_top_case_refs"]


def test_analog_gates_filter_by_history_and_channel_context() -> None:
    frame = pd.DataFrame(
        [
            {
                "analog_pos_rate": 0.72,
                "analog_avg_return": 5.0,
                "analog_loss_gt5_rate": 0.1,
                "analog_neighbor_count": 20,
                "analog_min_neighbors": 10,
                "news_low_warning": True,
                "financial_no_recent_event": True,
                "financial_high_risk_event": False,
                "financial_missing": False,
                "channel_hard_counter_count": 0,
                "chip_support_visible": True,
                "kline_not_overheated": True,
            },
            {
                "analog_pos_rate": 0.44,
                "analog_avg_return": -1.0,
                "analog_loss_gt5_rate": 0.4,
                "analog_neighbor_count": 20,
                "analog_min_neighbors": 10,
                "news_low_warning": False,
                "financial_no_recent_event": False,
                "financial_high_risk_event": True,
                "financial_missing": False,
                "channel_hard_counter_count": 1,
                "chip_support_visible": True,
                "kline_not_overheated": True,
            },
        ]
    )
    gates = apply_analog_gates(frame)
    assert len(gates["transfer_only"]) == 2
    assert len(gates["analog_pos_ge065"]) == 1
    assert len(gates["analog_pos_ge065_loss_le020"]) == 1
    assert len(gates["news_financial_clean_plus_analog065"]) == 1
    assert len(gates["analog_guard_remove_weak_cases"]) == 1


def test_analog_status_requires_prior_support() -> None:
    row = {
        "gate_id": "analog_pos_ge065",
        "h2026_selected_rows": 45,
        "prior_blocks": 3,
        "prior_evaluable_blocks": 3,
        "prior_selected_rows_mean": 25,
        "prior_evaluable_selected_rows_mean": 25,
        "prior_delta_pos_hit": 0.75,
        "prior_delta_avg_hit": 0.75,
        "prior_evaluable_delta_pos_hit": 0.75,
        "prior_evaluable_delta_avg_hit": 0.75,
        "h2026_selected_pos20": 0.72,
        "h2026_selected_avg20": 5.2,
        "h2026_selected_loss_gt5": 0.1,
        "h2026_delta_pos_vs_transfer": 0.05,
        "h2026_delta_avg_vs_transfer": 0.2,
    }
    assert analog_status(row) == "green_candidate_for_ds_confirmation"
    assert analog_status({**row, "gate_id": "transfer_only"}) == "transfer_reference"
    assert analog_status({**row, "h2026_selected_rows": 10}) == "reject_too_sparse"
    assert analog_status({**row, "prior_evaluable_delta_pos_hit": 0.5}) == "yellow_candidate_needs_fresh_panel"
    assert (
        analog_status({**row, "prior_evaluable_blocks": 1})
        == "observe_latest_block_bright_time_support_insufficient"
    )


def test_analog_features_and_preview_exclude_future_fields(tmp_path) -> None:
    frame = pd.DataFrame([{"_transfer_score": 0.5, "return_20d": 3.0, "gt_status": "ok"}])
    cols = analog_feature_columns(frame)
    assert "return_20d" not in cols
    assert "gt_status" not in cols
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "bad.jsonl", pd.DataFrame([{"future_return_20d": 1.0}]))
