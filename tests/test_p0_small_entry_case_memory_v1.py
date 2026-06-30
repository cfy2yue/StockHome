from __future__ import annotations

import pytest

from scripts.audit_p0_small_entry_case_memory_v1 import promotion_status, rank_score
from scripts.audit_p0_friday_stack_case_memory_v1 import assert_no_future_fields


def test_case_memory_promotion_requires_prior_support() -> None:
    strong_h2026_sparse_prior = {
        "guard_policy": "condition_financial_report_context",
        "h2026_retained_rate": 0.65,
        "h2026_retained_pos20": 0.70,
        "h2026_retained_avg20_pp": 5.2,
        "h2026_retained_loss_gt5": 0.10,
        "h2026_delta_pos": 0.08,
        "h2026_delta_avg": 1.1,
        "h2026_delta_loss": -0.03,
        "h2026_false_veto_positive_rows": 4,
        "h2026_captured_loss_gt5_rows": 5,
        "prior_blocks": 1,
        "prior_retained_rows_mean": 60,
        "prior_delta_pos_hit": 1.0,
        "prior_delta_avg_hit": 1.0,
    }
    assert promotion_status(strong_h2026_sparse_prior) == "observe_diagnostic_only"

    supported = {**strong_h2026_sparse_prior, "prior_blocks": 2}
    assert promotion_status(supported) == "green_candidate_for_small_ds_smoke"


def test_case_memory_false_veto_blocks_promotion() -> None:
    too_many_false_vetoes = {
        "guard_policy": "applicable_any",
        "h2026_retained_rate": 0.70,
        "h2026_retained_pos20": 0.72,
        "h2026_retained_avg20_pp": 4.1,
        "h2026_retained_loss_gt5": 0.08,
        "h2026_delta_pos": 0.07,
        "h2026_delta_avg": 0.9,
        "h2026_delta_loss": -0.02,
        "h2026_false_veto_positive_rows": 30,
        "h2026_captured_loss_gt5_rows": 2,
        "prior_blocks": 3,
        "prior_retained_rows_mean": 80,
        "prior_delta_pos_hit": 1.0,
        "prior_delta_avg_hit": 1.0,
    }
    assert promotion_status(too_many_false_vetoes) == "observe_diagnostic_only"
    assert rank_score(too_many_false_vetoes) > 0


def test_safe_preview_future_fields_still_forbidden() -> None:
    assert_no_future_fields({"case_guard_hint": "review"})
    with pytest.raises(ValueError):
        assert_no_future_fields({"return_20d": 1.2})
