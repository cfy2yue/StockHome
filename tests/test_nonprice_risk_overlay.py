from __future__ import annotations

import pytest
import pandas as pd

from scripts.audit_nonprice_risk_overlay_v1 import (
    assert_no_future_fields,
    build_agent_previews,
    classify_prior_policy,
    validation_agrees,
)


def test_classify_prior_risk_downweight_candidate() -> None:
    metrics = {
        "flagged_rows": 120,
        "unflagged_rows": 300,
        "flag_vs_unflag_avg_delta": -2.5,
        "flag_vs_unflag_pos_delta": -0.08,
        "flag_vs_unflag_loss_delta": 0.06,
    }

    assert classify_prior_policy(metrics, "risk_or_friction") == "prior_risk_downweight_candidate"


def test_classify_prior_false_veto_guard_candidate() -> None:
    metrics = {
        "flagged_rows": 120,
        "unflagged_rows": 300,
        "flag_vs_unflag_avg_delta": 2.0,
        "flag_vs_unflag_pos_delta": 0.05,
        "flag_vs_unflag_loss_delta": -0.02,
    }

    assert classify_prior_policy(metrics, "risk_or_friction") == "prior_false_veto_guard_candidate"


def test_validation_agrees_with_risk_policy() -> None:
    current = {
        "flagged_rows": 30,
        "flag_vs_unflag_avg_delta": -0.5,
        "flag_vs_unflag_pos_delta": -0.01,
        "flag_vs_unflag_loss_delta": 0.02,
    }

    assert validation_agrees("prior_risk_downweight_candidate", current) is True


def test_preview_rejects_future_fields() -> None:
    with pytest.raises(ValueError):
        assert_no_future_fields({"nested": {"return_20d": 1.0}})


def test_agent_preview_keeps_policy_fields_without_future_metrics() -> None:
    metrics = pd.DataFrame(
        [
            {
                "scope_id": "pullback_high_rev_chip",
                "flag_id": "news_high_warning_any",
                "valid_block": "H2026_1",
                "train_blocks": "H2023_1,H2023_2,H2024_1,H2024_2,H2025_1,H2025_2",
                "prior_policy_status": "prior_risk_downweight_candidate",
                "flag_kind": "risk_or_friction",
            }
        ]
    )

    previews = build_agent_previews(metrics)

    assert len(previews) == 1
    preview = previews[0]
    assert preview["policy_status"] == "prior_risk_downweight_candidate"
    assert preview["action_hint"] == "downweight_or_request_confirmation"
    assert preview["selection_mode"] == "news_high_warning_any"
    assert preview["feature_group"] == "pullback_high_rev_chip"
    assert preview["top_features"]
    assert_no_future_fields(preview)
