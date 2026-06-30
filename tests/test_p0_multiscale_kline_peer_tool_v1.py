from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_p0_multiscale_kline_peer_tool_v1 import (
    apply_frequency,
    assert_no_future_fields,
    build_agent_preview_rows,
    promotion_status,
    write_jsonl,
)


def test_apply_frequency_keeps_expected_decision_days() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-06", "2026-01-09", "2026-01-13", "2026-01-16"],
            "code": ["000001", "000002", "000003", "000004"],
        }
    )

    assert list(apply_frequency(frame, "weekly_tuesday")["date"]) == ["2026-01-06", "2026-01-13"]
    assert list(apply_frequency(frame, "weekly_friday")["date"]) == ["2026-01-09", "2026-01-16"]
    assert list(apply_frequency(frame, "every_2_weeks")["date"]) == ["2026-01-06", "2026-01-09"]


def test_promotion_status_requires_prior_latest_and_risk_gates() -> None:
    green = {
        "prior_active_pos_delta_hit_rate": 0.75,
        "prior_active_avg_delta_hit_rate": 1.0,
        "h2026_active_pos": 0.61,
        "h2026_active_avg_delta": 0.8,
        "h2026_active_rate": 0.20,
        "h2026_loss_exposure_reduction": 0.01,
        "h2026_risk_false_veto_positive_rate": 0.50,
    }
    latest_only = {**green, "prior_active_pos_delta_hit_rate": 0.50}
    prior_only = {**green, "h2026_active_pos": 0.48}
    risky = {**green, "h2026_risk_false_veto_positive_rate": 0.70}

    assert promotion_status(green) == "green_candidate_for_small_agent_ablation"
    assert promotion_status(latest_only) == "yellow_latest_positive_prior_weak"
    assert promotion_status(prior_only) == "yellow_prior_positive_latest_weak"
    assert promotion_status(risky) == "reject_or_diagnostic_only"


def test_agent_preview_jsonl_rejects_future_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="future/result field leaked"):
        assert_no_future_fields({"date": "2026-01-02", "code": "000001", "return_20d": 1.0})

    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "code": "000001",
                "name": "测试股份",
                "time_block": "H2026_1",
                "opp_score": 0.91,
                "risk_score": 0.10,
            },
            {
                "date": "2026-01-06",
                "code": "000002",
                "name": "风险股份",
                "time_block": "H2026_1",
                "opp_score": 0.40,
                "risk_score": 0.95,
            },
        ]
    )
    metrics = {
        "frequency": "weekly_tuesday",
        "feature_group": "kline_peer_chip",
        "model": "hgb",
        "opp_threshold": 0.80,
        "risk_threshold": 0.90,
    }

    rows = build_agent_preview_rows(frame, metrics)
    output = tmp_path / "agent_preview.jsonl"
    write_jsonl(output, pd.DataFrame(rows))
    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert loaded[0]["action_hint"] == "trial_buy_or_hold_review"
    assert loaded[1]["action_hint"] == "avoid_or_reduce_review"
    assert all("return_20d" not in row for row in loaded)
