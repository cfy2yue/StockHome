from __future__ import annotations

import json

import pandas as pd

from scripts.audit_single_stock_kline_frequency_tool_v1 import (
    build_agent_tool_preview,
    feature_rank_ic_summary,
)


def test_feature_rank_ic_summary_learns_sign_by_prior_dates() -> None:
    rows = []
    for date in ["2025-01-03", "2025-01-10"]:
        for i in range(25):
            rows.append(
                {
                    "date": date,
                    "return_20d": float(i),
                    "feature_positive": float(i),
                    "feature_negative": float(24 - i),
                    "feature_constant": 1.0,
                }
            )
    frame = pd.DataFrame(rows)

    summary = feature_rank_ic_summary(frame, ["feature_positive", "feature_negative", "feature_constant"])

    assert summary["feature_positive"].mean() > 0.99
    assert summary["feature_negative"].mean() < -0.99
    assert summary["feature_constant"].empty


def test_agent_tool_preview_is_sanitized_and_research_only() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "decision_frequency": "weekly_friday",
                "feature_group": "rev_chip_core_fixed",
                "promotion_status": "observe_latest_positive_with_risk_value",
                "return_20d": 99.0,
                "gt_status": "evaluated",
            }
        ]
    )
    weights = pd.DataFrame(
        [
            {
                "decision_frequency": "weekly_friday",
                "feature_group": "rev_chip_core_fixed",
                "feature": "kline_return_20d",
                "weight": -0.3,
            }
        ]
    )

    rows = build_agent_tool_preview(aggregate, weights)
    payload = json.loads(json.dumps(rows[0], ensure_ascii=False))
    keys = set()

    def collect_keys(value):
        if isinstance(value, dict):
            keys.update(value.keys())
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(payload)

    assert payload["research_only"] is True
    assert payload["not_investment_instruction"] is True
    assert payload["task_mode"] == "single_stock"
    assert "return_20d" not in keys
    assert "gt_status" not in keys
    assert "future_return" not in keys
