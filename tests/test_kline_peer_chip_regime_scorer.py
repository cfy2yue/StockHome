import json

import pandas as pd

from scripts.run_kline_peer_chip_regime_scorer import (
    apply_frequency,
    build_rule_outcomes,
    future_keys,
    write_rule_outcomes,
)


def test_apply_frequency_filters_expected_weekdays():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-06", "2026-01-09", "2026-01-13", "2026-01-16"],
            "code": ["000001", "000002", "000003", "000004"],
        }
    )

    tuesday = apply_frequency(frame, "weekly_tuesday")
    friday = apply_frequency(frame, "weekly_friday")

    assert list(tuesday["date"]) == ["2026-01-06", "2026-01-13"]
    assert list(friday["date"]) == ["2026-01-09", "2026-01-16"]


def test_kline_peer_chip_rule_outcomes_are_agent_safe(tmp_path):
    aggregate = pd.DataFrame(
        [
            {
                "task_mode": "portfolio_pool",
                "variant": "logistic_kline_peer_chip_regime",
                "top_pct": 0.05,
                "decision_frequency": "every_2_weeks",
                "promotion_status": "observe_latest_positive_prior_weak",
            },
            {
                "task_mode": "portfolio_pool",
                "variant": "baseline_rev_chip_score",
                "top_pct": 0.05,
                "decision_frequency": "every_2_weeks",
                "promotion_status": "rejected_or_diagnostic_only",
            },
        ]
    )

    rows = build_rule_outcomes(aggregate)

    assert len(rows) == 1
    assert rows[0]["usable_in_agent_default"] is False
    assert rows[0]["decision_frequency"] == "every_2_weeks"
    assert not future_keys(rows[0])

    output = tmp_path / "kline_peer_chip_rule_outcomes.jsonl"
    write_rule_outcomes(output, rows)
    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows
