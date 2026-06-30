import json

import pandas as pd

from scripts.run_channel_rule_outcome_classifier import (
    build_rule_outcomes,
    make_rule_outcome_label,
    select_top_by_date,
    write_rule_outcomes,
)
from src.agent_training.quant_tool_context import FUTURE_RESULT_FIELDS


def test_make_rule_outcome_label_separates_hard_soft_and_positive():
    frame = pd.DataFrame(
        {
            "return_20d": [-8.0, 1.5, 6.0, 0.2],
            "pool_excess_20d": [-6.0, -0.5, 3.2, -0.2],
            "pool_return_rank_pct": [0.05, 0.45, 0.82, 0.5],
            "news_missing_rate": [0.0, 0.9, 0.0, 0.0],
            "financial_report_missing_rate": [0.0, 0.8, 0.0, 0.0],
            "triggered_skills": ["BS-1", "", "BS-2", "BS-3"],
            "news_evidence_quality": [0.8, 0.2, 0.8, 0.8],
        }
    )

    labels = list(make_rule_outcome_label(frame))

    assert labels == ["hard_counter", "soft_gap", "positive_support", "neutral"]


def test_select_top_by_date_keeps_each_date_slice():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"],
            "code": ["000002", "000001", "000004", "000003"],
        }
    )
    score = pd.Series([0.1, 0.9, 0.2, 0.8], index=frame.index)

    selected = select_top_by_date(frame, score, top_pct=0.5)

    assert list(selected["code"]) == ["000001", "000003"]


def test_channel_rule_outcomes_are_agent_safe(tmp_path):
    aggregate = pd.DataFrame(
        [
            {
                "variant": "logistic_channel_outcome",
                "action_class": "hard_counter",
                "promotion_status": "accepted_guard_candidate",
            },
            {
                "variant": "manual",
                "action_class": "positive_support",
                "promotion_status": "rejected_or_diagnostic_only",
            },
        ]
    )

    rows = build_rule_outcomes(aggregate)

    assert len(rows) == 1
    assert rows[0]["usable_in_agent_default"] is False
    assert rows[0]["selection_mode"] == "hard_counter"
    assert not _future_keys(rows[0])

    output = tmp_path / "channel_rule_outcomes.jsonl"
    write_rule_outcomes(output, rows)
    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows


def _future_keys(value):
    if isinstance(value, dict):
        leaked = {str(key) for key in value if str(key) in FUTURE_RESULT_FIELDS}
        for child in value.values():
            leaked.update(_future_keys(child))
        return leaked
    if isinstance(value, list):
        leaked = set()
        for child in value:
            leaked.update(_future_keys(child))
        return leaked
    return set()
