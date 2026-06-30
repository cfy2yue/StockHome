import json

import pandas as pd

from scripts.audit_analogue_case_context_v2 import (
    analogue_branch,
    build_agent_preview_rows,
    evaluate_variant,
)


def test_analogue_branch_classifies_decay_and_support_cases():
    assert analogue_branch(0.40, -2.5, 20) == "decay_warning_low_support"
    assert analogue_branch(0.62, 0.1, 40) == "historical_supportive"
    assert analogue_branch(0.52, -3.0, 30) == "regime_decay_warning"
    assert analogue_branch(0.41, 0.5, 30) == "low_historical_support"
    assert analogue_branch(0.55, 0.5, 3) == "insufficient_case_pool"


def test_evaluate_variant_tracks_selected_analogue_branches():
    frame = pd.DataFrame(
        [
            {"date": "2026-01-06", "code": "000001", "time_block": "H2026_1", "return_20d": 3.0, "score": 0.9, "analogue_branch": "historical_supportive"},
            {"date": "2026-01-06", "code": "000002", "time_block": "H2026_1", "return_20d": -1.0, "score": 0.8, "analogue_branch": "regime_decay_warning"},
            {"date": "2026-01-06", "code": "000003", "time_block": "H2026_1", "return_20d": 1.0, "score": 0.1, "analogue_branch": "neutral_or_mixed"},
            {"date": "2026-01-13", "code": "000001", "time_block": "H2026_1", "return_20d": 2.0, "score": 0.2, "analogue_branch": "neutral_or_mixed"},
            {"date": "2026-01-13", "code": "000002", "time_block": "H2026_1", "return_20d": 4.0, "score": 0.95, "analogue_branch": "decay_warning_low_support"},
            {"date": "2026-01-13", "code": "000003", "time_block": "H2026_1", "return_20d": -2.0, "score": 0.1, "analogue_branch": "neutral_or_mixed"},
        ]
    )

    row = evaluate_variant(
        frame,
        variant="score",
        task_mode="portfolio_pool",
        valid_block="H2026_1",
        decision_frequency="weekly_tuesday",
        top_pct=1 / 3,
    )

    assert row["selected_rows"] == 2
    assert row["positive_20d_rate"] == 1.0
    assert row["selected_historical_supportive_rate"] == 0.5
    assert row["selected_decay_warning_rate"] == 0.5


def test_agent_preview_rows_are_sanitized_and_research_only():
    aggregate = pd.DataFrame(
        [
            {
                "task_mode": "single_stock",
                "variant": "rev_chip_analogue_guard",
                "decision_frequency": "every_2_weeks",
                "top_pct": 0.1,
                "promotion_status": "observe_relative_improvement_context_candidate",
            }
        ]
    )
    coverage = pd.DataFrame(
        [
            {"feature": "analogue_pos_rate", "non_null_rate": 0.95},
            {"feature": "regime_decay_signal", "non_null_rate": 0.90},
        ]
    )

    rows = build_agent_preview_rows(aggregate, coverage)
    payload = json.loads(json.dumps(rows[0], ensure_ascii=False))

    assert payload["research_only"] is True
    assert payload["not_investment_instruction"] is True
    assert payload["usable_in_agent_default"] is True
    text = json.dumps(payload, ensure_ascii=False)
    assert "return_20d" not in text
    assert "gt_status" not in text
    assert "future_return" not in text
