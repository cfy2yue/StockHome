from __future__ import annotations

import pytest

import pandas as pd

from scripts.audit_bookskill_grounding_v2 import (
    assert_no_future_fields,
    build_agent_previews,
    build_evidence_rows,
    classify_policy,
    summarize_branch_strategy,
    summarize_strategy_ids,
)


def test_bookskill_grounding_explodes_single_stock_ids(tmp_path) -> None:
    detail_path = tmp_path / "single.csv"
    pd.DataFrame(
        [
            {
                "variant": "full_agent_with_risk_review_queue",
                "task_mode": "single_stock",
                "valid_block": "H2026_1",
                "decision_date": "2026-01-06",
                "code": "1",
                "name": "测试",
                "bookskill_strategy_ids": "PPS-Q-017;DOW-B-017;__truncated__",
                "primary_risk_branch": "overheat_reversal_friction_without_hard_event",
                "risk_tier": "hard_counter",
                "research_grade": "放入观察",
                "simulated_weight_change": 0.05,
                "return_20d": -10,
                "cash_adjusted_return_20d": -0.2,
                "delta_cash": 0.5,
                "paired_change_type": "lowered_negative",
            }
        ]
    ).to_csv(detail_path, index=False)
    source_cards = {
        "PPS-Q-017": {"source_book": "专业投机原理", "source_status": "grounded"},
        "DOW-B-017": {"source_book": "道氏理论", "source_status": "grounded"},
    }

    rows = build_evidence_rows([("sample", detail_path)], source_cards)

    assert set(rows["strategy_id"]) == {"PPS-Q-017", "DOW-B-017"}
    assert set(rows["branch"]) == {"overheat_reversal_friction_without_hard_event"}
    assert int(rows["research_only"].sum()) == 2


def test_bookskill_grounding_policy_and_preview_are_safe(tmp_path) -> None:
    detail_path = tmp_path / "portfolio.csv"
    pd.DataFrame(
        [
            {
                "variant": "full_agent_with_quant_tools",
                "task_mode": "portfolio_pool",
                "valid_block": "H2025_1",
                "decision_date": "2025-01-03",
                "code": "1",
                "name": "测试",
                "strategy_id": "PPS-Q-017",
                "source_book": "专业投机原理",
                "source_status": "grounded",
                "research_grade": "放入观察",
                "simulated_weight_change": 0.1,
                "active_exposure": False,
                "posthoc_return_20d": 2.0,
                "posthoc_cash_adjusted_return_20d": 0.3,
            }
        ]
    ).to_csv(detail_path, index=False)
    source_cards = {
        "PPS-Q-017": {
            "source_book": "专业投机原理",
            "source_status": "grounded",
            "applicable_condition": "需要多通道确认",
            "failure_condition": "缺新闻或财报时降权",
        }
    }

    rows = build_evidence_rows([("portfolio", detail_path)], source_cards)
    summary = summarize_strategy_ids(rows, source_cards, pd.DataFrame())
    branch = summarize_branch_strategy(rows)
    previews = build_agent_previews(summary, branch, source_cards)

    assert summary.iloc[0]["policy_status"] == "observe_too_few_cases"
    assert previews
    text = str(previews)
    assert "return_20d" not in text
    assert "delta_cash" not in text
    for preview in previews:
        assert_no_future_fields(preview)


def test_classify_policy_weak_source_blocks_default_use() -> None:
    status, use = classify_policy(
        {
            "source_status": "needs_grounding",
            "rows": 100,
            "unique_stocks": 80,
            "active_exposure_cards": 0,
            "avg_delta_cash": 0.2,
            "lowered_negative": 10,
            "lowered_positive": 0,
            "raised_negative": 0,
        }
    )

    assert status == "weak_until_grounded"
    assert use == "source_reference_only"


def test_assert_no_future_fields_rejects_result_key() -> None:
    with pytest.raises(ValueError):
        assert_no_future_fields({"avg_delta_cash": 1.0})
