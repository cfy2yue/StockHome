from __future__ import annotations

import pandas as pd

from scripts.audit_single_stock_rag_bookskill_attribution import (
    build_attribution_detail,
    summarize_branch_bookskill,
    summarize_by_variant,
    summarize_exploded,
)


def test_rag_bookskill_attribution_pairs_control_and_extracts_ids() -> None:
    pack = {
        "variant": "full_agent_with_opportunity_tool",
        "task_mode": "single_stock",
        "valid_block": "H2026_1",
        "decision_date": "2026-01-06",
        "code": "1",
        "sample_panel_id": "panel_a",
        "case_memory_mode": "retrieved_cases_v2_applicable",
        "retrieved_cases_context": (
            "retrieved_cases_applicability:\n"
            "- FAIL-X | ledger=memory/failure_case_ledger.csv | applicability=applicable | guidance=checklist\n"
            "- EXP-Y | ledger=memory/strategy_experience_ledger.csv | applicability=partial | guidance=observe\n"
        ),
        "book_skill_candidates": [
            {"strategy_id": "PPS-Q-017", "source_book": "book"},
            {"strategy_id": "__truncated__"},
        ],
        "quant_tool_summaries": [
            {
                "tool_id": "single_stock_risk_calibration_v2_review_queue",
                "primary_risk_branch": "overheat_reversal_friction_without_hard_event",
                "risk_tier": "hard_counter_yellow_review_0.80_0.95",
            },
            {
                "tool_id": "single_stock_opportunity_scorer_v2",
                "policy_status": "green_candidate",
                "score": 0.2,
            },
        ],
    }
    card = {
        "variant": "full_agent_with_opportunity_tool",
        "task_mode": "single_stock",
        "valid_block": "H2026_1",
        "decision_date": "2026-01-06",
        "code": "000001",
        "sample_panel_id": "panel_a",
        "name": "测试",
        "research_grade": "放入观察",
        "simulated_action": "保持观察",
        "simulated_weight_change": 0.0,
        "memory_experience_used": "FAIL-X",
        "counter_evidence": ["case"],
    }
    control = {
        **card,
        "simulated_weight_change": 0.1,
        "simulated_action": "保持观察",
    }
    gt = pd.DataFrame([{"date": "2026-01-06", "code": "000001", "return_20d": -10.0}])

    detail = build_attribution_detail([pack], [card], gt, control_cards=[control])
    row = detail.iloc[0]

    assert bool(row["rag_useful_context"]) is True
    assert row["rag_applicable_count"] == 1
    assert row["rag_partial_count"] == 1
    assert row["rag_case_ids"] == "FAIL-X;EXP-Y"
    assert row["bookskill_strategy_ids"] == "PPS-Q-017"
    assert row["primary_risk_branch"] == "overheat_reversal_friction_without_hard_event"
    assert row["paired_change_type"] == "lowered_negative"
    assert row["delta_cash"] > 0

    variant = summarize_by_variant(detail)
    assert variant.iloc[0]["changed_weight_rows"] == 1
    assert variant.iloc[0]["lowered_negative"] == 1

    rag_cases = summarize_exploded(detail, "rag_case_ids")
    assert set(rag_cases["item_id"]) == {"FAIL-X", "EXP-Y"}

    bookskill = summarize_exploded(detail, "bookskill_strategy_ids")
    assert list(bookskill["item_id"]) == ["PPS-Q-017"]

    branch_bookskill = summarize_branch_bookskill(detail)
    assert branch_bookskill.iloc[0]["branch"] == "overheat_reversal_friction_without_hard_event"
    assert branch_bookskill.iloc[0]["strategy_id"] == "PPS-Q-017"
