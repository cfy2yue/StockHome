from __future__ import annotations

from scripts.build_bookskill_attribution_report import aggregate_bookskill_attribution, build_bookskill_attribution_rows


def test_bookskill_attribution_rows_join_cards_and_skills() -> None:
    evidence = [
        {
            "agent_policy_version": "p1",
            "variant": "full_agent_with_quant_tools",
            "step": 1,
            "valid_block": "H2025_1",
            "decision_date": "2025-01-03",
            "code": "1",
            "name": "测试",
            "task_mode": "portfolio_pool",
            "sample_panel_id": "panel_02",
            "book_skill_candidates": [
                {
                    "strategy_id": "PPS-Q-017",
                    "source_book": "专业投机原理",
                    "source_status": "grounded",
                    "page_range": "OCR_PAGE 1-2",
                    "confidence": "medium",
                }
            ],
        },
        {
            "agent_policy_version": "p1",
            "variant": "python_only",
            "step": 1,
            "valid_block": "H2025_1",
            "decision_date": "2025-01-03",
            "code": "2",
            "name": "测试2",
            "task_mode": "portfolio_pool",
            "sample_panel_id": "panel_01",
            "book_skill_candidates": [],
        },
    ]
    cards = [
        {
            "agent_policy_version": "p1",
            "variant": "full_agent_with_quant_tools",
            "step": 1,
            "valid_block": "H2025_1",
            "decision_date": "2025-01-03",
            "code": "000001",
            "task_mode": "portfolio_pool",
            "sample_panel_id": "panel_02",
            "research_grade": "放入观察",
            "simulated_action": "保持观察",
            "simulated_weight_change": 0.1,
        },
        {
            "agent_policy_version": "p1",
            "variant": "python_only",
            "step": 1,
            "valid_block": "H2025_1",
            "decision_date": "2025-01-03",
            "code": "000002",
            "task_mode": "portfolio_pool",
            "sample_panel_id": "panel_01",
            "research_grade": "继续深挖",
            "simulated_action": "增加研究暴露",
            "simulated_weight_change": 0.6,
        },
    ]

    rows = build_bookskill_attribution_rows(evidence, cards)
    aggregate = aggregate_bookskill_attribution(rows)

    assert set(rows["strategy_id"]) == {"PPS-Q-017", "NO_BOOKSKILL"}
    grounded = rows[rows["strategy_id"].eq("PPS-Q-017")].iloc[0]
    assert bool(grounded["has_grounded_source_detail"])
    no_skill = rows[rows["strategy_id"].eq("NO_BOOKSKILL")].iloc[0]
    assert bool(no_skill["active_exposure"])
    assert int(aggregate[aggregate["strategy_id"].eq("NO_BOOKSKILL")]["active_exposure_cards"].iloc[0]) == 1
