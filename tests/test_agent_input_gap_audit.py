from __future__ import annotations

import json

from scripts.audit_agent_input_gap import audit_agent_input_gap


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_agent_input_gap_audit_summarizes_adoption_reasons(tmp_path) -> None:
    evidence = [
        {
            "variant": "full_agent_with_quant_tools",
            "task_mode": "portfolio_pool",
            "valid_block": "H2026_1",
            "decision_date": "2026-01-02",
            "code": "1",
            "sample_panel_id": "panel_01",
            "quant_tool_summaries": [{"promotion_status": "accepted_cost_recheck_candidate"}],
            "news_features": {"news_count_30d": 0, "news_missing_rate": 1.0},
            "news_semantic_questionnaire": {"ds_news_risk_score": None},
            "financial_report_features": {
                "financial_report_event_count": 0,
                "financial_report_missing_rate": 1.0,
                "financial_report_join_status": "code_not_in_feature_table",
            },
            "book_skill_candidates": [{"strategy_id": "PPS-Q-017", "source_book": "专业投机原理", "source_status": "grounded"}],
            "peer_context_features": {"tushare_industry_positive_breadth_20d": 0.4, "tushare_industry_relative_return_20d": -3},
            "chip_features": {"upper_overhang": 1.3},
            "retrieved_cases_context": "none",
            "data_missing_flags": "financial_publish_date_missing_or_unavailable",
        },
        {
            "variant": "full_agent_with_quant_tools",
            "task_mode": "portfolio_pool",
            "valid_block": "H2026_1",
            "decision_date": "2026-01-16",
            "code": "2",
            "sample_panel_id": "panel_01",
            "quant_tool_summaries": [{"promotion_status": "accepted_cost_recheck_candidate"}],
            "news_features": {"news_count_30d": 2},
            "news_semantic_questionnaire": {"ds_news_opportunity_score": 0.7},
            "financial_report_features": {
                "financial_report_event_count": 1,
                "financial_report_available_at": "2026-01-15 15:00",
                "financial_report_missing_rate": 0.0,
            },
            "book_skill_candidates": [],
            "peer_context_features": {"tushare_industry_positive_breadth_20d": 0.6, "tushare_industry_relative_return_20d": 1.2},
            "chip_features": {"upper_overhang": 0.4},
            "retrieved_cases_context": "case: safe prior",
            "data_missing_flags": "none",
        },
    ]
    decisions = [
        {
            "variant": "full_agent_with_quant_tools",
            "task_mode": "portfolio_pool",
            "valid_block": "H2026_1",
            "decision_date": "2026-01-02",
            "code": "000001",
            "sample_panel_id": "panel_01",
            "simulated_action": "保持观察",
            "quant_tool_adoption_decision": "not_adopted_counter_evidence",
            "quant_tool_override_reasons": "news_gap;financial_gap;chip_overhang;data_missing",
        },
        {
            "variant": "full_agent_with_quant_tools",
            "task_mode": "portfolio_pool",
            "valid_block": "H2026_1",
            "decision_date": "2026-01-16",
            "code": "000002",
            "sample_panel_id": "panel_01",
            "simulated_action": "保持观察",
            "quant_tool_adoption_decision": "partially_adopted",
            "quant_tool_override_reasons": "none",
        },
    ]
    evidence_path = tmp_path / "evidence.jsonl"
    decision_path = tmp_path / "decisions.jsonl"
    _write_jsonl(evidence_path, evidence)
    _write_jsonl(decision_path, decisions)

    detail, summary = audit_agent_input_gap(evidence_path, decision_path)

    assert len(detail) == 2
    assert int(detail["accepted_tool_available"].sum()) == 2
    assert int(detail["actionable_news_available"].sum()) == 1
    assert int(detail["financial_event_asof_available"].sum()) == 1
    assert int(detail["grounded_bookskill_available"].sum()) == 1
    row = summary.iloc[0]
    assert row["cards"] == 2
    assert row["adoption_not_adopted_counter_evidence"] == 1
    assert row["adoption_partially_adopted"] == 1
    assert row["reason_news_gap_rate"] == 0.5
    assert row["chip_overhang_rate"] == 0.5
