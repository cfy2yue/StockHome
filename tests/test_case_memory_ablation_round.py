from __future__ import annotations

import pandas as pd

from scripts.run_deepseek_news_ablation_round import _apply_case_memory_modes, _planned_metrics


def _pack() -> dict:
    return {
        "type": "agent_evidence_pack",
        "agent_policy_version": "test",
        "variant": "no_news",
        "step": 1,
        "train_blocks": "H2023_1",
        "valid_block": "H2025_1",
        "decision_date": "2025-05-20",
        "code": "000628",
        "name": "高新发展",
        "task_mode": "portfolio_pool",
        "python_signal_summary": "relative_strength_rank=0.8975; prior_return_20d=16.78",
        "news_signal_summary": "news ablation: no news fields visible",
        "financial_report_signal_summary": "financial report channel visible",
        "book_skill_candidates": [],
        "counter_evidence": "新闻/财报/Book Skill确认不足",
        "data_missing_flags": "Book Skill未解析",
        "memory_context": "compact memory",
        "research_only": True,
        "not_investment_instruction": True,
    }


def test_case_memory_modes_expand_without_merging() -> None:
    packs = _apply_case_memory_modes(
        [_pack()],
        ["no_rag", "memory_compact_only", "retrieved_cases_v1", "retrieved_cases_v2_applicable"],
        top_k=2,
    )

    assert len(packs) == 4
    by_mode = {pack["case_memory_mode"]: pack for pack in packs}
    assert by_mode["no_rag"]["memory_context"] == "none"
    assert by_mode["no_rag"]["retrieved_cases_context"] == "none"
    assert by_mode["memory_compact_only"]["memory_context"] == "compact memory"
    assert by_mode["memory_compact_only"]["retrieved_cases_context"] == "none"
    assert by_mode["retrieved_cases_v1"]["memory_context"] == "compact memory"
    assert "retrieved_cases:" in by_mode["retrieved_cases_v1"]["retrieved_cases_context"]
    assert by_mode["retrieved_cases_v2_applicable"]["memory_context"] == "compact memory"
    assert "retrieved_cases_applicability" in by_mode["retrieved_cases_v2_applicable"]["retrieved_cases_context"]

    planned = _planned_metrics(packs)
    assert set(planned["case_memory_mode"]) == {"no_rag", "memory_compact_only", "retrieved_cases_v1", "retrieved_cases_v2_applicable"}
    assert int(pd.to_numeric(planned["planned_evidence_packs"], errors="coerce").sum()) == 4


def test_case_memory_retrieved_context_excludes_future_field_names() -> None:
    packs = _apply_case_memory_modes([_pack()], ["retrieved_cases_v1"], top_k=5)
    text = packs[0]["retrieved_cases_context"]

    assert "return_20d" not in text
    assert "gt_status" not in text
    assert "future_return" not in text
