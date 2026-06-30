from __future__ import annotations

import json

import pytest

from src.agent_training.quant_tool_context import load_quant_tool_summaries, quant_tool_summary_text, select_quant_tool_summaries


def test_load_quant_tool_summaries_filters_task_and_future_keys(tmp_path) -> None:
    path = tmp_path / "outcomes.jsonl"
    rows = [
        {
            "tool_id": "portfolio_tool",
            "tool_version": "v1",
            "task_mode": "portfolio_pool_optimize",
            "feature_group": "price_core",
            "selection_mode": "score_threshold",
            "score": 0.66,
            "confidence": 0.4,
            "usable_in_agent_default": False,
            "promotion_status": "observe_latest_block_failed",
            "top_features": ["prior_return_20d", "future_return_20d"],
            "counter_evidence": ["latest_time_block_failed"],
        },
        {
            "tool_id": "single_tool",
            "task_mode": "single_stock_watch",
            "score": 0.7,
            "confidence": 0.3,
            "usable_in_agent_default": False,
            "promotion_status": "observe_unstable_across_time",
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    summaries = load_quant_tool_summaries(path, task_mode="portfolio_pool", max_items=3)

    assert len(summaries) == 1
    assert summaries[0]["tool_id"] == "portfolio_tool"
    assert summaries[0]["research_only"] is True
    assert summaries[0]["not_investment_instruction"] is True
    assert "future_return_20d" not in summaries[0]["top_features"]
    assert "portfolio_tool" in quant_tool_summary_text(summaries)
    assert "latest_time_block_failed" in quant_tool_summary_text(summaries)


def test_load_quant_tool_summaries_rejects_exact_future_result_key(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"tool_id": "bad", "return_20d": 0.5}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="future result field leaked"):
        load_quant_tool_summaries(path)


def test_select_quant_tool_summaries_prefers_usable_then_observe() -> None:
    rows = [
        {"tool_id": "reject", "task_mode": "single_stock_watch", "usable_in_agent_default": False, "promotion_status": "reject_too_few_samples", "confidence": 0.9, "score": 0.9},
        {"tool_id": "observe", "task_mode": "single_stock_watch", "usable_in_agent_default": False, "promotion_status": "observe_latest_block_failed", "confidence": 0.3, "score": 0.7},
        {"tool_id": "accepted", "task_mode": "single_stock_watch", "usable_in_agent_default": True, "promotion_status": "accepted", "confidence": 0.2, "score": 0.6},
    ]

    selected = select_quant_tool_summaries(rows, task_mode="single_stock", max_items=2)

    assert [row["tool_id"] for row in selected] == ["accepted", "observe"]
