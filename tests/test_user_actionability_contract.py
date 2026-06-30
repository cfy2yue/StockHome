from __future__ import annotations

import scripts.audit_user_actionability_contract as audit


def test_p0_contract_requires_operation_position_and_triggers() -> None:
    card = {
        "variant": "full_agent",
        "decision_date": "2026-01-02",
        "code": "000001",
        "name": "平安银行",
        "user_operation_suggestion": "放入观察",
        "target_position": 0.1,
        "position_plan": "小仓10%",
        "buy_or_add_trigger": "放量突破20日均线",
        "reduce_or_sell_trigger": "跌破前低",
        "review_condition": "下周复查",
        "counter_evidence": "同行弱",
        "final_agent_reasoning_summary": "软缺口降置信",
    }

    result = audit.audit_p0_card("x", audit.REPORT_DIR / "x.jsonl", card)

    assert result["actionability_status"] == "fail"
    assert "missing_or_vague_operation" in result["issues"]


def test_p0_contract_accepts_clear_small_entry_plan() -> None:
    card = {
        "variant": "full_agent",
        "decision_date": "2026-01-02",
        "code": "000001",
        "name": "平安银行",
        "user_operation_suggestion": "试探买入/持有",
        "target_position": 0.2,
        "position_plan": "新仓10%-20%试探，上限30%",
        "buy_or_add_trigger": "保持强于同行且新闻预警<0.5",
        "reduce_or_sell_trigger": "跌破前低或财务风险>=0.6",
        "review_condition": "每两周复查",
        "counter_evidence": "新闻空窗",
        "final_agent_reasoning_summary": "无硬反证，低仓位试探",
    }

    result = audit.audit_p0_card("x", audit.REPORT_DIR / "x.jsonl", card)

    assert result["actionability_status"] == "pass"


def test_p0_reduce_review_accepts_reentry_condition_in_position_plan() -> None:
    card = {
        "variant": "full_agent",
        "decision_date": "2026-01-02",
        "code": "000001",
        "name": "平安银行",
        "user_operation_suggestion": "减仓/卖出复核",
        "target_position": 0.05,
        "position_plan": "减仓至20%仓位；若出现反转信号且同行改善可重新评估",
        "buy_or_add_trigger": "不适用",
        "reduce_or_sell_trigger": "硬反证持续",
        "review_condition": "每两周复查",
        "counter_evidence": "同行弱",
        "final_agent_reasoning_summary": "硬反证复核",
    }

    result = audit.audit_p0_card("x", audit.REPORT_DIR / "x.jsonl", card)

    assert result["actionability_status"] == "pass"


def test_p1_top2_wait_without_hard_counter_fails() -> None:
    card = {"variant": "ranker_anchor_agent", "decision_date": "2026-01-02"}
    item = {
        "rank": 1,
        "code": "000001",
        "name": "平安银行",
        "operation_recommendation": "等待",
        "position_threshold": "新仓0%",
        "buy_or_add_trigger": "突破20日均线",
        "reduce_or_sell_trigger": "跌破前低",
        "research_grade": "继续深挖",
        "priority_reason": "默认Top1",
        "counter_evidence": "轻度同行弱",
    }

    result = audit.audit_p1_candidate("p1", audit.REPORT_DIR / "p1.jsonl", card, item)

    assert result["actionability_status"] == "fail"
    assert "top2_non_actionable_without_hard_counter" in result["issues"]


def test_p1_top2_small_trial_passes() -> None:
    card = {"variant": "ranker_anchor_agent", "decision_date": "2026-01-02"}
    item = {
        "rank": 2,
        "code": "000001",
        "name": "平安银行",
        "operation_recommendation": "条件化试探买入/继续持有",
        "position_threshold": "新仓5%-10%试探，上限20%",
        "buy_or_add_trigger": "保持Top2且新闻预警<0.5",
        "reduce_or_sell_trigger": "跌出Top2或财务风险>=0.6",
        "research_grade": "放入观察",
        "priority_reason": "默认Top2",
        "counter_evidence": "财报空窗",
    }

    result = audit.audit_p1_candidate("p1", audit.REPORT_DIR / "p1.jsonl", card, item)

    assert result["actionability_status"] == "pass"


def test_p1_normalization_fills_not_applicable_triggers() -> None:
    card = {"variant": "ranker_anchor_agent", "decision_date": "2026-01-02"}
    item = {
        "rank": 4,
        "code": "000001",
        "name": "平安银行",
        "operation_recommendation": "等待",
        "position_threshold": "新仓0%",
        "buy_or_add_trigger": "不适用",
        "reduce_or_sell_trigger": "不适用",
        "research_grade": "信息不足",
        "priority_reason": "关键数据缺失",
        "counter_evidence": "财报缺失",
    }

    result = audit.audit_p1_candidate(
        "p1",
        audit.REPORT_DIR / "p1.jsonl",
        card,
        item,
        normalize_for_user=True,
    )

    assert result["actionability_status"] == "pass"
    assert "buy_trigger_from_grade" in result["normalization_notes"]


def test_eval_fields_are_rejected() -> None:
    card = {
        "variant": "full_agent",
        "user_operation_suggestion": "试探买入",
        "target_position": 0.2,
        "position_plan": "新仓10%",
        "buy_or_add_trigger": "突破",
        "reduce_or_sell_trigger": "跌破",
        "review_condition": "复查",
        "counter_evidence": "无",
        "final_agent_reasoning_summary": "摘要",
        "return_20d": 1.0,
    }

    result = audit.audit_p0_card("x", audit.REPORT_DIR / "x.jsonl", card)

    assert result["actionability_status"] == "fail"
    assert "eval_field_present" in result["issues"]
