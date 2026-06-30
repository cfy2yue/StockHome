from __future__ import annotations

from typing import Any


REQUIRED_DECISION_CARD_FIELDS = [
    "type",
    "agent_policy_version",
    "variant",
    "step",
    "train_blocks",
    "valid_block",
    "decision_date",
    "code",
    "name",
    "task_mode",
    "research_grade",
    "simulated_action",
    "simulated_weight_change",
    "python_signal_summary",
    "news_signal_summary",
    "book_skill_evidence",
    "memory_experience_used",
    "counter_evidence",
    "accepted_quant_tool_ids",
    "quant_tool_adoption_decision",
    "quant_tool_override_reasons",
    "final_agent_reasoning_summary",
    "confidence_level",
    "data_missing_flags",
    "error_reflection",
    "research_only",
    "not_investment_instruction",
]


ALLOWED_RESEARCH_GRADES = {"继续深挖", "放入观察", "暂时剔除", "信息不足"}
ALLOWED_SIMULATED_ACTIONS = {"增加研究暴露", "降低研究暴露", "保持观察", "转入现金", "信息不足不动作"}
ALLOWED_QUANT_TOOL_ADOPTION_DECISIONS = {
    "adopted",
    "partially_adopted",
    "not_adopted_counter_evidence",
    "not_applicable",
}
ACTION_WEIGHT_BOUNDS = {
    "增加研究暴露": (0.50, 1.00),
    "保持观察": (0.00, 0.20),
    "降低研究暴露": (0.00, 0.10),
    "转入现金": (0.00, 0.00),
    "信息不足不动作": (0.00, 0.00),
}
INVESTMENT_INSTRUCTION_REPLACEMENTS = {
    "强烈推荐": "研究优先级较高",
    "目标价必达": "目标结果不确定",
    "稳赚": "收益不确定",
    "必涨": "方向不确定",
}
USER_FACING_DECISION_TEXT_FIELDS = [
    "python_signal_summary",
    "kline_signal_summary",
    "news_signal_summary",
    "book_skill_evidence",
    "memory_experience_used",
    "counter_evidence",
    "accepted_quant_tool_ids",
    "quant_tool_override_reasons",
    "final_agent_reasoning_summary",
    "user_operation_suggestion",
    "position_plan",
    "buy_or_add_trigger",
    "reduce_or_sell_trigger",
    "review_condition",
    "data_missing_flags",
    "error_reflection",
]


def validate_decision_card(card: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_DECISION_CARD_FIELDS if field not in card]
    if missing:
        raise ValueError(f"decision card missing fields: {missing}")
    if card["research_grade"] not in ALLOWED_RESEARCH_GRADES:
        raise ValueError(f"invalid research_grade: {card['research_grade']}")
    if card["simulated_action"] not in ALLOWED_SIMULATED_ACTIONS:
        raise ValueError(f"invalid simulated_action: {card['simulated_action']}")
    try:
        weight = float(card["simulated_weight_change"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid simulated_weight_change: {card['simulated_weight_change']}") from exc
    if weight < 0 or weight > 1:
        raise ValueError(f"simulated_weight_change out of range: {card['simulated_weight_change']}")
    weight = normalize_action_weight(str(card["simulated_action"]), weight)
    try:
        confidence = float(card["confidence_level"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid confidence_level: {card['confidence_level']}") from exc
    if confidence < 0 or confidence > 1:
        raise ValueError(f"confidence_level out of range: {card['confidence_level']}")
    if not bool(card["research_only"]) or not bool(card["not_investment_instruction"]):
        raise ValueError("decision card must be research_only and not_investment_instruction")
    if card["quant_tool_adoption_decision"] not in ALLOWED_QUANT_TOOL_ADOPTION_DECISIONS:
        raise ValueError(f"invalid quant_tool_adoption_decision: {card['quant_tool_adoption_decision']}")
    card["simulated_weight_change"] = weight
    card["confidence_level"] = confidence
    card["accepted_quant_tool_ids"] = _normalize_text_field(card.get("accepted_quant_tool_ids"))
    card["quant_tool_override_reasons"] = _normalize_text_field(card.get("quant_tool_override_reasons"))
    sanitize_decision_card_text_fields(card)
    return card


def normalize_action_weight(action: str, weight: float) -> float:
    lower, upper = ACTION_WEIGHT_BOUNDS.get(action, (0.0, 1.0))
    return max(lower, min(upper, float(weight)))


def _normalize_text_field(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item).strip() for item in value if str(item).strip()) or "none"
    if value is None:
        return "none"
    text = str(value).strip()
    return text or "none"


def sanitize_investment_instruction_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value
    for source, replacement in INVESTMENT_INSTRUCTION_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text


def sanitize_decision_card_text_fields(card: dict[str, Any]) -> dict[str, Any]:
    for field in USER_FACING_DECISION_TEXT_FIELDS:
        if field in card:
            card[field] = sanitize_investment_instruction_text(card[field])
    return card
