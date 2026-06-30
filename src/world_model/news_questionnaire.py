from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_QUESTIONNAIRE_PATH = Path("config/news_deepseek_questionnaire.yaml")
DERIVED_SCORE_RANGES = {
    "ds_news_risk_score": (0.0, 1.0),
    "ds_news_opportunity_score": (0.0, 1.0),
    "ds_news_peer_support_score": (-1.0, 1.0),
    "ds_news_policy_support_score": (-1.0, 1.0),
    "ds_news_region_support_score": (-1.0, 1.0),
    "ds_news_uncertainty_score": (0.0, 1.0),
    "ds_news_quality_score": (0.0, 1.0),
    "ds_news_net_score": (-1.0, 1.0),
}


def load_news_questionnaire(path: str | Path = DEFAULT_QUESTIONNAIRE_PATH) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid news questionnaire config: {config_path}")
    return data


def questionnaire_output_fields(config: dict[str, Any]) -> list[str]:
    questions = config.get("questions") if isinstance(config.get("questions"), list) else []
    fields = []
    for item in questions:
        if isinstance(item, dict) and item.get("output_field"):
            fields.append(str(item["output_field"]))
    return fields


def questionnaire_derived_score_fields(config: dict[str, Any]) -> list[str]:
    schema = config.get("required_output_schema") if isinstance(config.get("required_output_schema"), dict) else {}
    derived = schema.get("derived_scores") if isinstance(schema.get("derived_scores"), dict) else {}
    return [str(field) for field in derived.keys()]


def compact_news_questionnaire_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep the API prompt compact while preserving the scoring contract."""
    schema = config.get("required_output_schema") if isinstance(config.get("required_output_schema"), dict) else {}
    time_safety = config.get("time_safety") if isinstance(config.get("time_safety"), dict) else {}
    questions = config.get("questions") if isinstance(config.get("questions"), list) else []
    return {
        "news_deepseek_questionnaire_version": config.get("news_deepseek_questionnaire_version"),
        "research_only": True,
        "no_auto_execution": True,
        "time_safety": {
            "leakage_guard": time_safety.get("leakage_guard"),
            "decision_time_default": time_safety.get("decision_time_default"),
            "date_only_policy": time_safety.get("date_only_policy"),
        },
        "score_ranges": {
            "signed_relevance": "-2..2",
            "probability_like": "0..1",
            "derived_scores": DERIVED_SCORE_RANGES,
        },
        "required_output_schema": {
            "type": schema.get("type"),
            "required_fields": schema.get("required_fields"),
            "derived_scores": schema.get("derived_scores"),
            "formula_guidance": schema.get("formula_guidance"),
        },
        "questions": [
            {
                "id": item.get("id"),
                "output_field": item.get("output_field"),
                "category": item.get("category"),
                "score_type": item.get("score_type"),
                "question": item.get("question"),
                "scoring_hint": item.get("scoring_hint"),
            }
            for item in questions
            if isinstance(item, dict)
        ],
    }


def build_news_questionnaire_messages(
    *,
    questionnaire_config: dict[str, Any],
    evidence: dict[str, Any],
) -> list[dict[str, str]]:
    """Build JSON-only DeepSeek messages for the semantic news questionnaire."""
    system = (
        "你是A股研究辅助系统中的新闻语义分类Agent。"
        "你只能基于evidence中available_at不晚于decision_time的材料回答。"
        "不能使用未来收益、未来新闻或后验标签。"
        "本工具只做新闻语义量化，不自动下单，不承诺收益；不得输出强烈推荐、目标价必达、稳赚、必涨等强承诺。"
        "必须输出严格JSON对象，不要markdown。"
        "answers字段必须是JSON对象，键为output_field，值为数字；禁止把answers写成数组，禁止逐题解释。"
        "如需解释，只能在key_reasons中写最多6条短理由。"
    )
    payload = {
        "task": "阅读新闻/公告材料并按固定问卷输出语义量化结果。",
        "questionnaire": compact_news_questionnaire_config(questionnaire_config),
        "evidence": evidence,
        "output_contract": compact_news_questionnaire_config(questionnaire_config).get("required_output_schema", {}),
        "response_style": {
            "answers": "用JSON对象，键为32个output_field，值只填数字分数；不要为每题逐条写reason。",
            "key_reasons": "最多6条，只解释最高风险、最高机会、最大不确定性和冲突来源。",
            "reason_limit": "每条 key_reasons 不超过 24 个汉字。",
            "summary_limit": "mainline_summary 不超过 80 个汉字。",
            "strict_json": True,
            "forbidden": "禁止 answers 数组；禁止每题 reason；禁止 markdown。",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
    ]


def validate_news_questionnaire_result(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    expected_version = str(config.get("news_deepseek_questionnaire_version", ""))
    required = config.get("required_output_schema", {}).get("required_fields", [])
    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError(f"questionnaire result missing fields: {missing}")
    if result.get("type") != "news_semantic_questionnaire_result":
        raise ValueError(f"invalid questionnaire result type: {result.get('type')}")
    if str(result.get("questionnaire_version")) != expected_version:
        raise ValueError(f"invalid questionnaire version: {result.get('questionnaire_version')}")
    if result.get("no_investment_instruction") is not True:
        raise ValueError("questionnaire result must set no_investment_instruction=true")
    derived = result.get("derived_scores")
    if not isinstance(derived, dict):
        raise ValueError("derived_scores must be an object")
    for field in questionnaire_derived_score_fields(config):
        if field not in derived:
            raise ValueError(f"derived_scores missing {field}")
        value = _coerce_float(derived[field], field)
        lower, upper = DERIVED_SCORE_RANGES.get(field, (0.0, 1.0))
        if value < lower or value > upper:
            raise ValueError(f"{field} out of range: {value}")
        derived[field] = round(value, 4)
    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("answers must be an object keyed by output_field")
    question_by_field = {
        str(item["output_field"]): item
        for item in config.get("questions", [])
        if isinstance(item, dict) and item.get("output_field")
    }
    missing_answers = [field for field in question_by_field if field not in answers]
    if missing_answers:
        raise ValueError(f"answers missing fields: {missing_answers}")
    for field, item in question_by_field.items():
        value = _coerce_float(answers[field], field)
        score_type = str(item.get("score_type", "probability_like"))
        lower, upper = (-2.0, 2.0) if score_type == "signed_relevance" else (0.0, 1.0)
        if value < lower or value > upper:
            raise ValueError(f"{field} out of range: {value}")
        answers[field] = round(value, 4)
    return result


def flatten_news_questionnaire_result(result: dict[str, Any]) -> dict[str, Any]:
    derived = result.get("derived_scores") if isinstance(result.get("derived_scores"), dict) else {}
    row = {
        "type": result.get("type"),
        "questionnaire_version": result.get("questionnaire_version"),
        "code": result.get("code"),
        "decision_date": result.get("decision_date"),
        "decision_time": result.get("decision_time"),
        "mainline_summary": result.get("mainline_summary"),
        "missing_or_conflict_notes": result.get("missing_or_conflict_notes"),
        "no_investment_instruction": result.get("no_investment_instruction"),
    }
    coverage = result.get("source_coverage")
    if isinstance(coverage, dict):
        for key, value in coverage.items():
            row[f"source_coverage_{key}"] = value
    answers = result.get("answers")
    if isinstance(answers, dict):
        row.update(answers)
    row.update(derived)
    return row


def _coerce_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric: {value}") from exc
