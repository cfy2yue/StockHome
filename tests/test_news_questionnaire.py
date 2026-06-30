from __future__ import annotations

import json

from src.world_model.news_questionnaire import (
    build_news_questionnaire_messages,
    compact_news_questionnaire_config,
    flatten_news_questionnaire_result,
    load_news_questionnaire,
    questionnaire_derived_score_fields,
    questionnaire_output_fields,
    validate_news_questionnaire_result,
)


def test_news_questionnaire_config_has_32_questions_and_scores() -> None:
    config = load_news_questionnaire()
    fields = questionnaire_output_fields(config)
    derived = questionnaire_derived_score_fields(config)
    assert config["news_deepseek_questionnaire_version"] == "news_semantic_questionnaire_v1"
    assert len(fields) == 32
    assert len(set(fields)) == 32
    assert "ds_news_risk_score" in derived
    assert "ds_news_net_score" in derived


def test_compact_questionnaire_keeps_contract_without_verbose_sections() -> None:
    config = load_news_questionnaire()
    compact = compact_news_questionnaire_config(config)
    assert len(compact["questions"]) == 32
    assert "input_pack_policy" not in compact
    assert "score_scale" not in compact
    assert compact["required_output_schema"]["derived_scores"]


def test_news_questionnaire_messages_are_json_only_and_time_safe() -> None:
    config = load_news_questionnaire()
    messages = build_news_questionnaire_messages(
        questionnaire_config=config,
        evidence={
            "code": "000001",
            "decision_date": "2026-01-05",
            "decision_time": "15:00:00",
            "events": [
                {
                    "available_at": "2026-01-05 14:30:00",
                    "source_type": "official_disclosure",
                    "title": "测试公告",
                }
            ],
        },
    )
    assert "严格JSON对象" in messages[0]["content"]
    assert "不自动下单" in messages[0]["content"]
    assert "目标价必达" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["questionnaire"]["time_safety"]["leakage_guard"]
    assert payload["evidence"]["code"] == "000001"
    assert "键为32个output_field" in payload["response_style"]["answers"]
    assert "禁止 answers 数组" in payload["response_style"]["forbidden"]
    assert "answers字段必须是JSON对象" in messages[0]["content"]


def test_validate_and_flatten_questionnaire_result() -> None:
    config = load_news_questionnaire()
    result = {
        "type": "news_semantic_questionnaire_result",
        "questionnaire_version": "news_semantic_questionnaire_v1",
        "code": "000001",
        "decision_date": "2026-01-05",
        "decision_time": "15:00:00",
        "source_coverage": {"self": 1, "peer": 0},
        "mainline_summary": "公告证据有限，主线不强。",
        "answers": {field: 0 for field in questionnaire_output_fields(config)},
        "derived_scores": {
            "ds_news_risk_score": 0.2,
            "ds_news_opportunity_score": 0.1,
            "ds_news_peer_support_score": -0.1,
            "ds_news_policy_support_score": 0,
            "ds_news_region_support_score": 0,
            "ds_news_uncertainty_score": 0.7,
            "ds_news_quality_score": 0.4,
            "ds_news_net_score": -0.45,
        },
        "missing_or_conflict_notes": "同行和地域材料不足。",
        "no_investment_instruction": True,
    }
    validated = validate_news_questionnaire_result(result, config)
    flat = flatten_news_questionnaire_result(validated)
    assert flat["ds_news_net_score"] == -0.45
    assert flat["ds_news_mainline_clarity"] == 0
    assert flat["source_coverage_self"] == 1
