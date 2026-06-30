from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.agent_training.deepseek_runner import DeepSeekRunResult, decide_evidence_packs
from src.agent_training.deepseek_client import BACKTEST_TRAINING_MODEL, model_concurrency_limit
from src.agent_training.evidence_pack import build_decision_messages, build_evidence_pack
from src.agent_training.preflight import run_preflight, write_preflight_reports


def _sample_row() -> pd.Series:
    return pd.Series(
        {
            "date": "2023-07-04",
            "code": "1",
            "name": "测试股票",
            "industry": "测试行业",
            "timeline_score": 8.2,
            "relative_strength_rank": 0.81,
            "prior_return_20d": 3.4,
            "news_count_30d": 2,
            "news_warning_score_30d": 0,
            "news_missing_rate": 0.2,
            "self_news_intensity": 0.3,
            "peer_news_intensity": 0.6,
            "policy_background_score": 0.1,
            "region_background_score": -0.1,
            "self_vs_peer_attention_gap": -0.3,
            "peer_active_self_silent_flag": 1,
            "news_warning_score": 0.4,
            "news_opportunity_score": 0.2,
            "news_evidence_quality": 0.7,
            "news_timestamp_quality": 1.0,
            "news_peer_diffusion_score": -0.2,
            "official_confirmation_score": 0.5,
            "community_attention_score": 0.8,
            "community_crowding_risk": 0.6,
            "announcement_materiality_score": 0.9,
            "news_semantic_questionnaire_version": "news_semantic_questionnaire_v1",
            "ds_news_mainline_summary": "行业主线清楚但自身证据偏弱",
            "ds_news_mainline_clarity": 0.7,
            "ds_news_peer_relative_support": -1,
            "ds_news_conflict_intensity": 0.6,
            "ds_news_risk_score": 0.35,
            "ds_news_opportunity_score": 0.44,
            "ds_news_peer_support_score": 0.2,
            "ds_news_policy_support_score": 0.1,
            "ds_news_region_support_score": 0.0,
            "ds_news_uncertainty_score": 0.5,
            "ds_news_quality_score": 0.7,
            "ds_news_net_score": -0.16,
            "ds_news_missing_or_conflict_notes": "同行活跃但自身证据不足",
            "financial_report_event_count": 1,
            "financial_report_materiality_score": 0.8,
            "financial_quality_risk_score": 0.65,
            "financial_surprise_score": -0.4,
            "financial_disclosure_quality_score": 0.7,
            "financial_report_missing_rate": 0.1,
            "financial_report_latest_period": "20230331",
            "financial_report_event_types": "quarterly_report",
            "financial_report_available_at": "2023-04-29 00:00:00",
            "financial_report_join_status": "event_window_matched",
            "quant_tool_summaries": [
                {
                    "tool_id": "date_regime_gate_minimal_v1",
                    "tool_version": "quant_tool_minimal_v1",
                    "task_mode": "portfolio_pool_optimize",
                    "feature_group": "price_core",
                    "selection_mode": "tool_score_threshold_plus_date_gate",
                    "score": 0.71,
                    "confidence": 0.3,
                    "action_hint": "仅作反证或灰色参考",
                    "usable_in_agent_default": False,
                    "top_features": ["prior_return_20d", "trend_score"],
                    "counter_evidence": ["latest_time_block_failed"],
                    "promotion_status": "observe_latest_block_failed",
                }
            ],
            "data_gaps": "financial_publish_date_missing_or_unavailable",
            "triggered_skills": "PPS-Q-017;UNKNOWN-SKILL",
            "return_20d": 99.9,
            "gt_status": "evaluated",
            "kline_return_20d": -12.5,
            "kline_return_60d": -8.0,
            "kline_atr20_pct": 4.2,
            "kline_volatility_ratio_20_60": 1.1,
            "peer_kline_group_positive_breadth_20d": 0.52,
            "tushare_industry": "专用设备",
            "tushare_area": "江苏",
            "tushare_industry_group_size": 18,
            "tushare_industry_avg_return_20d": 1.5,
            "tushare_industry_relative_return_20d": -2.0,
            "tushare_industry_positive_breadth_20d": 0.35,
            "tushare_industry_above_ma200_rate": 0.42,
            "tushare_industry_news_attention_gap": -0.4,
            "tushare_area_group_size": 30,
            "tushare_area_avg_return_20d": 0.8,
            "tushare_area_relative_return_20d": -1.3,
            "tushare_area_positive_breadth_20d": 0.45,
        }
    )


def test_evidence_pack_excludes_future_fields_but_keeps_prior_feature() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )
    assert "return_20d" not in pack
    assert "gt_status" not in pack
    assert pack["python_features"]["prior_return_20d"] == 3.4
    assert pack["quant_tool_summaries"][0]["tool_id"] == "date_regime_gate_minimal_v1"
    assert "latest_time_block_failed" in pack["quant_tool_signal_summary"]
    assert "future_return_20d" not in json.dumps(pack["quant_tool_summaries"], ensure_ascii=False)
    assert pack["task_mode"] == "portfolio_pool"
    assert "候选池" in pack["task_mode_requirement"]
    for field in [
        "self_news_intensity",
        "peer_news_intensity",
        "news_warning_score",
        "news_opportunity_score",
        "news_peer_diffusion_score",
        "official_confirmation_score",
        "community_attention_score",
        "community_crowding_risk",
        "announcement_materiality_score",
    ]:
        assert field in pack["news_features"]
    assert pack["allowed_research_grades"]
    assert pack["allowed_simulated_actions"]
    assert pack["news_semantic_questionnaire"]["ds_news_net_score"] == -0.16
    assert pack["news_semantic_questionnaire"]["ds_news_mainline_clarity"] == 0.7
    assert pack["news_semantic_questionnaire"]["ds_news_peer_relative_support"] == -1


def test_operation_plan_requirement_locks_small_entry_softgap_and_floor_policy() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )
    pack["operation_plan_context"] = {
        "operation_action": "small_buy_hold",
        "target_position": 0.25,
        "default_position_floor_if_no_hard_counter": 0.1,
    }

    messages = build_decision_messages(pack)
    payload = json.loads(messages[1]["content"])
    requirement = payload["operation_plan_requirement"]

    assert "不能单独归零" in requirement
    assert "新闻语义问卷/分叉上下文" in requirement
    assert "0.20到0.35" in requirement
    assert "优先承接原仓位" in requirement
    assert pack["financial_report_features"]["financial_report_event_count"] == 1
    assert pack["financial_report_features"]["financial_quality_risk_score"] == 0.65
    assert "quality_risk=0.65" in pack["financial_report_signal_summary"]
    assert "financial_publish_date_missing" not in pack["data_missing_flags"]
    assert pack["kline_features"]["kline_return_20d"] == -12.5
    assert "20d_pullback_observe" in pack["kline_signal_summary"]
    assert pack["book_skill_candidates"][0]["strategy_id"] == "PPS-Q-017"
    assert pack["book_skill_candidates"][0]["source_book"] == "专业投机原理"
    assert "page_range" in pack["book_skill_candidates"][0]
    assert "raw_positive_20d_rate" not in pack["book_skill_candidates"][0]
    assert pack["book_skill_candidates"][1]["source_status"] == "missing_grounded_card"
    assert pack["peer_context_features"]["tushare_industry"] == "专用设备"
    assert pack["peer_context_features"]["tushare_industry_relative_return_20d"] == -2.0
    assert "industry_breadth_weak" in pack["peer_context_signal_summary"]
    assert "真实行业peer弱且目标落后" in pack["counter_evidence"]


def test_single_stock_evidence_pack_declares_task_mode_requirement() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        task_mode="single_stock",
    )
    assert pack["task_mode"] == "single_stock"
    assert "单支模式" in pack["task_mode_requirement"]
    assert "模拟研究暴露" in pack["task_mode_requirement"]


def test_evidence_pack_accepts_goal_task_mode_names() -> None:
    portfolio = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        task_mode="portfolio_pool_optimize",
    )
    single = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        task_mode="single_stock_watch",
    )
    assert portfolio["task_mode"] == "portfolio_pool_optimize"
    assert "组合模式" in portfolio["task_mode_requirement"]
    assert single["task_mode"] == "single_stock_watch"
    assert "单支模式" in single["task_mode_requirement"]


def test_decision_messages_are_json_only_prompt() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        conflict_quality_context="walk_forward_prior_only: kline_risk_conflict=acceptable_reversal_friction",
    )
    messages = build_decision_messages(pack)
    assert "JSON" in messages[0]["content"] or "json" in messages[0]["content"].lower()
    assert "买入" in messages[0]["content"]
    assert "不是自动判信息不足" in messages[0]["content"]
    assert "Book Skill是决策前必须审阅" in messages[0]["content"]
    assert "applicable_condition" in messages[0]["content"]
    assert "DeepSeek语义问卷" in messages[0]["content"]
    assert "财报/业绩公告是高可信新闻类事件通道" in messages[0]["content"]
    assert "不得再声称财报披露日缺失" in messages[0]["content"]
    assert "no_event_in_window只是近期无财报事件" in messages[0]["content"]
    assert "K线通道是量价辅助" in messages[0]["content"]
    assert "真实行业/地域peer_context" in messages[0]["content"]
    assert "量化工具层是机器学习/规则训练后形成的辅助工具" in messages[0]["content"]
    assert "信息空窗/置信度折扣" in messages[0]["content"]
    assert "软缺口时优先写partially_adopted" in messages[0]["content"]
    assert "conflict_quality_context若不为none" in messages[0]["content"]
    assert "promote_context若不为none" in messages[0]["content"]
    assert "sampler_context若不为none" in messages[0]["content"]
    assert "quant_tool_summary_only" in messages[0]["content"]
    assert "kline_requirement" in messages[1]["content"]
    assert "peer_context_requirement" in messages[1]["content"]
    assert "quant_tool_requirement" in messages[1]["content"]
    assert "软缺口时，应优先写quant_tool_adoption_decision=partially_adopted" in messages[1]["content"]
    assert "sampler_context_requirement" in messages[1]["content"]
    assert "conflict_quality_requirement" in messages[1]["content"]
    assert "promote_context_requirement" in messages[1]["content"]
    assert "portfolio_pool/portfolio_pool_optimize用于候选池排序" in messages[0]["content"]
    assert "数据源缺失" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["evidence_pack"]["code"] == "000001"
    assert "walk_forward_prior_only" in payload["evidence_pack"]["conflict_quality_context"]
    assert payload["evidence_pack"]["promote_context"] == "none"
    assert payload["evidence_pack"]["sampler_context"] == "none"
    assert "book_skill_requirement" in payload
    assert "task_mode_requirement" in payload


def test_evidence_pack_adds_observe_only_sampler_context() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        python_candidate="dual_mode_portfolio_pool:rev_plus_chip_core:all_dates:kline_reversal_friction_confirmed:every_2_weeks",
    )

    assert "observe_only_sampler=kline_reversal_friction_confirmed" in pack["sampler_context"]
    assert "不能直接升级" in pack["sampler_context"]
    text = json.dumps(pack, ensure_ascii=False)
    assert "future_return_20d" not in text
    assert '"return_20d"' not in text


def test_evidence_pack_treats_no_recent_financial_event_as_neutral() -> None:
    row = _sample_row()
    row["financial_report_event_count"] = 0
    row["financial_report_missing_rate"] = 1.0
    row["financial_report_join_status"] = "no_event_in_window"
    row["data_gaps"] = "financial_publish_date_missing_or_unavailable"

    pack = build_evidence_pack(
        row,
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )

    assert "financial_publish_date_missing" not in pack["data_missing_flags"]
    assert "财报披露日缺失" not in pack["counter_evidence"]
    assert "no_recent_financial_report_event" in pack["financial_report_signal_summary"]


def test_evidence_pack_keeps_financial_missing_flag_when_feature_history_missing() -> None:
    row = _sample_row()
    row["financial_report_event_count"] = 0
    row["financial_report_missing_rate"] = 1.0
    row["financial_report_join_status"] = "code_not_in_feature_table"
    row["data_gaps"] = "financial_publish_date_missing_or_unavailable"

    pack = build_evidence_pack(
        row,
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )

    assert "financial_publish_date_missing" in pack["data_missing_flags"]
    assert "财报披露日缺失" in pack["counter_evidence"]


def test_deepseek_runner_validates_mock_response() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        assert kwargs["model"] == BACKTEST_TRAINING_MODEL
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "继续深挖",
                                "simulated_action": "增加研究暴露",
                                "simulated_weight_change": 1.0,
                                "final_agent_reasoning_summary": "测试通过",
                                "confidence_level": 0.72,
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    result = decide_evidence_packs([pack], chat_fn=fake_chat)
    assert len(result.ok_cards) == 1
    assert not result.invalid_outputs
    assert result.usage_rows[0]["total_tokens"] == 15


def test_deepseek_runner_normalizes_chinese_confidence() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "信息不足",
                                "simulated_action": "信息不足不动作",
                                "confidence_level": "低",
                                "final_agent_reasoning_summary": "测试中文置信度归一化",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    result = decide_evidence_packs([pack], chat_fn=fake_chat)
    assert result.ok_cards[0]["confidence_level"] == 0.25
    assert result.ok_cards[0]["simulated_weight_change"] == 0.0


def test_deepseek_runner_normalizes_action_weight_conflicts() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "信息不足",
                                "simulated_action": "转入现金",
                                "simulated_weight_change": 1.0,
                                "confidence_level": 0.2,
                                "final_agent_reasoning_summary": "测试动作权重冲突归一化",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    result = decide_evidence_packs([pack], chat_fn=fake_chat)
    assert result.ok_cards[0]["simulated_action"] == "转入现金"
    assert result.ok_cards[0]["simulated_weight_change"] == 0.0


def test_capped_news_guardrail_blocks_increase_when_confirmation_gap_exists() -> None:
    row = _sample_row().copy()
    row["ds_news_positive_capped_by_rule"] = True
    row["ds_news_positive_cap_rule_id"] = "news_questionnaire_routine_announcement_positive_cap_v1"
    row["ds_news_positive_cap_reason"] = "mainline=0.30<0.50; relevance=0.20<0.50"
    row["ds_news_original_opportunity_score"] = 1.0
    row["ds_news_original_net_score"] = 0.45
    row["ds_news_opportunity_score"] = 0.2
    row["ds_news_net_score"] = 0.0
    row["data_gaps"] = "financial_publish_date_missing"
    row["triggered_skills"] = ""
    pack = build_evidence_pack(
        row,
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "继续深挖",
                                "simulated_action": "增加研究暴露",
                                "simulated_weight_change": 0.8,
                                "confidence_level": 0.65,
                                "final_agent_reasoning_summary": "测试违规升级",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    result = decide_evidence_packs([pack], chat_fn=fake_chat)
    card = result.ok_cards[0]
    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "保持观察"
    assert card["simulated_weight_change"] == 0.1
    assert "弱主线规则封顶" in card["counter_evidence"]
    assert "guardrail_applied" in card["error_reflection"]
    assert "执行层改为放入观察" in card["final_agent_reasoning_summary"]


def test_capped_news_guardrail_allows_increase_when_confirmations_exist() -> None:
    row = _sample_row().copy()
    row["ds_news_positive_capped_by_rule"] = True
    row["ds_news_positive_cap_rule_id"] = "news_questionnaire_routine_announcement_positive_cap_v1"
    row["ds_news_positive_cap_reason"] = "mainline=0.30<0.50"
    row["ds_news_opportunity_score"] = 0.2
    row["ds_news_net_score"] = 0.0
    row["data_gaps"] = ""
    row["triggered_skills"] = "CORE_TREND_001"
    pack = build_evidence_pack(
        row,
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )
    pack["book_skill_candidates"] = [{"strategy_id": "CORE_TREND_001", "source_status": "grounded", "source_book": "测试书"}]
    pack["quant_tool_summaries"] = [
        {
            "tool_id": "accepted_test_tool",
            "task_mode": "portfolio_pool_optimize",
            "usable_in_agent_default": True,
            "promotion_status": "accepted",
        }
    ]
    pack["quant_tool_signal_summary"] = "accepted_test_tool status=accepted; usable_default=true"
    pack["peer_context_features"]["tushare_industry_positive_breadth_20d"] = 0.8
    pack["peer_context_features"]["tushare_industry_relative_return_20d"] = 1.0

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "继续深挖",
                                "simulated_action": "增加研究暴露",
                                "simulated_weight_change": 0.8,
                                "confidence_level": 0.65,
                                "final_agent_reasoning_summary": "测试有确认",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    result = decide_evidence_packs([pack], chat_fn=fake_chat)
    card = result.ok_cards[0]
    assert card["simulated_action"] == "增加研究暴露"
    assert card["simulated_weight_change"] == 0.8
    assert card["accepted_quant_tool_ids"] == "accepted_test_tool"
    assert card["quant_tool_adoption_decision"] == "adopted"
    assert card["quant_tool_override_reasons"] == "none"


def test_not_adopted_quant_tool_clears_accepted_ids() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        task_mode="portfolio_pool",
        variant="full_agent_with_quant_tools",
    )
    pack["quant_tool_summaries"] = [
        {
            "tool_id": "accepted_test_tool",
            "task_mode": "portfolio_pool_optimize",
            "usable_in_agent_default": True,
            "promotion_status": "accepted",
        }
    ]

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "放入观察",
                                "simulated_action": "保持观察",
                                "simulated_weight_change": 0.1,
                                "confidence_level": 0.55,
                                "quant_tool_adoption_decision": "not_adopted_counter_evidence",
                                "final_agent_reasoning_summary": "测试反证覆盖工具",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    result = decide_evidence_packs([pack], chat_fn=fake_chat)
    card = result.ok_cards[0]
    assert card["accepted_quant_tool_ids"] == "none"
    assert card["quant_tool_adoption_decision"] == "not_adopted_counter_evidence"
    assert card["quant_tool_override_reasons"] != "none"


def test_portfolio_cross_channel_guardrail_blocks_python_shortcut_for_full_agent_variants() -> None:
    row = _sample_row().copy()
    row["prior_return_20d"] = 75.0
    row["rsi14"] = 82.0
    row["atr20_pct"] = 5.0
    row["news_missing_rate"] = 1.0
    row["financial_report_missing_rate"] = 1.0
    row["financial_report_event_count"] = 0
    row["data_gaps"] = "financial_publish_date_missing_or_unavailable;news_missing_rate_100%"
    row["tushare_industry_positive_breadth_20d"] = 0.3
    row["tushare_industry_relative_return_20d"] = -2.0
    row["triggered_skills"] = ""
    pack = build_evidence_pack(
        row,
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        task_mode="portfolio_pool",
        variant="full_agent",
    )

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "继续深挖",
                                "simulated_action": "增加研究暴露",
                                "simulated_weight_change": 0.6,
                                "confidence_level": 0.7,
                                "final_agent_reasoning_summary": "测试Python捷径",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    result = decide_evidence_packs([pack], chat_fn=fake_chat)
    card = result.ok_cards[0]
    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "保持观察"
    assert card["simulated_weight_change"] == 0.1
    assert "portfolio_cross_channel_confirmation_gap_v1" in card["error_reflection"]

    quant_pack = dict(pack)
    quant_pack["variant"] = "full_agent_with_quant_tools"
    quant_result = decide_evidence_packs([quant_pack], chat_fn=fake_chat)
    assert quant_result.ok_cards[0]["research_grade"] == "放入观察"
    assert "portfolio_cross_channel_confirmation_gap_v1" in quant_result.ok_cards[0]["error_reflection"]

    without_quant_pack = dict(pack)
    without_quant_pack["variant"] = "full_agent_without_quant_tools"
    without_quant_pack["quant_tool_summaries"] = []
    without_quant_pack["quant_tool_signal_summary"] = "component ablation: hidden for full_agent_without_quant_tools"
    without_quant_result = decide_evidence_packs([without_quant_pack], chat_fn=fake_chat)
    assert without_quant_result.ok_cards[0]["research_grade"] == "放入观察"
    assert "portfolio_cross_channel_confirmation_gap_v1" in without_quant_result.ok_cards[0]["error_reflection"]

    ablation_pack = dict(pack)
    ablation_pack["variant"] = "no_news"
    ablation_result = decide_evidence_packs([ablation_pack], chat_fn=fake_chat)
    assert ablation_result.ok_cards[0]["research_grade"] == "继续深挖"


def test_portfolio_guardrail_allows_strong_rev_chip_when_only_soft_confirmation_gaps() -> None:
    row = _sample_row().copy()
    row["prior_return_20d"] = 3.0
    row["rsi14"] = 45.0
    row["atr20_pct"] = 1.5
    row["news_missing_rate"] = 1.0
    row["financial_report_missing_rate"] = 1.0
    row["financial_quality_risk_score"] = 0.2
    row["financial_report_event_count"] = 0
    row["data_gaps"] = "financial_publish_date_missing_or_unavailable;news_missing_rate_100%"
    row["tushare_industry_positive_breadth_20d"] = 0.7
    row["tushare_industry_relative_return_20d"] = 1.0
    row["triggered_skills"] = ""
    row["quant_tool_summaries"] = [_rev_chip_tool_summary(score_quantile=0.92)]
    pack = build_evidence_pack(
        row,
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        task_mode="portfolio_pool",
        variant="full_agent_with_quant_tools",
    )

    result = decide_evidence_packs([pack], chat_fn=_fake_increase_chat)
    card = result.ok_cards[0]
    assert card["research_grade"] == "继续深挖"
    assert card["simulated_action"] == "增加研究暴露"
    assert card["accepted_quant_tool_ids"] == "portfolio_rev_chip_core_ranker"
    assert card["quant_tool_adoption_decision"] == "adopted"
    assert "portfolio_cross_channel_confirmation_gap_v1" not in card["error_reflection"]


def test_portfolio_guardrail_still_blocks_strong_rev_chip_when_peer_or_volatility_hard_gap_exists() -> None:
    row = _sample_row().copy()
    row["prior_return_20d"] = 3.0
    row["rsi14"] = 45.0
    row["atr20_pct"] = 1.5
    row["news_missing_rate"] = 1.0
    row["financial_report_missing_rate"] = 1.0
    row["financial_quality_risk_score"] = 0.2
    row["financial_report_event_count"] = 0
    row["data_gaps"] = "financial_publish_date_missing_or_unavailable;news_missing_rate_100%"
    row["tushare_industry_positive_breadth_20d"] = 0.25
    row["tushare_industry_relative_return_20d"] = -2.0
    row["triggered_skills"] = ""
    row["quant_tool_summaries"] = [_rev_chip_tool_summary(score_quantile=0.92)]
    pack = build_evidence_pack(
        row,
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        task_mode="portfolio_pool",
        variant="full_agent_with_quant_tools",
    )

    result = decide_evidence_packs([pack], chat_fn=_fake_increase_chat)
    card = result.ok_cards[0]
    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "保持观察"
    assert card["accepted_quant_tool_ids"] == "portfolio_rev_chip_core_ranker"
    assert card["quant_tool_adoption_decision"] == "partially_adopted"
    assert "peer_gap" in card["quant_tool_override_reasons"]
    assert "portfolio_cross_channel_confirmation_gap_v1" in card["error_reflection"]


def _rev_chip_tool_summary(*, score_quantile: float) -> dict[str, object]:
    return {
        "tool_id": "portfolio_rev_chip_core_ranker",
        "tool_version": "v1",
        "task_mode": "portfolio_pool",
        "policy_profile": "dual_mode_rev_plus_chip_core",
        "feature_group": "reversal_plus_tushare_chip_core",
        "selection_mode": "cross_section_equal_weight_z_composite",
        "score": 1.25,
        "score_quantile": score_quantile,
        "confidence": 0.6,
        "usable_in_agent_default": True,
        "top_features": ["reversal_composite", "lower_support"],
        "counter_evidence": ["H2026_cost_net_not_green"],
        "promotion_status": "default_combo_ranker_yellow",
        "research_only": True,
        "not_investment_instruction": True,
    }


def _fake_increase_chat(*args, **kwargs):  # noqa: ANN002, ANN003
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "research_grade": "继续深挖",
                            "simulated_action": "增加研究暴露",
                            "simulated_weight_change": 0.6,
                            "confidence_level": 0.7,
                            "final_agent_reasoning_summary": "ranker高分位且硬反证少，有限研究暴露。",
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }


def test_deepseek_runner_records_invalid_output() -> None:
    pack = build_evidence_pack(
        _sample_row(),
        agent_policy_version="test_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
    )

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        return {"choices": [{"message": {"content": "{\"research_grade\":\"买入\"}"}}]}

    result = decide_evidence_packs([pack], chat_fn=fake_chat, retries=0)
    assert not result.ok_cards
    assert len(result.invalid_outputs) == 1


def test_deepseek_runner_parallel_preserves_order_and_user_id() -> None:
    packs = [
        build_evidence_pack(
            pd.Series({**_sample_row().to_dict(), "code": str(code)}),
            agent_policy_version="test_v0",
            step=1,
            train_blocks=["H2023_1"],
            valid_block="H2023_2",
        )
        for code in [3, 1, 2]
    ]
    seen_user_ids = []

    def fake_chat(*args, **kwargs):  # noqa: ANN002, ANN003
        seen_user_ids.append(kwargs.get("user_id"))
        code = json.loads(args[0][1]["content"])["evidence_pack"]["code"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "research_grade": "放入观察",
                                "simulated_action": "保持观察",
                                "simulated_weight_change": 0.2,
                                "final_agent_reasoning_summary": f"并发测试{code}",
                                "confidence_level": 0.5,
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 1},
        }

    result = decide_evidence_packs(packs, chat_fn=fake_chat, max_workers=3, user_id="unit-test-user")
    assert [card["code"] for card in result.ok_cards] == ["000003", "000001", "000002"]
    assert set(seen_user_ids) == {"unit-test-user"}
    assert [row["index"] for row in result.usage_rows] == [0, 1, 2]


def test_preflight_writes_reports(tmp_path: Path) -> None:
    root = tmp_path
    for rel in [
        "docs/DATE_GENERALIZATION_GOAL.md",
        "config/agent_workflow_strategy.yaml",
        "config/deepseek_agent.yaml",
        "src/agent_training/deepseek_client.py",
        "src/agent_training/decision_card.py",
        "book_skills/strategy_cards.yaml",
        "book_skills/grounded_skill_cards.yaml",
        "reports/backtest_scale_500/epoch1/ground_truth.csv",
        "reports/backtest_scale_500/test/ground_truth.csv",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder sk-your-real-key", encoding="utf-8")
    (root / "config/agent_workflow_strategy.yaml").write_text(
        """
research_only: true
no_broker: true
no_auto_trade: true
no_investment_instruction: true
allow_actionable_research_suggestions: true
no_auto_execution_or_guaranteed_return: true
allowed_user_grades:
  - 继续深挖
  - 放入观察
  - 暂时剔除
  - 信息不足
task_modes:
  single_stock_watch: {}
  portfolio_pool_optimize: {}
hard_guards:
  book_skill:
    default_evidence_pack_files:
      - book_skills/grounded_skill_cards.yaml
    reference_only_files:
      - book_skills/strategy_cards.yaml
    allowed_active_files:
      - book_skills/strategy_cards.yaml
      - book_skills/grounded_skill_cards.yaml
""".strip(),
        encoding="utf-8",
    )
    (root / "book_skills/strategy_cards.yaml").write_text(
        """
- strategy_id: TEST-SKILL-001
  principle: 测试原则
  task_fit:
    - single_stock_analysis
  computable_rules:
    - 测试规则
  invalid_conditions:
    - 测试失效条件
  formal_status: 是
  source:
    book: 测试书
    chapter: 第一章
    page_range: OCR_PAGE 1
    raw_source: 测试来源
    extraction_method: full_ocr_txt_deep_dive
    confidence: high
""".strip(),
        encoding="utf-8",
    )
    (root / "book_skills/grounded_skill_cards.yaml").write_text(
        """
- strategy_id: TEST-SKILL-001
  source_book: 测试书
  chapter: 第一章
  page_range: OCR_PAGE 1
  extraction_method: full_ocr_txt_deep_dive
  confidence: high
  source_status: grounded
  validation_status: observe
  trigger_count: 10
  sample_count: 10
  raw_positive_20d_rate: 0.5
  raw_avg_return_20d: 1.0
  applicable_condition: 测试适用条件
  failure_condition: 测试失效条件
  user_output_boundary: 只能作为操作建议的辅助证据，不能单独生成买入/卖出/加减仓结论。
""".strip(),
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        "\n".join([".env", ".env.*", "secrets/", "*api_key*", "*secret*", "ds_api.txt", "tushare_token.txt", "*.key", "*.pem"]),
        encoding="utf-8",
    )
    report = run_preflight(root)
    assert report["ok"]
    md_path, json_path = write_preflight_reports(report, root / "reports/date_generalization")
    assert md_path.exists()
    assert json_path.exists()

def test_deepseek_runner_auto_workers_uses_model_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    packs = [
        build_evidence_pack(
            pd.Series({**_sample_row().to_dict(), "code": str(code)}),
            agent_policy_version="test_v0",
            step=1,
            train_blocks=["H2023_1"],
            valid_block="H2023_2",
        )
        for code in range(5)
    ]
    seen_workers = []

    def fake_parallel(evidence_packs, **kwargs):  # noqa: ANN001, ANN003
        seen_workers.append(kwargs["max_workers"])
        return DeepSeekRunResult(ok_cards=[], invalid_outputs=[], usage_rows=[])

    monkeypatch.setattr("src.agent_training.deepseek_runner._decide_evidence_packs_parallel", fake_parallel)
    decide_evidence_packs(packs, max_workers=0, model="deepseek-v4-flash")
    assert seen_workers == [5]


def test_deepseek_model_concurrency_limits() -> None:
    assert model_concurrency_limit("deepseek-v4-flash") == 2500
    assert model_concurrency_limit("deepseek-v4-pro") == 500

