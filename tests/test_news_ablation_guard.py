from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.agent_training.evidence_pack import apply_decision_guardrails


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_deepseek_news_ablation_round.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_deepseek_news_ablation_round", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guarded_questionnaire_caps_weak_mainline_positive_score() -> None:
    module = _module()
    pack = {
        "news_features": {"news_opportunity_score": 0.8},
        "news_signal_summary": "count=3; warning=0.1; opportunity=0.8; semantic_net=0.55",
        "news_semantic_questionnaire": {
            "ds_news_opportunity_score": 0.9,
            "ds_news_net_score": 0.55,
            "ds_news_mainline_clarity": 0.3,
            "ds_news_decision_relevance": 0.4,
            "ds_news_repetition_lag": 0.2,
            "ds_news_risk_score": 0.1,
        },
    }

    guarded = module._apply_variant(pack, "keyword_plus_questionnaire_guarded")
    fields = guarded["news_semantic_questionnaire"]

    assert fields["ds_news_positive_capped_by_rule"] is True
    assert fields["ds_news_positive_cap_rule_id"] == module.ROUTINE_POSITIVE_CAP_RULE_ID
    assert fields["ds_news_original_opportunity_score"] == 0.9
    assert fields["ds_news_original_net_score"] == 0.55
    assert fields["ds_news_opportunity_score"] == module.ROUTINE_POSITIVE_CAP_OPPORTUNITY_SCORE
    assert fields["ds_news_net_score"] == module.ROUTINE_POSITIVE_CAP_NET_SCORE
    assert "mainline=0.30<0.50" in fields["ds_news_positive_cap_reason"]
    assert "capped_opportunity" in guarded["news_signal_summary"]
    assert "positive news opportunity/net score is capped" in guarded["news_ablation_policy"]


def test_guarded_questionnaire_keeps_high_quality_positive_score() -> None:
    module = _module()
    pack = {
        "news_features": {"news_opportunity_score": 0.8},
        "news_signal_summary": "count=3; warning=0.1; opportunity=0.8; semantic_net=0.55",
        "news_semantic_questionnaire": {
            "ds_news_opportunity_score": 0.9,
            "ds_news_net_score": 0.55,
            "ds_news_mainline_clarity": 0.8,
            "ds_news_decision_relevance": 0.75,
            "ds_news_repetition_lag": 0.1,
            "ds_news_risk_score": 0.1,
        },
    }

    guarded = module._apply_variant(pack, "keyword_plus_questionnaire_guarded")
    fields = guarded["news_semantic_questionnaire"]

    assert fields["ds_news_positive_capped_by_rule"] is False
    assert fields["ds_news_opportunity_score"] == 0.9
    assert fields["ds_news_net_score"] == 0.55
    assert guarded["news_signal_summary"] == "count=3; warning=0.1; opportunity=0.8; semantic_net=0.55"
    assert "same as keyword_plus_questionnaire" in guarded["news_ablation_policy"]


def test_parse_variants_accepts_experimental_questionnaire_arms() -> None:
    module = _module()

    variants = module._parse_variants(
        "uncertainty_only_questionnaire,quality_only_questionnaire,"
        "semantic_risk_only_questionnaire,risk_uncertainty_questionnaire,"
        "no_financial_report_channel,financial_report_only,news_plus_financial_report,"
        "news_plus_financial_report_guarded,no_kline,kline_weak_prompt"
    )

    assert variants == [
        "uncertainty_only_questionnaire",
        "quality_only_questionnaire",
        "semantic_risk_only_questionnaire",
        "risk_uncertainty_questionnaire",
        "no_financial_report_channel",
        "financial_report_only",
        "news_plus_financial_report",
        "news_plus_financial_report_guarded",
        "no_kline",
        "kline_weak_prompt",
    ]


def test_uncertainty_only_hides_positive_and_risk_fields() -> None:
    module = _module()
    pack = _news_variant_pack()

    variant = module._apply_variant(pack, "uncertainty_only_questionnaire")
    fields = variant["news_semantic_questionnaire"]

    assert variant["news_features"] == {}
    assert fields["ds_news_uncertainty_score"] == 0.65
    assert fields["ds_news_mainline_clarity"] == 0.3
    assert "ds_news_opportunity_score" not in fields
    assert "ds_news_net_score" not in fields
    assert "ds_news_risk_score" not in fields
    assert "uncertainty-only" in variant["news_signal_summary"]


def test_quality_only_hides_directional_fields() -> None:
    module = _module()
    pack = _news_variant_pack()

    variant = module._apply_variant(pack, "quality_only_questionnaire")
    fields = variant["news_semantic_questionnaire"]

    assert variant["news_features"] == {}
    assert fields["ds_news_quality_score"] == 0.4
    assert fields["ds_news_official_support"] == 0.2
    assert "ds_news_opportunity_score" not in fields
    assert "ds_news_risk_score" not in fields
    assert "ds_news_self_regulatory_legal" not in fields
    assert "quality-only" in variant["news_signal_summary"]


def test_semantic_risk_only_hides_keyword_and_positive_fields() -> None:
    module = _module()
    pack = _news_variant_pack()

    variant = module._apply_variant(pack, "semantic_risk_only_questionnaire")
    fields = variant["news_semantic_questionnaire"]

    assert variant["news_features"] == {}
    assert fields["ds_news_risk_score"] == 0.7
    assert fields["ds_news_self_regulatory_legal"] == -2
    assert "ds_news_opportunity_score" not in fields
    assert "ds_news_net_score" not in fields
    assert "semantic-risk-only" in variant["news_signal_summary"]


def test_risk_uncertainty_keeps_risk_keyword_but_hides_opportunity() -> None:
    module = _module()
    pack = _news_variant_pack()

    variant = module._apply_variant(pack, "risk_uncertainty_questionnaire")
    fields = variant["news_semantic_questionnaire"]

    assert "news_warning_score" in variant["news_features"]
    assert "news_opportunity_score" not in variant["news_features"]
    assert fields["ds_news_risk_score"] == 0.7
    assert fields["ds_news_uncertainty_score"] == 0.65
    assert "ds_news_opportunity_score" not in fields
    assert "ds_news_net_score" not in fields
    assert "risk-uncertainty" in variant["news_signal_summary"]


def test_financial_report_ablation_variants_are_isolated() -> None:
    module = _module()
    pack = _news_variant_pack()

    no_financial = module._apply_variant(pack, "no_financial_report_channel")
    financial_only = module._apply_variant(pack, "financial_report_only")
    both = module._apply_variant(pack, "news_plus_financial_report")

    assert no_financial["financial_report_features"] == {}
    assert no_financial["news_features"]["news_warning_score"] == 0.6
    assert financial_only["news_features"] == {}
    assert financial_only["news_semantic_questionnaire"] == {}
    assert financial_only["financial_report_features"]["financial_quality_risk_score"] == 0.7
    assert both["news_features"]["news_warning_score"] == 0.6
    assert both["financial_report_features"]["financial_report_event_count"] == 1


def test_financial_report_guarded_variant_adds_guardrail() -> None:
    module = _module()
    pack = _news_variant_pack()
    pack["python_features"] = {"prior_return_20d": 25.0, "rsi14": 74.0}
    pack["book_skill_candidates"] = [{"strategy_id": "QUALITY_001", "source_status": "must_resolve_before_strong_evidence"}]

    guarded = module._apply_variant(pack, "news_plus_financial_report_guarded")

    assert guarded["financial_report_guardrail"]["rule_id"] == "financial_risk_to_zero_guard_v1"
    assert guarded["financial_report_guardrail"]["triggered_for_pack"] is True
    assert "guardrail=financial_risk_to_zero_guard_v1_triggered" in guarded["financial_report_signal_summary"]
    assert guarded["news_features"]["news_warning_score"] == 0.6
    assert guarded["financial_report_features"]["financial_quality_risk_score"] == 0.7


def test_kline_ablation_variants_are_isolated() -> None:
    module = _module()
    pack = _news_variant_pack()
    pack["kline_features"] = {"kline_return_20d": -12.5}
    pack["kline_signal_summary"] = "return20=-12.5; flags=20d_pullback_observe"

    hidden = module._apply_variant(pack, "no_kline")
    visible = module._apply_variant(pack, "kline_weak_prompt")

    assert hidden["kline_features"] == {}
    assert "no K-line fields visible" in hidden["kline_signal_summary"]
    assert visible["kline_features"]["kline_return_20d"] == -12.5
    assert "weak quantitative context" in visible["kline_ablation_policy"]


def test_financial_report_guardrail_forces_zero_research_weight() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "financial_report_guardrail": {
                "rule_id": "financial_risk_to_zero_guard_v1",
                "triggered_for_pack": True,
            },
            "book_skill_candidates": [
                {"strategy_id": "QUALITY_001", "source_status": "must_resolve_before_strong_evidence"}
            ],
        }
    )
    card = {
        "research_grade": "继续深挖",
        "simulated_action": "增加研究暴露",
        "simulated_weight_change": 0.8,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "信息不足不动作"
    assert card["simulated_weight_change"] == 0.0
    assert "financial_risk_to_zero_guard_v1" in card["error_reflection"]
    assert "财报风险归零护栏" in card["counter_evidence"]


def test_financial_report_only_guardrail_blocks_standalone_upgrade() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack["variant"] = "financial_report_only"
    card = {
        "research_grade": "继续深挖",
        "simulated_action": "增加研究暴露",
        "simulated_weight_change": 0.6,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "保持观察"
    assert card["simulated_weight_change"] == 0.1
    assert "financial_report_only_no_upgrade_without_confirmation_v1" in card["error_reflection"]
    assert "财报单通道" in card["counter_evidence"]


def test_rag_applicable_failure_guardrail_blocks_upgrade() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "case_memory_mode": "retrieved_cases_v2_applicable",
            "variant": "news_plus_financial_report_guarded",
            "retrieved_cases_context": (
                "retrieved_cases_applicability:\n"
                "- case_id=FAIL-001 | applicability=applicable | "
                "guidance=treat as applicable checklist and counter-evidence before upgrading research exposure. "
                "prior verdict=rejected_for_default because of bad active exposure"
            ),
        }
    )
    card = {
        "research_grade": "继续深挖",
        "simulated_action": "增加研究暴露",
        "simulated_weight_change": 0.7,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "保持观察"
    assert card["simulated_weight_change"] == 0.1
    assert "rag_applicable_failure_no_upgrade_v1" in card["error_reflection"]
    assert "历史相似失败条件命中" in card["counter_evidence"]


def test_single_stock_risk_review_queue_blocks_upgrade_and_caps_weight() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "task_mode": "single_stock",
            "quant_tool_summaries": [
                {
                    "tool_id": "single_stock_risk_calibration_v2_review_queue",
                    "usable_in_agent_default": False,
                    "promotion_status": "observe_review_only",
                    "policy_status": "fixed15_next_oot_candidate_not_retroactive_default",
                    "risk_tier": "hard_counter_yellow_review_0.80_0.95",
                    "counter_evidence": ["review_only_not_trade_instruction"],
                }
            ],
        }
    )
    card = {
        "research_grade": "继续深挖",
        "simulated_action": "增加研究暴露",
        "simulated_weight_change": 0.7,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "降低研究暴露"
    assert card["simulated_weight_change"] == 0.05
    assert "single_stock_risk_review_queue_no_raise_v1" in card["error_reflection"]
    assert "review-only工具" in card["counter_evidence"]


def test_single_stock_risk_review_queue_caps_high_observe_weight() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "task_mode": "single_stock",
            "quant_tool_summaries": [
                {
                    "tool_id": "single_stock_risk_calibration_v2_review_queue",
                    "usable_in_agent_default": False,
                    "promotion_status": "observe_review_only",
                }
            ],
        }
    )
    card = {
        "research_grade": "放入观察",
        "simulated_action": "保持观察",
        "simulated_weight_change": 0.1,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "降低研究暴露"
    assert card["simulated_weight_change"] == 0.05
    assert "single_stock_risk_review_queue_no_raise_v1" in card["error_reflection"]


def test_single_stock_risk_review_low_hard_support_does_not_mechanically_downweight() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "task_mode": "single_stock",
            "quant_tool_summaries": [
                {
                    "tool_id": "single_stock_risk_calibration_v2_review_queue",
                    "usable_in_agent_default": False,
                    "promotion_status": "observe_review_only",
                    "primary_risk_branch": "low_hard_counter_with_reversal_support",
                    "risk_branch_labels": ["low_hard_counter_probability", "peer_relative_support"],
                }
            ],
        }
    )
    card = {
        "research_grade": "放入观察",
        "simulated_action": "保持观察",
        "simulated_weight_change": 0.1,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "保持观察"
    assert card["simulated_weight_change"] == 0.1
    assert card["error_reflection"] == ""


def test_single_stock_risk_review_low_hard_support_repairs_branch_false_veto() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "task_mode": "single_stock",
            "quant_tool_summaries": [
                {
                    "tool_id": "single_stock_risk_calibration_v2_review_queue",
                    "usable_in_agent_default": False,
                    "promotion_status": "observe_review_only",
                    "primary_risk_branch": "low_hard_counter_with_reversal_support",
                    "risk_branch_labels": ["low_hard_counter_probability", "chip_support_or_low_overhang"],
                }
            ],
        }
    )
    card = {
        "research_grade": "放入观察",
        "simulated_action": "降低研究暴露",
        "simulated_weight_change": 0.05,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "保持观察"
    assert card["simulated_weight_change"] == 0.1
    assert "single_stock_risk_review_queue_branch_no_downweight_v1" in card["error_reflection"]
    assert "机械降权" in card["counter_evidence"]


def test_single_stock_risk_review_explicit_hard_branch_still_caps_weight() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "task_mode": "single_stock",
            "quant_tool_summaries": [
                {
                    "tool_id": "single_stock_risk_calibration_v2_review_queue",
                    "usable_in_agent_default": False,
                    "promotion_status": "observe_review_only",
                    "primary_risk_branch": "explicit_hard_negative_event",
                    "risk_branch_labels": ["explicit_negative_news_event", "low_hard_counter_probability"],
                }
            ],
        }
    )
    card = {
        "research_grade": "放入观察",
        "simulated_action": "保持观察",
        "simulated_weight_change": 0.1,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "放入观察"
    assert card["simulated_action"] == "降低研究暴露"
    assert card["simulated_weight_change"] == 0.05
    assert "single_stock_risk_review_queue_no_raise_v1" in card["error_reflection"]


def test_single_stock_risk_review_queue_keeps_existing_zero_action() -> None:
    evidence_pack = _news_variant_pack()
    evidence_pack.update(
        {
            "task_mode": "single_stock",
            "quant_tool_summaries": [
                {
                    "tool_id": "single_stock_risk_calibration_v2_review_queue",
                    "usable_in_agent_default": False,
                    "promotion_status": "observe_review_only",
                }
            ],
        }
    )
    card = {
        "research_grade": "信息不足",
        "simulated_action": "信息不足不动作",
        "simulated_weight_change": 0.0,
        "counter_evidence": "无强反证",
        "error_reflection": "",
        "final_agent_reasoning_summary": "",
    }

    apply_decision_guardrails(card, evidence_pack)

    assert card["research_grade"] == "信息不足"
    assert card["simulated_action"] == "信息不足不动作"
    assert card["simulated_weight_change"] == 0.0
    assert card["error_reflection"] == ""


def test_financial_report_matched_selector_uses_status_or_event_count() -> None:
    module = _module()
    frame = pd.DataFrame(
        [
            {"financial_report_join_status": "no_event_in_window", "financial_report_event_count": 0},
            {"financial_report_join_status": "event_window_matched", "financial_report_event_count": 0},
            {"financial_report_join_status": "", "financial_report_event_count": 2},
        ]
    )

    assert module._financial_report_matched_selector(frame).tolist() == [False, True, True]


def test_sample_plan_builds_dual_mode_packs_without_future_fields(tmp_path) -> None:
    module = _module()
    plan = tmp_path / "sample_plan.csv"
    pd.DataFrame(
        [
            {
                "date": "2025-04-29",
                "code": "000001",
                "candidate_rule": "financial_multi_report_review_v1",
                "reason_to_test": "复核多报告事件",
            }
        ]
    ).to_csv(plan, index=False)
    frame = pd.DataFrame([_sample_plan_source_row()])
    args = SimpleNamespace(
        sample_plan=str(plan),
        sample_plan_rules="",
        sample_plan_per_rule=0,
        sample_plan_max_rows=0,
        sample_plan_task_modes="both",
        agent_policy_version="sample_plan_test_v1",
    )

    packs = module._build_sample_plan_packs(frame, args=args)

    assert len(packs) == 2
    assert {pack["task_mode"] for pack in packs} == {"portfolio_pool", "single_stock"}
    assert {pack["valid_block"] for pack in packs} == {"H2025_1"}
    assert not _contains_key(packs, "return_20d")
    assert all("financial_multi_report_review_v1" in pack["python_signal_summary"] for pack in packs)


def test_sample_plan_rejects_future_result_columns(tmp_path) -> None:
    module = _module()
    plan = tmp_path / "sample_plan.csv"
    pd.DataFrame(
        [{"date": "2025-04-29", "code": "000001", "return_20d": 9.9}]
    ).to_csv(plan, index=False)
    args = SimpleNamespace(
        sample_plan=str(plan),
        sample_plan_rules="",
        sample_plan_per_rule=0,
        sample_plan_max_rows=0,
        sample_plan_task_modes="single_stock",
        agent_policy_version="sample_plan_test_v1",
    )

    with pytest.raises(ValueError, match="future/result fields"):
        module._build_sample_plan_packs(pd.DataFrame([_sample_plan_source_row()]), args=args)


def _news_variant_pack() -> dict:
    return {
        "news_features": {
            "news_warning_score": 0.6,
            "news_opportunity_score": 0.8,
            "policy_background_score": 0.5,
            "official_confirmation_score": 0.2,
            "news_missing_rate": 0.1,
        },
        "news_signal_summary": "count=3; warning=0.6; opportunity=0.8; semantic_net=0.1",
        "news_semantic_questionnaire": {
            "news_semantic_questionnaire_version": "news_semantic_questionnaire_v1",
            "ds_news_mainline_summary": "主线弱且风险高",
            "ds_news_mainline_clarity": 0.3,
            "ds_news_source_coverage": 0.5,
            "ds_news_official_support": 0.2,
            "ds_news_timestamp_confidence": 0.6,
            "ds_news_self_regulatory_legal": -2,
            "ds_news_self_capital_financing": -1,
            "ds_news_peer_risk_diffusion": 0.8,
            "ds_news_policy_headwind": 0.4,
            "ds_news_region_risk": 0.3,
            "ds_news_cross_stock_confirmation": 0.1,
            "ds_news_peer_silent_gap": 0.7,
            "ds_news_conflict_intensity": 0.8,
            "ds_news_consensus_crowding": 0.7,
            "ds_news_novelty": 0.2,
            "ds_news_repetition_lag": 0.9,
            "ds_news_decision_relevance": 0.4,
            "ds_news_risk_score": 0.7,
            "ds_news_opportunity_score": 0.9,
            "ds_news_uncertainty_score": 0.65,
            "ds_news_quality_score": 0.4,
            "ds_news_net_score": -0.1,
            "ds_news_missing_or_conflict_notes": "来源弱且重复",
        },
        "financial_report_signal_summary": "events=1; quality_risk=0.7",
        "financial_report_features": {
            "financial_report_event_count": 1,
            "financial_report_materiality_score": 0.8,
            "financial_quality_risk_score": 0.7,
            "financial_report_missing_rate": 0.0,
            "financial_report_join_status": "event_window_matched",
        },
    }


def _sample_plan_source_row() -> dict:
    return {
        "date": "2025-04-29",
        "code": "000001",
        "name": "样本公司",
        "industry": "测试行业",
        "timeline_score": 6.5,
        "relative_strength_rank": 0.5,
        "counter_score": 7.0,
        "rsi14": 55.0,
        "prior_return_20d": 3.0,
        "news_missing_rate": 0.1,
        "triggered_skills": "QUALITY_001",
        "financial_report_event_count": 3,
        "financial_report_materiality_score": 0.8,
        "financial_quality_risk_score": 0.2,
        "financial_surprise_score": 0.1,
        "financial_disclosure_quality_score": 0.8,
        "financial_report_missing_rate": 0.0,
        "financial_report_event_types": "annual_report;audit_report",
        "financial_report_available_at": "2025-04-29 00:00:00",
        "financial_report_join_status": "event_window_matched",
    }


def _contains_key(value, key: str) -> bool:  # noqa: ANN001
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False
