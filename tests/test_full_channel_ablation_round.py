from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import scripts.run_full_channel_ablation_round as full_channel_module
from scripts.run_full_channel_ablation_round import (
    _attach_case_memory_retrieval,
    _attach_channel_classifier_scores,
    _attach_portfolio_quant_adoption_guard,
    _attach_quant_tool_context,
    _attach_single_stock_risk_review_queue,
    _attach_single_stock_opportunity_preview,
    _attach_single_stock_action_label_preview,
    _attach_news_branch_case_context,
    _attach_analogue_case_context,
    _attach_nonprice_risk_overlay_context,
    _card_key,
    _channel_classifier_quant_tool_rows,
    _hard_counter_calibration_policy,
    _load_portfolio_quant_adoption_guard,
    _load_safe_sample_plan,
    _load_channel_classifier_scores,
    _load_single_stock_opportunity_preview,
    _load_single_stock_action_label_preview,
    _load_single_stock_risk_review_queue,
    _load_news_branch_case_preview,
    _load_analogue_case_preview,
    _load_nonprice_risk_overlay_flags,
    _load_nonprice_risk_overlay_preview,
    _load_questionnaire_scores,
    _pack_key,
    _parse_task_modes,
    _portfolio_quant_adoption_guard_quant_tool_row,
    _sample_plan_operation_context,
    _single_stock_action_label_quant_tool_row,
    _single_stock_opportunity_quant_tool_row,
    _single_stock_risk_review_quant_tool_row,
    _write_summary,
    apply_full_channel_variant,
    assign_sample_panels,
    expand_full_channel_ablation_packs,
    planned_variant_metrics,
    quant_tool_adoption_summary,
    write_questionnaire_sample_plan,
)
from src.agent_training.risk_branch_policy import build_single_stock_risk_branch_policy


def _pack() -> dict:
    return {
        "agent_policy_version": "test_policy",
        "variant": "deepseek_agent",
        "step": 1,
        "train_blocks": "H2023_1",
        "valid_block": "H2023_2",
        "decision_date": "2023-07-04",
        "code": "000001",
        "task_mode": "single_stock",
        "python_signal_summary": "candidate=dual; relative_strength_rank=0.8",
        "python_features": {"relative_strength_rank": 0.8, "prior_return_20d": 5.0},
        "quant_tool_signal_summary": "date_regime_gate_minimal_v1 status=observe_latest_block_failed",
        "quant_tool_summaries": [
            {
                "tool_id": "date_regime_gate_minimal_v1",
                "task_mode": "single_stock_watch",
                "feature_group": "price_core",
                "usable_in_agent_default": False,
                "promotion_status": "observe_latest_block_failed",
                "counter_evidence": ["latest_time_block_failed"],
            }
        ],
        "quant_tool_requirement": "must treat as grey reference",
        "kline_signal_summary": "return20=-12",
        "kline_features": {"kline_return_20d": -12.0, "peer_kline_group_positive_breadth_20d": 0.4},
        "peer_context_signal_summary": "industry=测试",
        "peer_context_features": {"tushare_industry": "测试行业"},
        "chip_signal_summary": "lower_support=0.2",
        "chip_features": {"lower_support": 0.2, "upper_overhang": 0.1},
        "news_signal_summary": "warning=0.1",
        "news_features": {"news_warning_score": 0.1},
        "news_semantic_questionnaire": {"ds_news_risk_score": 0.2},
        "news_branch_case_context": {
            "primary_news_branch": "reversible_reversal_friction",
            "agent_use": "checklist_and_counterevidence_only_not_alpha",
        },
        "news_branch_case_requirement": "use as checklist only",
        "analogue_case_context": [
            {
                "tool_id": "analogue_case_context:single_stock:rev_chip_analogue_guard:every_2_weeks:top5",
                "task_mode": "single_stock",
                "promotion_status": "observe_relative_improvement_context_candidate",
                "agent_use": "base_rate_regime_decay_failure_case_checklist_only_not_alpha",
                "forbidden_use": "do_not_use_as_standalone_alpha_or_research_grade_raise",
                "research_only": True,
                "not_investment_instruction": True,
            }
        ],
        "analogue_case_requirement": "use as checklist only",
        "nonprice_risk_overlay_context": [
            {
                "tool_id": "nonprice_risk_overlay:pullback_high_rev_chip:peer_area_weak:H2023_2",
                "tool_version": "nonprice_risk_overlay_v1",
                "policy_status": "prior_false_veto_guard_candidate",
                "feature_group": "pullback_high_rev_chip",
                "selection_mode": "peer_area_weak",
                "action_hint": "do_not_mechanically_veto",
                "agent_use": "nonprice_conflict_policy_checklist_only_not_alpha",
                "forbidden_use": "do_not_use_as_standalone_alpha_or_trade_instruction",
                "research_only": True,
                "not_investment_instruction": True,
            }
        ],
        "nonprice_risk_overlay_requirement": "use as conflict checklist only",
        "financial_report_signal_summary": "events=1",
        "financial_report_features": {"financial_report_event_count": 1},
        "book_skill_candidates": [
            {"strategy_id": "PPS-Q-017", "source_book": "专业投机原理"},
            {"strategy_id": "DOW-B-004", "source_book": "道氏理论"},
        ],
        "book_skill_requirement": "must review",
        "memory_context": "accepted: test",
        "retrieved_cases_context": "case: test",
        "conflict_quality_context": "walk_forward_prior_only: kline_risk=acceptable_reversal_friction",
        "promote_context": "walk_forward_prior_only: kline_reversal_friction_confirmed=promote_candidate",
        "case_memory_mode": "retrieved_cases_v1",
        "counter_evidence": "新闻覆盖不足",
        "data_missing_flags": "financial_publish_date_missing",
    }


def test_full_channel_variants_hide_only_target_channels() -> None:
    base = _pack()

    no_questionnaire = apply_full_channel_variant(base, "no_questionnaire")
    assert no_questionnaire["news_features"]
    assert no_questionnaire["news_semantic_questionnaire"] == {}
    assert no_questionnaire["news_branch_case_context"] == {}

    no_branch_case_context = apply_full_channel_variant(base, "no_branch_case_context")
    assert no_branch_case_context["news_features"]
    assert no_branch_case_context["news_semantic_questionnaire"]
    assert no_branch_case_context["news_branch_case_context"] == {}
    assert no_branch_case_context["analogue_case_context"]
    assert no_branch_case_context["nonprice_risk_overlay_context"]
    assert no_branch_case_context["memory_context"] != "none"

    no_analogue_case_context = apply_full_channel_variant(base, "no_analogue_case_context")
    assert no_analogue_case_context["news_branch_case_context"]
    assert no_analogue_case_context["analogue_case_context"] == []
    assert no_analogue_case_context["nonprice_risk_overlay_context"]
    assert no_analogue_case_context["memory_context"] != "none"

    no_chip_context = apply_full_channel_variant(base, "no_chip_context")
    assert no_chip_context["chip_features"] == {}
    assert no_chip_context["kline_features"]
    assert no_chip_context["news_features"]
    assert no_chip_context["nonprice_risk_overlay_context"] == []

    no_financial_report = apply_full_channel_variant(base, "no_financial_report")
    assert no_financial_report["financial_report_features"] == {}
    assert no_financial_report["news_features"]
    assert no_financial_report["chip_features"]
    assert no_financial_report["nonprice_risk_overlay_context"] == []

    no_nonprice_risk_overlay = apply_full_channel_variant(base, "no_nonprice_risk_overlay")
    assert no_nonprice_risk_overlay["news_branch_case_context"]
    assert no_nonprice_risk_overlay["analogue_case_context"]
    assert no_nonprice_risk_overlay["nonprice_risk_overlay_context"] == []

    questionnaire_only = apply_full_channel_variant(base, "questionnaire_only")
    assert questionnaire_only["news_features"] == {}
    assert questionnaire_only["news_semantic_questionnaire"]

    no_news = apply_full_channel_variant(base, "no_news")
    assert no_news["news_features"] == {}
    assert no_news["news_semantic_questionnaire"] == {}
    assert no_news["news_branch_case_context"] == {}
    assert no_news["nonprice_risk_overlay_context"] == []
    assert no_news["book_skill_candidates"]

    news_hard = apply_full_channel_variant(base, "news_hard_risk_only")
    assert news_hard["news_features"]["news_warning_score"] == 0.0
    assert news_hard["news_semantic_questionnaire"]["ds_news_risk_score"] == 0.0
    assert news_hard["news_branch_case_context"] == {}
    assert news_hard["nonprice_risk_overlay_context"] == []
    assert "新闻覆盖不足" not in news_hard["counter_evidence"]
    assert news_hard["book_skill_candidates"]

    no_peer = apply_full_channel_variant(base, "no_peer")
    assert no_peer["peer_context_features"] == {}
    assert "peer_kline_group_positive_breadth_20d" not in no_peer["kline_features"]
    assert no_peer["kline_features"]["kline_return_20d"] == -12.0
    assert no_peer["nonprice_risk_overlay_context"] == []

    no_bookskill = apply_full_channel_variant(base, "no_bookskill")
    assert no_bookskill["book_skill_candidates"] == []
    assert no_bookskill["nonprice_risk_overlay_context"] == []
    assert no_bookskill["memory_context"] != "none"

    no_pps_q017 = apply_full_channel_variant(base, "no_pps_q017")
    assert [item["strategy_id"] for item in no_pps_q017["book_skill_candidates"]] == ["DOW-B-004"]
    assert no_pps_q017["specific_bookskill_ablation"]["hidden_strategy_id"] == "PPS-Q-017"
    assert no_pps_q017["specific_bookskill_ablation"]["hidden_cards"] == 1
    assert no_pps_q017["nonprice_risk_overlay_context"] == []
    assert no_pps_q017["memory_context"] != "none"

    no_memory = apply_full_channel_variant(base, "no_memory")
    assert no_memory["memory_context"] == "none"
    assert no_memory["retrieved_cases_context"] == "none"
    assert no_memory["news_branch_case_context"] == {}
    assert no_memory["analogue_case_context"] == []
    assert no_memory["nonprice_risk_overlay_context"] == []
    assert no_memory["conflict_quality_context"] == "none"
    assert no_memory["promote_context"] == "none"
    assert no_memory["case_memory_mode"] == "no_memory"

    no_python = apply_full_channel_variant(base, "no_python_gate")
    assert no_python["python_features"] == {}
    assert no_python["news_features"]
    assert no_python["quant_tool_summaries"]

    no_quant_tools = apply_full_channel_variant(base, "no_quant_tools")
    assert no_quant_tools["quant_tool_summaries"] == []
    assert no_quant_tools["chip_features"]
    assert no_quant_tools["news_features"]

    python_only = apply_full_channel_variant(base, "python_only")
    assert python_only["nonprice_risk_overlay_context"] == []


def test_aggressive_small_entry_variant_raises_operation_floor() -> None:
    base = _pack()
    base["operation_plan_context"] = {
        "operation_action": "small_buy_hold",
        "target_position": 0.2,
        "default_position_floor_if_no_hard_counter": 0.1,
        "default_position_ceiling": 0.35,
    }

    aggressive = apply_full_channel_variant(base, "aggressive_small_entry_035")

    assert aggressive["operation_plan_context"]["target_position"] == 0.35
    assert aggressive["operation_plan_context"]["default_position_floor_if_no_hard_counter"] == 0.35
    assert aggressive["operation_plan_context"]["default_position_ceiling"] == 0.5
    assert aggressive["news_features"]
    assert aggressive["book_skill_candidates"]


def test_news_branch_case_preview_attaches_safely_and_ablation_hides(tmp_path: Path) -> None:
    preview_path = tmp_path / "news_branch_cases.jsonl"
    preview = {
        "tool_id": "news_questionnaire_branch_case_auditor",
        "tool_version": "v1",
        "date": "2023-07-04",
        "code": "000001",
        "primary_news_branch": "reversible_reversal_friction",
        "news_branch_tags": "reversible_reversal_friction;soft_gap",
        "branch_policy": "do_not_hard_veto_from_news_risk_alone",
        "branch_rationale": "risk may be reversible friction",
        "prior_case_count_bucket": "medium",
        "prior_branch_policy_status": "mixed_prior_cases",
        "prior_branch_policy_hint": "use checklist",
        "similar_prior_cases": [{"prior_date": "2023-01-01", "prior_code": "000002", "similarity_band": "medium"}],
        "agent_use": "checklist_and_counterevidence_only_not_alpha",
        "forbidden_use": "do_not_use_as_alpha",
        "source_ref_ids": ["reports/date_generalization/news_questionnaire_branch_case_audit_v1.md"],
    }
    preview_path.write_text(json.dumps(preview, ensure_ascii=False) + "\n", encoding="utf-8")
    packs = [_pack()]

    loaded = _load_news_branch_case_preview(preview_path)
    _attach_news_branch_case_context(packs, loaded)

    assert packs[0]["news_branch_case_context"]["primary_news_branch"] == "reversible_reversal_friction"
    assert "future" not in str(packs[0]["news_branch_case_context"]).lower()
    hidden = apply_full_channel_variant(packs[0], "python_only")
    assert hidden["news_branch_case_context"] == {}


def test_news_branch_case_preview_rejects_future_fields(tmp_path: Path) -> None:
    base = _pack()
    preview_path = tmp_path / "bad_news_branch_cases.jsonl"
    preview_path.write_text(json.dumps({"date": "2023-07-04", "code": "000001", "return_20d": 1.0}) + "\n", encoding="utf-8")

    try:
        _load_news_branch_case_preview(preview_path)
    except ValueError as exc:
        assert "return_20d" in str(exc)
    else:
        raise AssertionError("future field was not rejected")

    with_quant = apply_full_channel_variant(base, "full_agent_with_quant_tools")
    assert with_quant["quant_tool_summaries"]
    assert with_quant["news_features"]

    without_quant = apply_full_channel_variant(base, "full_agent_without_quant_tools")
    assert without_quant["quant_tool_summaries"] == []


def test_nonprice_risk_overlay_preview_attaches_row_active_context_and_ablation_hides(tmp_path: Path) -> None:
    preview_path = tmp_path / "nonprice_overlay.jsonl"
    preview = {
        "tool_id": "nonprice_risk_overlay:pullback_high_rev_chip:peer_area_weak:H2023_2",
        "tool_version": "nonprice_risk_overlay_v1",
        "policy_profile": "nonprice_risk_overlay_prior_only",
        "policy_status": "prior_false_veto_guard_candidate",
        "feature_group": "pullback_high_rev_chip",
        "selection_mode": "peer_area_weak",
        "risk_tier": "do_not_mechanically_veto",
        "risk_branch_labels": ["peer_area_weak", "risk_or_friction", "valid_block=H2023_2", "scope=pullback_high_rev_chip"],
        "branch_policy": "prior_false_veto_guard_candidate; agent_use=do_not_mechanically_veto",
        "action_hint": "do_not_mechanically_veto",
        "top_features": ["area relative return < 0"],
        "source_ref_ids": ["reports/date_generalization/nonprice_risk_overlay_v1.md"],
        "research_only": True,
        "not_investment_instruction": True,
    }
    preview_path.write_text(json.dumps(preview, ensure_ascii=False) + "\n", encoding="utf-8")
    flags_path = tmp_path / "nonprice_flags.csv"
    pd.DataFrame(
        [
            {
                "date": "2023-07-04",
                "code": "000001",
                "name": "平安银行",
                "time_block": "H2023_2",
                "scope_high_rev_chip": True,
                "scope_pullback_high_rev_chip": True,
                "peer_area_weak": True,
                "peer_industry_weak": False,
                "research_only": True,
                "not_investment_instruction": True,
            }
        ]
    ).to_csv(flags_path, index=False)
    pack = _pack()
    pack["valid_block"] = "H2023_2"

    previews = _load_nonprice_risk_overlay_preview(preview_path)
    flags = _load_nonprice_risk_overlay_flags(flags_path)
    _attach_nonprice_risk_overlay_context([pack], previews, flags)

    assert pack["nonprice_risk_overlay_context"]
    context = pack["nonprice_risk_overlay_context"][0]
    assert context["selection_mode"] == "peer_area_weak"
    assert context["action_hint"] == "do_not_mechanically_veto"
    assert context["flag_active_on_current_row"] is True
    assert "future" not in str(context).lower()
    hidden = apply_full_channel_variant(pack, "no_nonprice_risk_overlay")
    assert hidden["nonprice_risk_overlay_context"] == []


def test_nonprice_risk_overlay_task_mode_filter_defaults_to_portfolio_only(tmp_path: Path) -> None:
    preview_path = tmp_path / "nonprice_overlay.jsonl"
    preview = {
        "tool_id": "nonprice_risk_overlay:pullback_high_rev_chip:peer_area_weak:H2023_2",
        "tool_version": "nonprice_risk_overlay_v1",
        "policy_status": "prior_false_veto_guard_candidate",
        "feature_group": "pullback_high_rev_chip",
        "selection_mode": "peer_area_weak",
        "risk_branch_labels": ["valid_block=H2023_2"],
        "action_hint": "do_not_mechanically_veto",
        "research_only": True,
        "not_investment_instruction": True,
    }
    preview_path.write_text(json.dumps(preview, ensure_ascii=False) + "\n", encoding="utf-8")
    flags_path = tmp_path / "nonprice_flags.csv"
    pd.DataFrame(
        [
            {
                "date": "2023-07-04",
                "code": "000001",
                "time_block": "H2023_2",
                "scope_pullback_high_rev_chip": True,
                "peer_area_weak": True,
                "research_only": True,
                "not_investment_instruction": True,
            }
        ]
    ).to_csv(flags_path, index=False)
    previews = _load_nonprice_risk_overlay_preview(preview_path)
    flags = _load_nonprice_risk_overlay_flags(flags_path)
    single_pack = {**_pack(), "valid_block": "H2023_2", "task_mode": "single_stock"}
    portfolio_pack = {**_pack(), "valid_block": "H2023_2", "task_mode": "portfolio_pool"}

    _attach_nonprice_risk_overlay_context(
        [single_pack, portfolio_pack],
        previews,
        flags,
        task_modes=_parse_task_modes("portfolio_pool"),
    )

    assert single_pack["nonprice_risk_overlay_context"] == []
    assert "hidden for task_mode=single_stock" in single_pack["nonprice_risk_overlay_requirement"]
    assert portfolio_pack["nonprice_risk_overlay_context"]
    assert _parse_task_modes("all") is None


def test_nonprice_risk_overlay_preview_rejects_future_fields(tmp_path: Path) -> None:
    base = _pack()
    preview_path = tmp_path / "bad_nonprice_overlay.jsonl"
    preview_path.write_text(json.dumps({"tool_id": "x", "return_20d": 1.0}) + "\n", encoding="utf-8")

    try:
        _load_nonprice_risk_overlay_preview(preview_path)
    except ValueError as exc:
        assert "return_20d" in str(exc)
    else:
        raise AssertionError("future field was not rejected")
    without_quant = apply_full_channel_variant(base, "full_agent_without_quant_tools")
    assert "hidden" in without_quant["quant_tool_signal_summary"]
    assert without_quant["news_features"]

    with_classifier = apply_full_channel_variant(base, "full_agent_with_hard_counter_tool")
    assert with_classifier["quant_tool_summaries"]
    assert "hard-counter" in with_classifier["component_ablation_policy"]

    with_risk_queue = apply_full_channel_variant(base, "full_agent_with_risk_review_queue")
    assert with_risk_queue["quant_tool_summaries"]
    assert "risk review queue" in with_risk_queue["component_ablation_policy"]

    without_classifier = apply_full_channel_variant(
        {
            **base,
            "quant_tool_summaries": [
                *base["quant_tool_summaries"],
                {"tool_id": "channel_rule_outcome_classifier_v1_hard_counter", "task_mode": "single_stock", "promotion_status": "accepted_guard_candidate"},
            ],
        },
        "full_agent_without_channel_classifier",
    )
    assert without_classifier["quant_tool_summaries"]
    assert all("channel_rule_outcome_classifier_v1" not in item["tool_id"] for item in without_classifier["quant_tool_summaries"])

    without_risk_queue = apply_full_channel_variant(
        {
            **base,
            "quant_tool_summaries": [
                *base["quant_tool_summaries"],
                {"tool_id": "single_stock_risk_calibration_v2_review_queue", "task_mode": "single_stock_watch", "promotion_status": "observe_review_only"},
            ],
        },
        "full_agent_without_risk_review_queue",
    )
    assert without_risk_queue["quant_tool_summaries"]
    assert all("single_stock_risk_calibration_v2" not in item["tool_id"] for item in without_risk_queue["quant_tool_summaries"])

    with_opportunity = apply_full_channel_variant(base, "full_agent_with_opportunity_tool")
    assert with_opportunity["quant_tool_summaries"]
    assert "opportunity scorer" in with_opportunity["component_ablation_policy"]

    without_opportunity = apply_full_channel_variant(
        {
            **base,
            "quant_tool_summaries": [
                *base["quant_tool_summaries"],
                {"tool_id": "single_stock_opportunity_scorer_v2", "task_mode": "single_stock_watch", "promotion_status": "green_candidate_requires_cross_channel_audit"},
            ],
        },
        "full_agent_without_opportunity_tool",
    )
    assert without_opportunity["quant_tool_summaries"]
    assert all("single_stock_opportunity_scorer_v2" not in item["tool_id"] for item in without_opportunity["quant_tool_summaries"])


def test_analogue_case_preview_attaches_by_task_mode_and_ablation_hides(tmp_path: Path) -> None:
    preview_path = tmp_path / "analogue_cases.jsonl"
    rows = [
        {
            "tool_id": "analogue_case_context:single_stock:rev_chip_analogue_guard:every_2_weeks:top5",
            "tool_version": "v2",
            "task_mode": "single_stock",
            "promotion_status": "observe_relative_improvement_context_candidate",
            "policy_status": "relative_improvement_context_only",
            "confidence": 0.5,
            "counter_evidence": ["relative_improvement_not_absolute_profit_proof"],
            "research_only": True,
            "not_investment_instruction": True,
        },
        {
            "tool_id": "analogue_case_context:portfolio_pool:analogue_support_score:every_2_weeks:top10",
            "tool_version": "v2",
            "task_mode": "portfolio_pool",
            "promotion_status": "observe_relative_improvement_context_candidate",
            "research_only": True,
            "not_investment_instruction": True,
        },
    ]
    preview_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    pack = _pack()

    loaded = _load_analogue_case_preview(preview_path)
    _attach_analogue_case_context([pack], loaded, max_items=4)

    assert len(pack["analogue_case_context"]) == 1
    assert pack["analogue_case_context"][0]["task_mode"] == "single_stock"
    assert "return_20d" not in json.dumps(pack["analogue_case_context"], ensure_ascii=False)
    hidden = apply_full_channel_variant(pack, "no_analogue_case_context")
    assert hidden["analogue_case_context"] == []


def test_analogue_case_preview_prefers_row_level_match(tmp_path: Path) -> None:
    preview_path = tmp_path / "row_analogue_cases.jsonl"
    rows = [
        {
            "tool_id": "p0_transfer_analog_rag_v1",
            "date": "2023-07-04",
            "code": "000001",
            "time_block": "H2023_2",
            "frequency": "every_2_weeks",
            "base_branch": "branch_stack_v1.small_buy_hold",
            "variant": "green_rule_a",
            "analog_id": "analog_k15_min10",
            "gate_id": "chip_support_plus_analog065",
            "transfer_score": 0.72,
            "transfer_threshold": 0.6,
            "analog_neighbor_count": 15,
            "analog_pos_rate": 0.8,
            "analog_avg_return": 6.5,
            "analog_historical_tail_risk_rate": 0.05,
            "analog_top_case_refs": "2023-01-01:000002:0.123",
            "channel_support_count": 3,
            "channel_hard_counter_count": 0,
            "news_low_warning": True,
            "financial_no_recent_event": True,
            "chip_support_visible": True,
            "agent_instruction": "use as checklist only",
            "auto_trade": False,
        },
        {
            "tool_id": "analogue_case_context:global",
            "task_mode": "single_stock",
            "confidence": 0.2,
        },
    ]
    preview_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    pack = _pack()

    loaded = _load_analogue_case_preview(preview_path)
    _attach_analogue_case_context([pack], loaded, max_items=4)

    assert len(pack["analogue_case_context"]) == 1
    context = pack["analogue_case_context"][0]
    assert context["tool_id"] == "p0_transfer_analog_rag_v1"
    assert context["code"] == "000001"
    assert context["analog_pos_rate"] == 0.8
    assert "row-level matched" in pack["analogue_case_requirement"]
    assert "return_20d" not in json.dumps(pack["analogue_case_context"], ensure_ascii=False)


def test_analogue_case_preview_rejects_future_fields(tmp_path: Path) -> None:
    preview_path = tmp_path / "bad_analogue_cases.jsonl"
    preview_path.write_text(json.dumps({"tool_id": "analogue", "return_20d": 1.0}) + "\n", encoding="utf-8")

    try:
        _load_analogue_case_preview(preview_path)
    except ValueError as exc:
        assert "return_20d" in str(exc)
    else:
        raise AssertionError("future field was not rejected")


def test_python_only_keeps_python_and_hides_agent_channels() -> None:
    python_only = apply_full_channel_variant(_pack(), "python_only")

    assert python_only["python_features"] == {"relative_strength_rank": 0.8, "prior_return_20d": 5.0}
    assert python_only["quant_tool_summaries"] == []
    assert python_only["kline_features"] == {}
    assert python_only["peer_context_features"] == {}
    assert python_only["news_features"] == {}
    assert python_only["news_semantic_questionnaire"] == {}
    assert python_only["analogue_case_context"] == []
    assert python_only["financial_report_features"] == {}
    assert python_only["book_skill_candidates"] == []
    assert python_only["memory_context"] == "none"
    assert python_only["retrieved_cases_context"] == "none"
    assert python_only["conflict_quality_context"] == "none"
    assert python_only["promote_context"] == "none"
    assert "data_missing_flags" in python_only


def test_quant_tool_summary_only_hides_non_quant_channels() -> None:
    quant_only = apply_full_channel_variant(_pack(), "quant_tool_summary_only")

    assert quant_only["quant_tool_summaries"]
    assert quant_only["python_features"] == {}
    assert quant_only["kline_features"] == {}
    assert quant_only["peer_context_features"] == {}
    assert quant_only["news_features"] == {}
    assert quant_only["news_semantic_questionnaire"] == {}
    assert quant_only["analogue_case_context"] == []
    assert quant_only["financial_report_features"] == {}
    assert quant_only["book_skill_candidates"] == []
    assert quant_only["memory_context"] == "none"
    assert quant_only["conflict_quality_context"] == "none"
    assert quant_only["promote_context"] == "none"
    assert quant_only["case_memory_mode"] == "quant_tool_summary_only"


def test_attach_quant_tool_context_filters_after_global_load() -> None:
    pack = _pack()
    pack["task_mode"] = "portfolio_pool"
    summaries = [
        {"tool_id": "single", "task_mode": "single_stock_watch", "usable_in_agent_default": False, "promotion_status": "observe_latest_block_failed"},
        {"tool_id": "portfolio", "task_mode": "portfolio_pool_optimize", "usable_in_agent_default": False, "promotion_status": "reject_too_few_samples"},
    ]

    _attach_quant_tool_context(pack, summaries, max_items=3)

    assert [item["tool_id"] for item in pack["quant_tool_summaries"]] == ["portfolio"]
    assert "portfolio" in pack["quant_tool_signal_summary"]


def test_channel_classifier_scores_attach_row_level_safe_tools(tmp_path) -> None:
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "date": "2023-07-04",
                "code": "1",
                "logistic_channel_outcome__prob_hard_counter": 0.82,
                "logistic_channel_outcome__prob_soft_gap": 0.12,
                "logistic_channel_outcome__prob_positive_support": 0.06,
                "return_20d": -9.0,
            }
        ]
    ).to_csv(csv_path, index=False)
    scores = _load_channel_classifier_scores(csv_path)
    pack = _pack()

    _attach_channel_classifier_scores([pack], scores)

    ids = [item["tool_id"] for item in pack["quant_tool_summaries"]]
    assert "channel_rule_outcome_classifier_v1_hard_counter" in ids
    hard = next(item for item in pack["quant_tool_summaries"] if item["tool_id"] == "channel_rule_outcome_classifier_v1_hard_counter")
    assert hard["score"] == 0.82
    assert hard["risk_tier"] == "hard_counter_yellow_review_0.80_0.95"
    assert "risk_tier=hard_counter_yellow_review_0.80_0.95" in pack["quant_tool_signal_summary"]
    assert "return_20d" not in str(hard)


def test_portfolio_quant_adoption_guard_load_attach_and_hide(tmp_path) -> None:
    csv_path = tmp_path / "portfolio_guard.csv"
    pd.DataFrame(
        [
            {
                "date": "2023-07-04",
                "code": "1",
                "guard_probability": 0.42,
                "guard_threshold": 0.50,
                "guard_allow_raise": False,
                "quant_score_pct_by_date": 0.12,
                "quant_raise_candidate": True,
                "logistic_kline_peer_chip": 0.33,
                "logistic_kline_peer_chip_regime": 0.28,
                "ml_keypoint_score": 0.66,
                "return_20d": -8.0,
                "pool_excess_20d": -3.0,
            }
        ]
    ).to_csv(csv_path, index=False)
    guard = _load_portfolio_quant_adoption_guard(csv_path)
    pack = _pack()
    pack["task_mode"] = "portfolio_pool"
    pack["sampler_context"] = "decision_keypoint_sampler_ml_v1;stratum=ordinary_control_midkey"

    _attach_portfolio_quant_adoption_guard([pack], guard)

    tool = next(item for item in pack["quant_tool_summaries"] if item["tool_id"] == "portfolio_quant_adoption_guard_v1_row_context")
    assert tool["score"] == 0.42
    assert tool["score_quantile"] == 0.12
    assert tool["risk_tier"] == "row_quant_low_percentile_do_not_raise"
    assert tool["primary_risk_branch"] == "low_row_quant_percentile_ordinary_control_midkey"
    assert "global_tool_summary_not_row_signal" in tool["counter_evidence"]
    assert "ordinary_control_midkey_quant_no_raise_by_default" in tool["counter_evidence"]
    assert "v4_no_quant_raise_without_nonquant_confirmation" in tool["counter_evidence"]
    assert "ordinary_control_midkey" in tool["action_hint"]
    assert "not_adopted_counter_evidence" in pack["quant_tool_requirement"]
    assert "return_20d" not in str(tool)
    assert "pool_excess_20d" not in str(tool)
    assert "行级采用保护上下文" in pack["quant_tool_requirement"]

    hidden = apply_full_channel_variant(pack, "full_agent_without_quant_tools")
    assert hidden["quant_tool_summaries"] == []


def test_portfolio_quant_adoption_guard_row_is_sanitized() -> None:
    tool = _portfolio_quant_adoption_guard_quant_tool_row(
        {
            "guard_probability": 0.62,
            "guard_threshold": 0.55,
            "guard_allow_raise": True,
            "quant_score_pct_by_date": 0.81,
            "quant_raise_candidate": True,
            "logistic_kline_peer_chip": 0.73,
            "return_20d": 11.0,
        }
    )

    assert tool["tool_id"] == "portfolio_quant_adoption_guard_v1_row_context"
    assert tool["task_mode"] == "portfolio_pool"
    assert tool["tool_version"] == "row_level_quant_percentile_guard_v4_no_raise_without_nonquant_confirmation"
    assert tool["risk_tier"] == "row_quant_context_allows_review_not_auto_raise"
    assert tool["usable_in_agent_default"] is False
    assert "review_only" in tool["action_hint"]
    assert "at_least_two_target_specific_nonquant_confirmations_from_news_financial_peer_bookskill_or_announcement" in tool["required_confirmation"]
    assert "return_20d" not in str(tool)


def test_single_stock_risk_review_queue_load_attach_and_hide(tmp_path) -> None:
    queue_path = tmp_path / "queue.jsonl"
    row = {
        "date": "2023-07-04",
        "code": "1",
        "risk_score": 0.42,
        "review_priority_score": 0.77,
        "cap_pct": 0.10,
        "risk_tier": "hard_counter_yellow_review_0.80_0.95",
        "research_grade": "放入观察",
        "policy_status": "validation_selected_review_only",
        "tool_version": "capped_review_queue_v2",
        "decision_frequency": "scheduled_twice_weekly_or_key_points",
    }
    queue_path.write_text(pd.Series(row).to_json(force_ascii=False) + "\n", encoding="utf-8")
    queue = _load_single_stock_risk_review_queue(queue_path)
    assert ("2023-07-04", "000001") in queue

    pack = _pack()
    pack["task_mode"] = "single_stock"
    _attach_single_stock_risk_review_queue([pack], queue)

    ids = [item["tool_id"] for item in pack["quant_tool_summaries"]]
    assert "single_stock_risk_calibration_v2_review_queue" in ids
    tool = next(item for item in pack["quant_tool_summaries"] if item["tool_id"] == "single_stock_risk_calibration_v2_review_queue")
    assert tool["score"] == 0.42
    assert tool["risk_tier"] == "hard_counter_yellow_review_0.80_0.95"
    assert tool["policy_status"] == "validation_selected_review_only"
    assert tool["cap_pct"] == 0.10
    assert tool["tool_grade"] == "放入观察"
    assert "return_20d" not in str(tool)

    hidden = apply_full_channel_variant(pack, "full_agent_without_risk_review_queue")
    assert all("single_stock_risk_calibration_v2" not in item["tool_id"] for item in hidden["quant_tool_summaries"])


def test_single_stock_risk_review_queue_rejects_future_fields(tmp_path) -> None:
    queue_path = tmp_path / "bad_queue.jsonl"
    queue_path.write_text('{"date":"2023-07-04","code":"000001","return_20d":-9.0}\n', encoding="utf-8")

    try:
        _load_single_stock_risk_review_queue(queue_path)
    except ValueError as exc:
        assert "future/result" in str(exc)
    else:
        raise AssertionError("expected future-field rejection")


def test_single_stock_risk_review_quant_tool_row_is_sanitized() -> None:
    tool = _single_stock_risk_review_quant_tool_row(
        {
            "risk_score": 0.33,
            "review_priority_score": 0.66,
            "cap_pct": 0.15,
            "risk_tier": "hard_counter_high_risk_review_ge_0.95",
            "research_grade": "暂时剔除",
            "policy_status": "fixed15_next_oot_candidate",
        }
    )

    assert tool["tool_id"] == "single_stock_risk_calibration_v2_review_queue"
    assert tool["task_mode"] == "single_stock_watch"
    assert tool["policy_status"] == "fixed15_next_oot_candidate"
    assert tool["cap_pct"] == 0.15
    assert tool["tool_grade"] == "暂时剔除"
    assert tool["research_only"] is True
    assert "return_20d" not in str(tool)
    assert "never_raise_opportunity" in tool["action_hint"]
    assert tool["primary_risk_branch"]
    assert tool["risk_branch_labels"]


def test_risk_branch_policy_detects_low_hard_reversal_support() -> None:
    pack = _pack()
    pack["news_features"] = {"news_count_30d": 20, "news_missing_rate": 0.0, "news_warning_score": 0.2}
    pack["financial_report_features"] = {
        "financial_report_join_status": "no_event_in_window",
        "financial_report_missing_rate": 0.0,
        "financial_report_event_count": 0,
    }
    pack["kline_features"] = {"kline_return_20d": 45.0, "kline_return_60d": 52.0, "kline_rsi14": 68.0}
    pack["peer_context_features"] = {
        "tushare_industry_relative_return_20d": 35.0,
        "tushare_industry_positive_breadth_20d": 0.7,
    }
    pack["chip_features"] = {"lower_support": 0.22, "upper_overhang": 0.08, "winner_rate_pct": 90.0}
    row = {"risk_tier": "low_hard_counter_probability"}

    policy = build_single_stock_risk_branch_policy(pack, row)

    assert policy["primary_risk_branch"] == "low_hard_counter_with_reversal_support"
    assert "low_hard_counter_probability" in policy["risk_branch_labels"]
    assert "peer_relative_support" in policy["risk_branch_labels"]
    assert "do_not_downweight_from_risk_queue_alone" in policy["branch_action_hint"]


def test_risk_branch_policy_does_not_treat_winner_rate_alone_as_chip_support() -> None:
    pack = _pack()
    pack["news_features"] = {"news_count_30d": 20, "news_missing_rate": 0.0, "news_warning_score": 0.0}
    pack["financial_report_features"] = {
        "financial_report_join_status": "code_not_in_feature_table",
        "financial_report_missing_rate": 1.0,
        "financial_report_event_count": 0,
    }
    pack["kline_features"] = {"kline_return_20d": 12.0, "kline_return_60d": 29.0, "kline_rsi14": 38.0}
    pack["peer_context_features"] = {
        "tushare_industry_relative_return_20d": 11.0,
        "tushare_industry_positive_breadth_20d": 0.63,
    }
    pack["chip_features"] = {"lower_support": 0.13, "upper_overhang": 0.06, "winner_rate_pct": 92.0}
    row = {"risk_tier": "low_hard_counter_probability"}

    policy = build_single_stock_risk_branch_policy(pack, row)

    assert policy["primary_risk_branch"] != "low_hard_counter_with_reversal_support"
    assert "chip_support_or_low_overhang" not in policy["risk_branch_labels"]


def test_risk_branch_policy_detects_explicit_hard_negative_event() -> None:
    pack = _pack()
    pack["news_features"] = {
        "news_count_30d": 4,
        "news_missing_rate": 0.0,
        "news_warning_score": 0.9,
        "news_evidence_quality": 0.8,
    }
    pack["news_semantic_questionnaire"] = {"ds_news_risk_score": 0.8}
    pack["financial_report_features"] = {
        "financial_report_event_count": 1,
        "financial_quality_risk_score": 0.75,
        "financial_report_missing_rate": 0.0,
    }
    row = {"risk_tier": "hard_counter_high_risk_review_ge_0.95"}

    policy = build_single_stock_risk_branch_policy(pack, row)

    assert policy["primary_risk_branch"] == "explicit_hard_negative_event"
    assert "explicit_negative_news_event" in policy["risk_branch_labels"]
    assert "explicit_financial_risk_event" in policy["risk_branch_labels"]


def test_single_stock_opportunity_preview_load_attach_and_hide(tmp_path) -> None:
    preview_path = tmp_path / "opportunity.jsonl"
    row = {
        "date": "2023-07-04",
        "code": "1",
        "tool_id": "single_stock_opportunity_scorer_v2",
        "tool_version": "safe_orthogonal_channels_v2",
        "task_mode": "single_stock_watch",
        "model_variant": "additive_bin_baseline_existing",
        "feature_group": "baseline_existing",
        "opportunity_score": 0.25,
        "opportunity_quantile_in_date": 0.82,
        "opportunity_threshold": 0.2,
        "tool_status": "green_candidate",
        "research_grade": "继续深挖",
        "required_confirmation": "normal_cross_channel_review",
        "top_feature_names": "kline_return_20d;corr_peer_avg_return_20d",
        "source_ref_ids": "single_stock_opportunity_scorer_v2,local_time_safe_feature_cache",
    }
    preview_path.write_text(pd.Series(row).to_json(force_ascii=False) + "\n", encoding="utf-8")
    preview = _load_single_stock_opportunity_preview(preview_path)
    assert ("2023-07-04", "000001") in preview

    pack = _pack()
    pack["task_mode"] = "single_stock"
    _attach_single_stock_opportunity_preview([pack], preview)

    tool = next(item for item in pack["quant_tool_summaries"] if item["tool_id"] == "single_stock_opportunity_scorer_v2")
    assert tool["score"] == 0.25
    assert tool["score_quantile"] == 0.82
    assert tool["tool_grade"] == "继续深挖"
    assert tool["usable_in_agent_default"] is True
    assert "return_20d" not in tool["top_features"]

    hidden = apply_full_channel_variant(pack, "full_agent_without_opportunity_tool")
    assert all("single_stock_opportunity_scorer_v2" not in item["tool_id"] for item in hidden["quant_tool_summaries"])


def test_single_stock_opportunity_preview_rejects_future_fields(tmp_path) -> None:
    preview_path = tmp_path / "bad_opportunity.jsonl"
    preview_path.write_text('{"date":"2023-07-04","code":"000001","return_20d":9.0}\n', encoding="utf-8")

    try:
        _load_single_stock_opportunity_preview(preview_path)
    except ValueError as exc:
        assert "future/result" in str(exc)
    else:
        raise AssertionError("expected future-field rejection")


def test_single_stock_opportunity_quant_tool_row_is_sanitized() -> None:
    tool = _single_stock_opportunity_quant_tool_row(
        {
            "opportunity_score": 0.11,
            "opportunity_quantile_in_date": 0.77,
            "opportunity_threshold": 0.08,
            "tool_status": "green_candidate",
            "research_grade": "放入观察",
            "model_variant": "additive_bin_baseline_existing",
            "feature_group": "baseline_existing",
            "top_feature_names": "kline_return_20d;return_20d;neg_log_mv",
            "source_ref_ids": "single_stock_opportunity_scorer_v2",
        }
    )

    assert tool["tool_id"] == "single_stock_opportunity_scorer_v2"
    assert tool["task_mode"] == "single_stock_watch"
    assert tool["usable_in_agent_default"] is True
    assert "return_20d" not in tool["top_features"]
    assert "opportunity_candidate_summary_requires_agent_audit" == tool["action_hint"]


def test_single_stock_action_label_preview_load_attach_and_hide(tmp_path) -> None:
    preview_path = tmp_path / "action_label.jsonl"
    row = {
        "date": "2026-03-17",
        "code": "1330",
        "tool_id": "p0_action_label_scorer_v1",
        "frequency": "every_2_weeks",
        "feature_group": "wide_safe",
        "model": "hgb",
        "policy_name": "precision_entry_v1",
        "entry_prob": 0.69,
        "strong_entry_prob": 0.61,
        "reduce_prob": 0.12,
        "action_edge_score": 0.86,
        "entry_threshold": 0.28,
        "reduce_threshold": 0.58,
        "target_position": 0.6,
        "operation_hint": "trial_buy_or_add_review",
        "source_ref_ids": "joined_ground_truth_combined_news_asof_cache;p0_action_label_scorer_v1",
    }
    preview_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    preview = _load_single_stock_action_label_preview(preview_path)
    assert ("2026-03-17", "001330") in preview

    pack = _pack()
    pack["task_mode"] = "single_stock"
    pack["decision_date"] = "2026-03-17"
    pack["code"] = "001330"
    _attach_single_stock_action_label_preview([pack], preview)

    tool = next(item for item in pack["quant_tool_summaries"] if item["tool_id"] == "p0_action_label_scorer_v1")
    assert tool["score"] == 0.86
    assert tool["cap_pct"] == 0.6
    assert tool["usable_in_agent_default"] is True
    assert "entry_label" not in json.dumps(tool)

    hidden = apply_full_channel_variant(pack, "no_action_label_tool")
    assert all("p0_action_label_scorer_v1" not in item["tool_id"] for item in hidden["quant_tool_summaries"])
    assert hidden.get("operation_plan_context", {}) == {}


def test_single_stock_action_label_preview_rejects_future_fields(tmp_path) -> None:
    preview_path = tmp_path / "bad_action_label.jsonl"
    preview_path.write_text('{"date":"2026-03-17","code":"001330","entry_label":1}\n', encoding="utf-8")

    try:
        _load_single_stock_action_label_preview(preview_path)
    except ValueError as exc:
        assert "future/result" in str(exc)
    else:
        raise AssertionError("expected future-field rejection")


def test_single_stock_action_label_preview_prefers_precision_duplicate(tmp_path) -> None:
    preview_path = tmp_path / "dupe_action_label.jsonl"
    balanced = {
        "date": "2026-03-17",
        "code": "001330",
        "tool_id": "p0_action_label_scorer_v1",
        "frequency": "every_2_weeks",
        "feature_group": "wide_safe",
        "model": "hgb",
        "policy_name": "balanced_action_v1",
        "entry_prob": 0.9,
        "strong_entry_prob": 0.8,
        "reduce_prob": 0.1,
        "action_edge_score": 0.9,
        "target_position": 0.55,
    }
    precision = {**balanced, "policy_name": "precision_entry_v1", "action_edge_score": 0.7, "target_position": 0.6}
    preview_path.write_text(
        json.dumps(balanced, ensure_ascii=False) + "\n" + json.dumps(precision, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    preview = _load_single_stock_action_label_preview(preview_path)

    assert preview[("2026-03-17", "001330")]["policy_name"] == "precision_entry_v1"


def test_single_stock_action_label_quant_tool_row_is_sanitized() -> None:
    tool = _single_stock_action_label_quant_tool_row(
        {
            "frequency": "every_2_weeks",
            "feature_group": "wide_safe",
            "model": "hgb",
            "policy_name": "precision_entry_v1",
            "entry_prob": 0.7,
            "strong_entry_prob": 0.6,
            "reduce_prob": 0.1,
            "action_edge_score": 0.8,
            "entry_threshold": 0.3,
            "reduce_threshold": 0.6,
            "target_position": 0.6,
            "operation_hint": "trial_buy_or_add_review",
            "source_ref_ids": "joined_ground_truth_combined_news_asof_cache;p0_action_label_scorer_v1",
        }
    )

    assert tool["tool_id"] == "p0_action_label_scorer_v1"
    assert tool["task_mode"] == "single_stock_watch"
    assert tool["policy_status"] == "yellow_entry_candidate_requires_confirmation"
    assert "review_trial_buy_or_add" in tool["action_hint"]
    assert "return_20d" not in json.dumps(tool)


def test_action_label_buy_add_operation_context_soft_gaps_do_not_zero() -> None:
    context = _sample_plan_operation_context(
        pd.Series(
            {
                "operation_action": "buy_add",
                "operation_action_cn": "试探买入/加仓",
                "local_target_position": 0.6,
                "local_reason_code": "p0_action_label_scorer_v1",
                "frequency": "every_2_weeks",
                "valid_block": "H2026_1",
            }
        )
    )

    assert context["local_validation_status"] == "yellow_action_label_entry_candidate_for_ds_confirmation"
    assert context["default_position_floor_if_no_hard_counter"] == 0.10
    assert context["default_position_ceiling"] == 0.6
    assert "不能在没有明确硬反证时直接归零" in context["soft_gap_policy"]


def test_attach_case_memory_retrieval_injects_safe_applicable_context() -> None:
    pack = _pack()
    pack["task_mode"] = "single_stock"
    pack["python_signal_summary"] = "relative strength high; prior_return overheat"
    pack["news_signal_summary"] = "新闻缺失 high uncertainty"
    pack["data_missing_flags"] = "financial_publish_date_missing"

    _attach_case_memory_retrieval([pack], mode="retrieved_cases_v2_applicable", top_k=2)

    assert pack["case_memory_mode"] == "retrieved_cases_v2_applicable"
    assert pack["retrieved_cases_context"].startswith("retrieved_cases_applicability:")
    assert "return_20d" not in pack["retrieved_cases_context"]
    assert "future_return" not in pack["retrieved_cases_context"]


def test_channel_classifier_quant_tool_rows_do_not_expose_future_fields() -> None:
    rows = _channel_classifier_quant_tool_rows(
        {
            "logistic_channel_outcome__prob_hard_counter": 0.7,
            "logistic_channel_outcome__prob_soft_gap": 0.2,
            "logistic_channel_outcome__prob_positive_support": 0.1,
            "return_20d": -3.0,
        },
        task_mode="portfolio_pool",
    )

    assert len(rows) == 3
    assert rows[0]["selection_mode"] == "hard_counter"
    assert rows[0]["risk_tier"] == "low_hard_counter_probability"
    assert rows[0]["required_confirmation"]
    assert rows[0]["known_false_veto_risk"] == "high_false_veto_risk_if_used_as_veto"
    assert "return_20d" not in str(rows)


def test_hard_counter_calibration_policy_tiers() -> None:
    high = _hard_counter_calibration_policy(hard=0.96, soft=0.01, positive=0.02)
    yellow = _hard_counter_calibration_policy(hard=0.85, soft=0.03, positive=0.02)
    soft = _hard_counter_calibration_policy(hard=0.20, soft=0.40, positive=0.10)

    assert high["risk_tier"] == "hard_counter_high_risk_review_ge_0.95"
    assert high["action_hint"] == "strong_downweight_only_if_cross_channel_conflicts_confirmed"
    assert yellow["risk_tier"] == "hard_counter_yellow_review_0.80_0.95"
    assert yellow["known_false_veto_risk"] == "high_false_veto_risk_soft_gap_and_reversal_samples"
    assert soft["risk_tier"] == "soft_gap_dominant_low_hard"


def test_load_safe_sample_plan_drops_future_fields(tmp_path) -> None:
    path = tmp_path / "sample_plan.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "code": "301",
                "task_mode": "portfolio_pool",
                "return_20d": -3.0,
                "gt_status": "evaluated",
                "rule_outcome_label": "hard_counter",
                "pool_excess_20d": -1.2,
            }
        ]
    ).to_csv(path, index=False)

    plan = _load_safe_sample_plan(path)

    assert list(plan["code"]) == ["000301"]
    assert "return_20d" not in plan.columns
    assert "gt_status" not in plan.columns
    assert "rule_outcome_label" not in plan.columns
    assert plan.iloc[0]["valid_block"] == "H2026_1"


def test_resume_key_defaults_to_panel_one_for_legacy_cards() -> None:
    pack = _pack()
    pack["sample_panel_id"] = "panel_01"
    card = {key: pack.get(key) for key in ["agent_policy_version", "variant", "step", "valid_block", "decision_date", "code", "task_mode"]}

    assert _pack_key(pack) == _card_key(card)


def test_expand_and_planned_metrics_include_variant_dimension() -> None:
    packs = expand_full_channel_ablation_packs([_pack()], ["full_agent", "full_agent_without_quant_tools", "no_news"])
    assert [pack["variant"] for pack in packs] == ["full_agent", "full_agent_without_quant_tools", "no_news"]

    planned = planned_variant_metrics(packs)
    assert set(planned["variant"]) == {"full_agent", "full_agent_without_quant_tools", "no_news"}
    assert set(planned["planned_evidence_packs"]) == {1}


def test_assign_sample_panels_is_non_overlapping_by_mode() -> None:
    packs = []
    for index in range(5):
        row = _pack()
        row["code"] = f"{index + 1:06d}"
        row["task_mode"] = "portfolio_pool"
        packs.append(row)

    selected = assign_sample_panels(packs, panel_size=2, panel_count=2)

    assert [pack["code"] for pack in selected] == ["000001", "000002", "000003", "000004"]
    assert [pack["sample_panel_id"] for pack in selected] == ["panel_01", "panel_01", "panel_02", "panel_02"]
    assert [pack["sample_rank_in_panel"] for pack in selected] == [1, 2, 1, 2]


def test_write_questionnaire_sample_plan_omits_duplicates_and_future_fields(tmp_path) -> None:
    first = _pack()
    duplicate = _pack()
    duplicate["task_mode"] = "portfolio_pool"
    first["sample_panel_id"] = "panel_01"
    plan_path = tmp_path / "plan.csv"

    write_questionnaire_sample_plan([first, duplicate], plan_path)

    text = plan_path.read_text(encoding="utf-8-sig")
    assert text.count("000001") == 1
    assert "return_20d" not in text
    assert "gt_status" not in text
    assert "sample_panel_id" in text


def test_load_questionnaire_scores_prefers_latest_same_rank_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(full_channel_module, "OUTPUT", tmp_path)
    older = tmp_path / "news_questionnaire_flash_old_scores.csv"
    newer = tmp_path / "news_questionnaire_flash_new_scores.csv"
    row = {
        "decision_date": "2026-01-06",
        "code": "1",
        "questionnaire_version": "news_semantic_questionnaire_v1",
        "mainline_summary": "old",
        "ds_news_net_score": -0.5,
    }
    pd.DataFrame([row]).to_csv(older, index=False)
    newer_row = {**row, "mainline_summary": "new", "ds_news_net_score": 0.25}
    pd.DataFrame([newer_row]).to_csv(newer, index=False)
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    scores = _load_questionnaire_scores()

    assert len(scores) == 1
    assert scores.iloc[0]["code"] == "000001"
    assert scores.iloc[0]["source_score_file"] == newer.name
    assert scores.iloc[0]["ds_news_net_score"] == 0.25


def test_quant_tool_adoption_summary_groups_structured_fields() -> None:
    cards = [
        {
            "variant": "full_agent_with_quant_tools",
            "task_mode": "portfolio_pool",
            "accepted_quant_tool_ids": "tool_a",
            "quant_tool_adoption_decision": "not_adopted_counter_evidence",
            "quant_tool_override_reasons": "news_gap;financial_gap",
        },
        {
            "variant": "full_agent_with_quant_tools",
            "task_mode": "portfolio_pool",
            "accepted_quant_tool_ids": "none",
            "quant_tool_adoption_decision": "not_applicable",
            "quant_tool_override_reasons": "none",
        },
    ]

    summary = quant_tool_adoption_summary(cards)

    assert int(summary["decision_cards"].sum()) == 2
    accepted = summary[summary["quant_tool_adoption_decision"].eq("not_adopted_counter_evidence")].iloc[0]
    assert accepted["accepted_tool_cards"] == 1
    assert "financial_gap" in accepted["quant_tool_override_reasons"]


def test_write_summary_accepts_dryrun_without_cards(tmp_path) -> None:
    args = argparse.Namespace(
        agent_policy_version="test_policy",
        model="deepseek-v4-flash",
        panel_count=2,
        variants="full_agent_with_quant_tools",
        valid_blocks="H2026_1",
        quant_tool_rule_outcomes="reports/test.jsonl",
        quant_tool_max_items=4,
    )
    path = tmp_path / "summary.md"

    _write_summary(
        path,
        args=args,
        base_count=1,
        pack_count=1,
        called=False,
        reused=False,
        metrics=pd.DataFrame(),
        step_metrics=pd.DataFrame([{"variant": "full_agent_with_quant_tools", "planned_evidence_packs": 1}]),
        usage=pd.DataFrame(columns=["total_tokens"]),
        invalid_count=0,
        questionnaire_plan_path=tmp_path / "sample_plan.csv",
        cards=[],
    )

    text = path.read_text(encoding="utf-8")
    assert "called_deepseek: `False`" in text
    assert "## Quant Tool Adoption" in text
