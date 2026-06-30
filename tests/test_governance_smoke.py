from __future__ import annotations

from scripts.run_governance_smoke import (
    build_asof_manifest,
    build_decision_critic_review,
    build_rule_outcomes,
    build_source_ref_manifest,
    _find_forbidden_tokens,
)


def _pack() -> dict:
    return {
        "agent_policy_version": "test_policy",
        "valid_block": "H2025_1",
        "decision_date": "2025-03-14",
        "available_at": "2025-03-14 15:00",
        "code": "000001",
        "task_mode": "portfolio_pool",
        "variant": "full_agent",
        "sample_panel_id": "panel_01",
        "python_features": {"relative_strength_rank": 0.9, "counter_score": 8, "prior_return_20d": 18.0},
        "python_signal_summary": "relative_strength_rank=0.9; prior_return_20d=18",
        "quant_tool_summaries": [
            {
                "tool_id": "date_regime_gate_minimal_v1",
                "usable_in_agent_default": False,
                "promotion_status": "observe_latest_block_failed",
            }
        ],
        "quant_tool_signal_summary": "date_regime_gate_minimal_v1 status=observe_latest_block_failed",
        "kline_features": {"kline_return_20d": -12.0, "kline_return_60d": -20.0},
        "kline_signal_summary": "return20=-12",
        "peer_context_features": {"tushare_industry_positive_breadth_20d": 0.1},
        "peer_context_signal_summary": "industry breadth weak",
        "news_features": {
            "news_missing_rate": 1.0,
            "news_warning_score": 0.1,
            "source_type": "local_cache",
            "source_name": "news_features",
        },
        "news_signal_summary": "missing_rate=1",
        "news_semantic_questionnaire": {"ds_news_uncertainty_score": 0.8},
        "financial_report_features": {
            "financial_report_event_count": 0,
            "financial_report_missing_rate": 1.0,
            "financial_report_latest_period": "2024Q4",
            "financial_report_available_at": "2025-03-01 00:00",
        },
        "financial_report_signal_summary": "events=0",
        "book_skill_candidates": [
            {
                "strategy_id": "DOW-B-017",
                "source_status": "grounded",
                "source_book": "道氏理论",
                "page_range": "OCR_PAGE 1-2",
                "confidence": "high",
            }
        ],
        "memory_context": "EXP-1; rule_or_observation=test; accepted_or_rejected=observe; next_action=test",
        "retrieved_cases_context": "none",
        "counter_evidence": "news missing",
    }


def test_governance_smoke_builds_source_asof_rules_and_critic() -> None:
    pack = _pack()
    refs, by_channel = build_source_ref_manifest(pack, "pack-1")
    asof = build_asof_manifest(pack, "pack-1", refs)
    outcomes = build_rule_outcomes(pack, "pack-1", by_channel)
    critic = build_decision_critic_review(pack, "pack-1", refs, asof, outcomes)

    assert refs
    assert by_channel["news"]
    assert by_channel["bookskill"]
    assert by_channel["quant_tool"]
    assert asof["asof_pass"]
    assert any(row["rule_id"] == "quant_tool_unusable_default_counter_v1" for row in outcomes)
    assert any(row["rule_id"] == "news_missing_is_uncertainty_v1" for row in outcomes)
    assert any(row["rule_id"] == "portfolio_cross_channel_confirmation_gap_v1" for row in outcomes)
    assert critic["critic_pass"]
    assert "portfolio_confirmation_gap_guard_triggered" in critic["warning_findings"]


def test_governance_critic_blocks_forbidden_memory_metric() -> None:
    pack = _pack()
    pack["memory_context"] = "EXP-1; metric_after=bad hindsight metric"
    refs, by_channel = build_source_ref_manifest(pack, "pack-2")
    asof = build_asof_manifest(pack, "pack-2", refs)
    outcomes = build_rule_outcomes(pack, "pack-2", by_channel)
    critic = build_decision_critic_review(pack, "pack-2", refs, asof, outcomes)

    assert not critic["critic_pass"]
    assert any("forbidden_prompt_tokens_visible" in item for item in critic["blocking_findings"])


def test_governance_critic_blocks_future_asof_ref() -> None:
    pack = _pack()
    pack["financial_report_features"]["financial_report_available_at"] = "2025-03-15 00:00"
    refs, by_channel = build_source_ref_manifest(pack, "pack-3")
    asof = build_asof_manifest(pack, "pack-3", refs)
    outcomes = build_rule_outcomes(pack, "pack-3", by_channel)
    critic = build_decision_critic_review(pack, "pack-3", refs, asof, outcomes)

    assert not asof["asof_pass"]
    assert not critic["critic_pass"]
    assert "asof_manifest_has_refs_after_decision_time" in critic["blocking_findings"]


def test_governance_forbidden_token_scan_allows_historical_feature_names() -> None:
    pack = {
        "kline_features": {
            "kline_return_5d": 1.2,
            "kline_return_20d": -3.4,
            "prior_return_20d": -1.0,
        },
        "quant_tool_signal_summary": "top=kline_return_20d,kline_return_60d,prior_return_20d",
    }

    assert _find_forbidden_tokens(pack) == []


def test_governance_forbidden_token_scan_blocks_future_result_names() -> None:
    pack = {
        "memory_context": "bad hindsight token return_20d should not be visible",
        "metric_after": "bad",
    }

    findings = _find_forbidden_tokens(pack)

    assert "memory_context" in findings
    assert "metric_after" in findings
