from __future__ import annotations

import json

import pandas as pd

from scripts.audit_p0_friday_stack_case_memory_v1 import (
    build_case_evidence_pack,
    build_bookskill_candidates,
    guard_policy_flags,
)
from src.agent_training.case_memory_retriever import (
    ApplicableRetrievedCase,
    BROAD_OR_META_TAGS,
    RetrievedCase,
    _case_condition_tags,
    _evidence_condition_tags,
    _is_meta_or_broad_case,
)


def _case(*conditions: str) -> ApplicableRetrievedCase:
    return ApplicableRetrievedCase(
        case=RetrievedCase(
            case_id="CASE-X",
            ledger="memory/failure_case_ledger.csv",
            score=1.0,
            matched_terms=(),
            output={},
            rank_text="",
        ),
        applicability="applicable",
        matched_conditions=tuple(conditions),
        missing_conditions=(),
        guidance="test",
    )


def test_guard_policy_flags_support_condition_specific_guards() -> None:
    flags = guard_policy_flags(
        [
            _case("single_stock", "financial_report_context", "weak_peer_confirmation"),
            _case("single_stock", "news_hidden_or_missing"),
        ]
    )

    assert flags["guard_applicable_any"] is True
    assert flags["guard_condition_financial_report_context"] is True
    assert flags["guard_condition_news_hidden_or_missing"] is True
    assert flags["guard_condition_weak_peer_confirmation"] is True
    assert flags["guard_condition_financial_or_news"] is True
    assert flags["guard_condition_financial_and_peer"] is True


def test_evidence_pack_uses_conditional_risk_words_not_neutral_feature_names() -> None:
    row = pd.Series(
        {
            "date": "2026-01-02",
            "code": "000001",
            "time_block": "H2026_1",
            "opp_score": 0.2,
            "opp_threshold": 0.1,
            "opp_quantile_in_date": 0.9,
            "target_position": 0.6,
            "kline_opp_score": 0.4,
            "kline_opp_threshold": 0.3,
            "kline_risk_score": 0.2,
            "kline_risk_threshold": 0.6,
            "kline_return_20d": 2.0,
            "kline_rsi14": 55.0,
            "upper_overhang": 0.2,
            "lower_support": 0.05,
            "news_missing_rate": 0.1,
            "news_warning_score": 0.2,
            "news_opportunity_score": 0.3,
            "financial_report_join_status": "no_event_in_window",
            "financial_report_event_count": 0,
            "financial_report_missing_rate": 0.0,
            "tushare_industry_positive_breadth_20d": 0.7,
            "tushare_industry_relative_return_20d": 1.0,
            "triggered_skills": "",
        }
    )

    pack = build_case_evidence_pack(row)
    text = json.dumps(pack, ensure_ascii=False)
    tags = _evidence_condition_tags(pack)

    assert "return_20d" not in pack
    assert "future_return" not in text
    assert "news_hidden_or_missing" not in tags
    assert "chip_overhang_pressure" not in tags
    assert "chip_support_confirmed" not in tags
    assert "financial_no_recent_event" in tags


def test_small_entry_branch_tag_is_context_not_standalone_guard() -> None:
    pack = {
        "task_mode": "single_stock",
        "policy_name": "branch_stack_v1",
        "operation_action": "small_buy_hold",
        "operation_hint": "小仓试探/持有",
    }
    pack_tags = _evidence_condition_tags(pack)
    case_tags = _case_condition_tags("single_stock small_entry small buy 小仓试探")
    matched = pack_tags & case_tags
    specific = {tag for tag in matched if tag not in BROAD_OR_META_TAGS}

    assert "small_entry_branch" in matched
    assert not specific

    pack_with_financial_gap = {
        **pack,
        "financial_report_features": {"financial_report_join_status": "code_not_in_feature_table"},
    }
    pack_gap_tags = _evidence_condition_tags(pack_with_financial_gap)
    case_gap_tags = _case_condition_tags("small_entry financial_publish_date_missing")

    assert "financial_missing" in (pack_gap_tags & case_gap_tags)


def test_broad_bookskill_gap_failure_is_meta_case() -> None:
    case = RetrievedCase(
        case_id="FAIL-20260629-082",
        ledger="memory/failure_case_ledger.csv",
        score=1.0,
        matched_terms=(),
        output={"countermeasure": "BookSkill coverage gap is not a sell/downweight rule"},
        rank_text="BookSkill missing is too broad for automatic guard",
    )

    assert _is_meta_or_broad_case(case)


def test_bookskill_candidates_use_grounded_resolver_fields() -> None:
    candidates = build_bookskill_candidates(pd.Series({"triggered_skills": "PPS-Q-017"}))

    assert candidates
    first = candidates[0]
    assert first["strategy_id"] == "PPS-Q-017"
    assert "source_book" in first
    assert "page_range" in first
    assert "applicable_condition" in first
    assert "failure_condition" in first
