from __future__ import annotations

from pathlib import Path

from scripts.retrieve_case_memory_smoke import render_report
from src.agent_training.case_memory_retriever import (
    _evidence_condition_tags,
    format_applicable_retrieved_cases,
    format_retrieved_cases,
    retrieve_applicable_cases,
    retrieve_cases,
)


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_retrieve_cases_finds_relevant_failure_without_future_metrics(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "memory" / "failure_case_ledger.csv",
        "\n".join(
            [
                "failure_id,source_round,task_mode,failure_pattern,countermeasure,status",
                "FAIL-X,round,portfolio_pool_optimize,no_news Python relative strength caused bad active exposure 20d=-14.96,require news financial peer Book Skill confirmation,open",
            ]
        ),
    )
    _write_csv(
        tmp_path / "memory" / "strategy_experience_ledger.csv",
        "\n".join(
            [
                "experience_id,source_round,task_mode,rule_or_observation,train_blocks,validation_block,metric_before,metric_after,accepted_or_rejected,failure_condition,next_action",
                "EXP-X,round,all,hidden news should not mean low risk,train,val,before,after,observe,news missing,run small shard",
            ]
        ),
    )

    cases = retrieve_cases(tmp_path, "no_news Python relative strength financial Book Skill confirmation", top_k=3)
    text = format_retrieved_cases(cases)

    assert cases
    assert "FAIL-X" in text or "EXP-X" in text
    assert "return_20d" not in text
    assert "gt_status" not in text
    assert "-14.96" not in text
    assert "confirmation" in text or "Book Skill" in text


def test_retrieval_smoke_report_has_queries_and_boundaries() -> None:
    text = render_report(["financial_report_only Book Skill confirmation"], top_k=2)

    assert "不调用 DeepSeek" in text
    assert "no_rag" in text
    assert "retrieved_cases_v1" in text
    assert "API key" in text
    assert "sk-" not in text


def test_retrieve_applicable_cases_requires_shared_failure_conditions(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "memory" / "failure_case_ledger.csv",
        "\n".join(
            [
                "failure_id,source_round,task_mode,failure_pattern,countermeasure,status",
                "FAIL-Y,round,portfolio_pool,no_news Python relative strength with financial disclosure missing and Book Skill gap caused bad active exposure,require news financial peer Book Skill confirmation,open",
            ]
        ),
    )
    pack = {
        "variant": "no_news",
        "task_mode": "portfolio_pool",
        "python_signal_summary": "relative_strength_rank=0.95; counter_score=8",
        "python_features": {"relative_strength_rank": 0.95, "counter_score": 8},
        "news_features": {"news_missing_rate": 1.0},
        "financial_report_features": {"financial_report_missing_rate": 1.0, "financial_report_event_count": 0},
        "financial_report_signal_summary": "财报披露日缺失",
        "peer_context_features": {"tushare_industry_positive_breadth_20d": 0.1},
        "book_skill_candidates": [],
        "counter_evidence": "新闻缺失; 财报披露日缺失; Book Skill未解析",
        "data_missing_flags": "financial_publish_date_missing_or_unavailable",
    }

    cases = retrieve_applicable_cases(tmp_path, pack, top_k=3)
    text = format_applicable_retrieved_cases(cases)

    assert cases
    assert cases[0].applicability == "applicable"
    assert "strong_python_signal" in cases[0].matched_conditions
    assert "news_hidden_or_missing" in cases[0].matched_conditions
    assert "retrieved_cases_applicability" in text
    assert "applicability=applicable" in text
    assert "return_20d" not in text
    assert "gt_status" not in text


def test_retrieve_applicable_cases_does_not_promote_broad_meta_rag_case(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "memory" / "failure_case_ledger.csv",
        "\n".join(
            [
                "failure_id,source_round,task_mode,failure_pattern,countermeasure,status",
                "FAIL-BROAD,round,single_stock_watch,broad retrieved case RAG retrieved_cases overly generic case memory can be misleading counterevidence,tighten applicability before use,open",
            ]
        ),
    )
    pack = {
        "variant": "full_agent",
        "task_mode": "single_stock",
        "counter_evidence": "RAG 案例检索 误用",
        "retrieved_cases_context": "retrieved_cases_applicability: none",
    }

    cases = retrieve_applicable_cases(tmp_path, pack, top_k=3)
    text = format_applicable_retrieved_cases(cases)

    assert cases
    assert cases[0].case.case_id == "FAIL-BROAD"
    assert cases[0].applicability == "partial"
    assert "meta caution only" in text
    assert "applicability=applicable" not in text


def test_known_broad_case_id_override_is_partial_only(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "memory" / "failure_case_ledger.csv",
        "\n".join(
            [
                "failure_id,source_round,task_mode,failure_pattern,countermeasure,status",
                "FAIL-20260626-007,round,single_stock_watch,single_stock news missing financial_report_context weak peer confirmation caused repeated decision errors,review as checklist,open",
            ]
        ),
    )
    pack = {
        "variant": "full_agent_with_risk_review_queue",
        "task_mode": "single_stock",
        "news_features": {"news_missing_rate": 1.0},
        "financial_report_signal_summary": "financial_report no_event_in_window",
        "peer_context_features": {"peer_group_positive_breadth_20d": 0.2},
    }

    cases = retrieve_applicable_cases(tmp_path, pack, top_k=3)
    text = format_applicable_retrieved_cases(cases)

    assert cases
    assert cases[0].case.case_id == "FAIL-20260626-007"
    assert cases[0].applicability == "partial"
    assert "meta caution only" in text
    assert "applicability=applicable" not in text


def test_retrieve_applicable_cases_uses_risk_branch_context(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "memory" / "failure_case_ledger.csv",
        "\n".join(
            [
                "failure_id,source_round,task_mode,failure_pattern,countermeasure,status",
                "FAIL-BRANCH,round,single_stock_watch,single_stock overheat_reversal_friction_without_hard_event with weak peer confirmation raised bad observation weight,cap risk branch unless direct positive catalyst exists,open",
            ]
        ),
    )
    pack = {
        "variant": "full_agent_with_risk_review_queue",
        "task_mode": "single_stock",
        "peer_context_features": {"peer_group_positive_breadth_20d": 0.2},
        "quant_tool_summaries": [
            {
                "tool_id": "single_stock_risk_calibration_v2_review_queue",
                "primary_risk_branch": "overheat_reversal_friction_without_hard_event",
                "risk_branch_labels": ["peer_relative_lag"],
            }
        ],
    }

    cases = retrieve_applicable_cases(tmp_path, pack, top_k=3)
    text = format_applicable_retrieved_cases(cases)

    assert cases
    assert cases[0].case.case_id == "FAIL-BRANCH"
    assert cases[0].applicability == "applicable"
    assert "overheat_reversal_friction_without_hard_event" in cases[0].matched_conditions
    assert "weak_peer_confirmation" in cases[0].matched_conditions
    assert "applicability=applicable" in text


def test_no_recent_financial_event_is_not_financial_report_context(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "memory" / "failure_case_ledger.csv",
        "\n".join(
            [
                "failure_id,source_round,task_mode,failure_pattern,countermeasure,status",
                "FAIL-FIN,round,single_stock_watch,single_stock financial_report_only context added noise without ordinary confirmation,keep financial report as review note,open",
            ]
        ),
    )
    pack = {
        "variant": "full_agent_with_risk_review_queue",
        "task_mode": "single_stock",
        "financial_report_signal_summary": "no_recent_financial_report_event; not_disclosure_missing=true; status=no_event_in_window",
        "financial_report_features": {
            "financial_report_join_status": "no_event_in_window",
            "financial_report_event_count": 0,
        },
    }

    tags = _evidence_condition_tags(pack)
    cases = retrieve_applicable_cases(tmp_path, pack, top_k=3)
    text = format_applicable_retrieved_cases(cases)

    assert "financial_no_recent_event" in tags
    assert "financial_report_context" not in tags
    assert "financial_missing" not in tags
    assert cases
    assert cases[0].case.case_id == "FAIL-FIN"
    assert cases[0].applicability == "partial"
    assert "applicability=applicable" not in text
