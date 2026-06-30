from __future__ import annotations

import pandas as pd

from scripts.build_p0_news_channel_case_memory import build_news_case_memory, validate_safe_ledger
from src.agent_training.case_memory_retriever import (
    _evidence_condition_tags,
    retrieve_applicable_cases,
    retrieve_cases,
)


def test_build_news_case_memory_drops_future_metrics() -> None:
    examples = pd.DataFrame(
        [
            {
                "example_bucket": "missing_news_false_positive",
                "source_label": "panel",
                "valid_block": "H2026_1",
                "decision_date": "2026-04-14",
                "code": "000816",
                "name": "智慧农业",
                "user_operation_suggestion": "试探买入",
                "target_position": 0.55,
                "return_20d": -11.69,
                "target_cash20": -6.32,
                "news_missing_questionnaire": True,
                "news_neutral_no_catalyst": False,
                "news_opportunity_or_catalyst": False,
                "news_hard_warning": False,
                "financial_missing_or_no_event": True,
                "peer_weak_or_lagging": True,
            }
        ]
    )

    ledger = build_news_case_memory(examples)
    validate_safe_ledger(ledger)
    text = ledger.to_csv(index=False)

    assert "return_20d" not in ledger.columns
    assert "target_cash20" not in ledger.columns
    assert "-11.69" not in text
    assert "news_missing_no_hard_warning" in text
    assert "cap position" in text


def test_news_case_memory_retrieves_missing_news_case(tmp_path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "p0_news_channel_case_memory_ledger.csv").write_text(
        "\n".join(
            [
                "case_id,source_round,task_mode,case_bucket,case_pattern,visible_conditions,countermeasure,status,source_ref",
                "P0NEWS-X,p0_news_channel_policy_audit_v1,single_stock_watch,missing_news_risk_false_veto,single_stock news_channel news missing no hard warning false veto news_missing_no_hard_warning financial_missing_or_no_event peer_weak_or_lagging risk_review_or_wait_branch,news_missing_no_hard_warning;financial_missing_or_no_event;peer_weak_or_lagging;news_missing_questionnaire;small_entry_branch,Do not blindly zero because news is missing; keep a low observation position or explicit re-entry trigger when no hard counter exists.,open_news_missing_false_veto_case,panel|H2026_1|2026-04-03|000980|众泰汽车",
            ]
        ),
        encoding="utf-8",
    )
    pack = {
        "task_mode": "single_stock",
        "operation_action": "small_buy_hold",
        "news_features": {"news_missing_rate": 1.0},
        "financial_report_features": {"financial_report_join_status": "no_event_in_window", "financial_report_event_count": 0},
        "peer_context_features": {"peer_group_positive_breadth_20d": 0.2},
        "book_skill_candidates": [],
    }

    cases = retrieve_applicable_cases(tmp_path, pack, top_k=3)

    assert cases
    assert cases[0].case.case_id == "P0NEWS-X"
    assert cases[0].applicability == "applicable"
    assert "news_hidden_or_missing" in cases[0].matched_conditions


def test_news_questionnaire_tags_opportunity_and_hard_warning(tmp_path) -> None:
    pack = {
        "task_mode": "single_stock",
        "news_semantic_questionnaire": {
            "ds_news_risk_score": 0.75,
            "ds_news_opportunity_score": 0.65,
            "ds_news_uncertainty_score": 0.8,
        },
    }

    tags = _evidence_condition_tags(pack)

    assert "news_hard_warning" in tags
    assert "explicit_hard_negative_event" in tags
    assert "news_opportunity_context" in tags
    assert "high_news_uncertainty" in tags


def test_news_case_memory_retrieve_cases_can_find_opportunity_context(tmp_path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "p0_news_channel_case_memory_ledger.csv").write_text(
        "\n".join(
            [
                "case_id,source_round,task_mode,case_bucket,case_pattern,visible_conditions,countermeasure,status,source_ref",
                "P0NEWS-OPP,p0_news_channel_policy_audit_v1,single_stock_watch,opportunity_success_buy,single_stock news_channel news opportunity context needs quant peer chip bookskill confirmation news_opportunity_context small_entry_branch,news_opportunity_context;news_opportunity_or_catalyst;small_entry_branch,Treat opportunity news as supportive context only; require quant peer chip BookSkill or financial confirmation before increasing size.,accepted_opportunity_support_only_not_alpha,panel|H2026_1|2026-04-10|000700|模塑科技",
            ]
        ),
        encoding="utf-8",
    )

    cases = retrieve_cases(tmp_path, "single_stock news opportunity context BookSkill confirmation", top_k=3)

    assert cases
    assert cases[0].case_id == "P0NEWS-OPP"
    assert "future" not in str(cases[0].output).lower()
