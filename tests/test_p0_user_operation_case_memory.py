from __future__ import annotations

import pandas as pd

from scripts.build_p0_user_operation_case_memory import build_case_memory, validate_safe_ledger
from src.agent_training.case_memory_retriever import retrieve_applicable_cases, retrieve_cases


def test_build_case_memory_drops_future_metrics() -> None:
    examples = pd.DataFrame(
        [
            {
                "example_bucket": "false_positive_buy",
                "source_label": "panel",
                "valid_block": "H2026_1",
                "decision_date": "2026-04-07",
                "code": "000892",
                "name": "欢瑞世纪",
                "user_operation_suggestion": "试探买入/持有",
                "return_20d": -21.58,
                "target_cash20": -5.21,
                "error_type": "large_loss_buy",
                "news_missing_or_empty": True,
                "financial_missing_or_no_event": True,
                "peer_weak_or_lagging": False,
                "bookskill_weak_or_missing": False,
                "chip_overhang_or_trapped": False,
                "rag_failure_or_case_risk": False,
                "explicit_or_financial_hard_risk": False,
            }
        ]
    )

    ledger = build_case_memory(examples)
    validate_safe_ledger(ledger)
    text = ledger.to_csv(index=False)

    assert "return_20d" not in ledger.columns
    assert "target_cash20" not in ledger.columns
    assert "-21.58" not in text
    assert "news_missing_or_empty" in text
    assert "second check" in text


def test_case_memory_retriever_reads_p0_user_operation_ledger(tmp_path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "p0_user_operation_case_memory_ledger.csv").write_text(
        "\n".join(
            [
                "case_id,source_round,task_mode,case_bucket,case_pattern,visible_conditions,countermeasure,status,source_ref",
                "P0CASE-X,p0_user_operation_error_pattern_audit_v1,single_stock_watch,risk_false_veto_large_gain,single_stock false_veto_or_missed_positive case news_missing_or_empty financial_missing_or_no_event rag_failure_or_case_risk risk_review_or_wait_branch,news_missing_or_empty;financial_missing_or_no_event;rag_failure_or_case_risk;risk_review_or_wait_branch,RAG failure/case-risk should shrink or trigger review only; do not use it as a standalone veto.,open_false_veto_case_do_not_blind_zero,panel|H2026_1|2026-03-27|001339|智微智能",
            ]
        ),
        encoding="utf-8",
    )

    cases = retrieve_cases(tmp_path, "single_stock RAG failure news missing financial no_event", top_k=3)

    assert cases
    assert cases[0].case_id == "P0CASE-X"
    assert "return_20d" not in str(cases[0].output)


def test_p0_case_memory_can_be_applicable_to_evidence_pack(tmp_path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "p0_user_operation_case_memory_ledger.csv").write_text(
        "\n".join(
            [
                "case_id,source_round,task_mode,case_bucket,case_pattern,visible_conditions,countermeasure,status,source_ref",
                "P0CASE-Y,p0_user_operation_error_pattern_audit_v1,single_stock_watch,false_positive_buy,single_stock small_entry_branch false_positive_or_large_loss case news_missing_or_empty financial_no_recent_event weak_peer_confirmation small_entry_branch,news_missing_or_empty;financial_missing_or_no_event;peer_weak_or_lagging;small_entry_branch,When news financial and peer are all weak or missing cap position and require one confirming channel before sizing above a small trial.,open_failure_case_requires_confirmation,panel|H2026_1|2026-04-07|000892|欢瑞世纪",
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

    applicable = retrieve_applicable_cases(tmp_path, pack, top_k=3)

    assert applicable
    assert applicable[0].case.case_id == "P0CASE-Y"
    assert applicable[0].applicability == "applicable"
    assert "news_hidden_or_missing" in applicable[0].matched_conditions
