from __future__ import annotations

import json

from scripts.audit_evidence_pack_channel_coverage import audit_evidence_pack_channel_coverage


def test_channel_coverage_counts_grounded_book_skill_source_detail(tmp_path) -> None:
    path = tmp_path / "packs.jsonl"
    path.write_text(
        json.dumps(
            {
                "code": "000001",
                "decision_date": "2026-01-05",
                "task_mode": "single_stock_watch",
                "variant": "grounded_full",
                "python_features": {"prior_return_20d": -3.0},
                "quant_tool_summaries": [{"tool_id": "date_regime_gate_minimal_v1"}],
                "quant_tool_signal_summary": "date_regime_gate_minimal_v1 status=observe",
                "kline_features": {"kline_return_20d": -12.0},
                "peer_context_features": {},
                "news_features": {"news_missing_rate": 0.8},
                "news_semantic_questionnaire": {},
                "news_branch_case_context": {"primary_news_branch": "soft_gap"},
                "analogue_case_context": [{"tool_id": "analogue_case_context"}],
                "nonprice_risk_overlay_context": [{"tool_id": "nonprice_risk_overlay"}],
                "financial_report_features": {"financial_report_event_count": 0},
                "book_skill_candidates": [
                    {
                        "strategy_id": "PPS-Q-017",
                        "source_status": "grounded",
                        "source_book": "专业投机原理",
                        "page_range": "OCR_PAGE 1-2",
                    }
                ],
                "memory_context": "accepted: no_news bad exposure",
                "retrieved_cases_context": "none",
                "counter_evidence": "新闻覆盖不足",
                "data_missing_flags": "",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    detail, summary = audit_evidence_pack_channel_coverage(path)

    assert len(detail) == 1
    assert bool(detail.iloc[0]["book_skill_has_source_detail"])
    assert int(detail.iloc[0]["book_skill_grounded_count"]) == 1
    coverage = dict(zip(summary["channel"], summary["coverage_rate"]))
    assert coverage["book_skill_candidates"] == 1.0
    assert coverage["quant_tool_summaries"] == 1.0
    assert coverage["quant_tool_signal_summary"] == 1.0
    assert coverage["news_branch_case_context"] == 1.0
    assert coverage["analogue_case_context"] == 1.0
    assert coverage["nonprice_risk_overlay_context"] == 1.0
    assert coverage["peer_context_features"] == 0.0
