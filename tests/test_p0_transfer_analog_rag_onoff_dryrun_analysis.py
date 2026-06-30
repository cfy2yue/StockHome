from __future__ import annotations

from scripts.analyze_p0_transfer_analog_rag_onoff_dryrun import (
    build_status,
    build_visibility_detail,
    build_visibility_summary,
)


def _pack(variant: str) -> dict:
    pack = {
        "variant": variant,
        "task_mode": "single_stock",
        "valid_block": "H2026_1",
        "decision_date": "2026-04-03",
        "code": "000877",
        "analogue_case_context": [
            {
                "date": "2026-04-03",
                "code": "000877",
                "analog_id": "analog_k15_min10",
                "gate_id": "chip_support_plus_analog065",
            }
        ],
        "chip_features": {"lower_support": 0.2},
        "financial_report_features": {"financial_report_event_count": 1},
        "news_features": {"news_warning_score": 0.1},
        "peer_context_features": {"tushare_industry": "test"},
        "book_skill_candidates": [{"strategy_id": "PPS-Q-017"}],
        "quant_tool_summaries": [{"tool_id": "kline_peer_chip"}],
        "python_features": {"prior_return_20d": -5},
        "kline_features": {"kline_return_20d": -5},
    }
    if variant == "no_analogue_case_context":
        pack["analogue_case_context"] = []
    if variant == "no_chip_context":
        pack["chip_features"] = {}
    if variant == "no_financial_report":
        pack["financial_report_features"] = {}
    if variant == "no_news":
        pack["news_features"] = {}
    if variant == "no_peer":
        pack["peer_context_features"] = {}
    if variant == "no_bookskill":
        pack["book_skill_candidates"] = []
    if variant == "no_quant_tools":
        pack["quant_tool_summaries"] = []
    return pack


def test_onoff_visibility_analysis_passes_expected_component_isolation() -> None:
    variants = [
        "full_agent",
        "no_analogue_case_context",
        "no_chip_context",
        "no_financial_report",
        "no_news",
        "no_peer",
        "no_bookskill",
        "no_quant_tools",
    ]
    detail = build_visibility_detail([_pack(variant) for variant in variants])
    summary = build_visibility_summary(detail)

    assert build_status(summary) == "pass"
    assert summary.loc[summary["variant"].eq("full_agent"), "row_level_analogue_visible_sum"].iloc[0] == 1
    assert summary.loc[summary["variant"].eq("no_analogue_case_context"), "analogue_visible_sum"].iloc[0] == 0
    assert summary.loc[summary["variant"].eq("no_chip_context"), "chip_visible_sum"].iloc[0] == 0
    assert summary.loc[summary["variant"].eq("no_financial_report"), "financial_visible_sum"].iloc[0] == 0
