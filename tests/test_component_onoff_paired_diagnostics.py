from __future__ import annotations

import pandas as pd

from scripts.audit_component_onoff_paired_diagnostics import build_paired_detail, classify_pair, summarize_pairs, _bank_return_20d


def test_classify_pair_directions() -> None:
    assert classify_pair(0.1, 5.0) == "raised_positive"
    assert classify_pair(0.1, -5.0) == "raised_negative"
    assert classify_pair(-0.1, 5.0) == "lowered_positive"
    assert classify_pair(-0.1, -5.0) == "lowered_negative"
    assert classify_pair(0.0, 5.0) == "unchanged"


def test_build_paired_detail_and_summary() -> None:
    joined = pd.DataFrame(
        [
            _row("full_agent", "portfolio_pool", "2026-01-02", "000001", 0.2, 10.0),
            _row("no_context", "portfolio_pool", "2026-01-02", "000001", 0.1, 10.0),
            _row("full_agent", "portfolio_pool", "2026-01-02", "000002", 0.2, -8.0),
            _row("no_context", "portfolio_pool", "2026-01-02", "000002", 0.1, -8.0),
            _row("full_agent", "single_stock", "2026-01-03", "000003", 0.0, 6.0),
            _row("no_context", "single_stock", "2026-01-03", "000003", 0.1, 6.0),
            _row("full_agent", "single_stock", "2026-01-03", "000004", 0.0, -6.0),
            _row("no_context", "single_stock", "2026-01-03", "000004", 0.1, -6.0),
        ]
    )

    detail = build_paired_detail(joined, treatment="full_agent", control="no_context")

    assert list(detail["pair_direction"]) == [
        "raised_positive",
        "raised_negative",
        "lowered_positive",
        "lowered_negative",
    ]
    first = detail.iloc[0]
    expected_delta = 0.1 * (10.0 - _bank_return_20d())
    assert round(first["delta_cash_adjusted_return_20d"], 6) == round(expected_delta, 6)

    summary = summarize_pairs(detail, ["task_mode"]).set_index("task_mode")
    assert summary.loc["portfolio_pool", "raised_positive"] == 1
    assert summary.loc["portfolio_pool", "raised_negative"] == 1
    assert summary.loc["single_stock", "lowered_positive"] == 1
    assert summary.loc["single_stock", "lowered_negative"] == 1


def _row(variant: str, task: str, date: str, code: str, weight: float, ret: float) -> dict:
    return {
        "variant": variant,
        "task_mode": task,
        "valid_block": "H2026_1",
        "decision_date": date,
        "code": code,
        "name": "测试",
        "sample_panel_id": "panel_01",
        "sample_rank_in_panel": 1,
        "research_grade": "放入观察",
        "simulated_action": "保持观察",
        "simulated_weight_change_num": weight,
        "cash_adjusted_return_20d": weight * ret,
        "return_20d": ret,
    }

