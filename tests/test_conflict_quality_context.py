from __future__ import annotations

import pandas as pd

from src.agent_training.conflict_quality_context import (
    attach_conflict_quality_contexts,
    build_walkforward_conflict_quality_rulebooks,
    render_conflict_quality_context,
)


def _row(index: int, *, date: str, code: str, kline20: float, ret20: float) -> dict:
    return {
        "date": date,
        "code": code,
        "name": f"测试{index}",
        "gt_status": "evaluated",
        "return_20d": ret20,
        "kline_return_20d": kline20,
        "kline_return_60d": kline20 / 2,
        "corr_peer_avg_return_20d": 0.0,
        "lower_support": 0.2,
        "chip_concentration": 0.5,
        "cost_band_width": 0.6,
        "upper_overhang": 0.4,
        "winner_rate_pct": 30,
        "neg_winner_rate": -30,
        "news_missing_rate": 0.2,
        "news_warning_score": 0.1,
        "news_opportunity_score": 0.2,
        "financial_report_join_status": "no_event_in_window",
        "financial_quality_risk_score": 0.0,
        "financial_surprise_score": 0.0,
        "tushare_industry_positive_breadth_20d": 0.7,
        "tushare_industry_relative_return_20d": 1.0,
        "triggered_skills": "CORE_TREND_001",
    }


def test_walkforward_conflict_quality_context_uses_prior_blocks_only() -> None:
    rows = []
    for i in range(40):
        rows.append(_row(i, date="2023-03-03", code=f"{i + 1:06d}", kline20=-30 if i >= 30 else 5, ret20=8 if i >= 30 else -2))
    for i in range(10):
        rows.append(_row(i + 40, date="2023-08-04", code=f"{i + 101:06d}", kline20=-30, ret20=-20))
    frame = pd.DataFrame(rows)

    rulebooks = build_walkforward_conflict_quality_rulebooks(frame, valid_blocks=["H2023_2"], min_rows=2)
    pack = {
        "valid_block": "H2023_2",
        "kline_features": {"kline_return_20d": -30.0, "kline_return_60d": -15.0, "kline_atr20_pct": 2.0},
        "chip_features": {"lower_support": 0.2, "upper_overhang": 0.4, "cost_band_width": 0.6},
        "news_features": {"news_missing_rate": 0.2, "news_warning_score": 0.1, "news_opportunity_score": 0.2},
        "financial_report_features": {"financial_report_join_status": "no_event_in_window", "financial_quality_risk_score": 0.0, "financial_surprise_score": 0.0},
        "peer_context_features": {"tushare_industry_positive_breadth_20d": 0.7, "tushare_industry_relative_return_20d": 1.0},
        "book_skill_candidates": [{"strategy_id": "CORE_TREND_001", "source_book": "测试"}],
    }

    context = render_conflict_quality_context(pack, rulebooks["H2023_2"])

    assert "walk_forward_prior_only" in context
    assert "kline_risk=" in context
    assert "H2023_2" in context
    assert "return_20d" not in context
    assert "avg20" not in context
    assert "-20" not in context


def test_attach_conflict_quality_contexts_mutates_packs() -> None:
    packs = [{"valid_block": "H2023_2", "kline_features": {}, "news_features": {}, "financial_report_features": {}}]
    rulebooks = {"H2023_2": {"status": "insufficient_prior_data", "valid_block": "H2023_2"}}

    attach_conflict_quality_contexts(packs, rulebooks)

    assert packs[0]["conflict_quality_context"] == "none"
