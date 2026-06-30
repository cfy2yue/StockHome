from __future__ import annotations

import pandas as pd

from src.agent_training.promote_context import (
    attach_promote_contexts,
    build_walkforward_promote_rulebooks,
    render_promote_context,
)


def _row(index: int, *, date: str, code: str, kline20: float, kline60: float, ret20: float) -> dict:
    return {
        "date": date,
        "code": code,
        "name": f"测试{index}",
        "gt_status": "evaluated",
        "return_20d": ret20,
        "kline_return_20d": kline20,
        "kline_return_60d": kline60,
        "corr_peer_avg_return_20d": 0.0,
        "lower_support": 0.25,
        "chip_concentration": 0.5,
        "cost_band_width": 0.8,
        "upper_overhang": 0.7,
        "winner_rate_pct": 30,
        "neg_winner_rate": -30,
        "news_count_30d": 1,
        "news_missing_rate": 0.2,
        "news_evidence_quality": 0.8,
        "news_warning_score": 0.1,
        "news_opportunity_score": 0.5,
        "financial_report_join_status": "event_window_matched",
        "financial_report_missing_rate": 0.1,
        "financial_report_event_count": 1,
        "financial_quality_risk_score": 0.1,
        "financial_surprise_score": 0.2,
        "tushare_industry_positive_breadth_20d": 0.7,
        "tushare_industry_relative_return_20d": 1.0,
        "triggered_skills": "CORE_TREND_001",
    }


def test_walkforward_promote_context_renders_prior_only_status() -> None:
    rows = []
    for i in range(120):
        rows.append(_row(i, date="2023-03-03", code=f"{i + 1:06d}", kline20=-25, kline60=-40, ret20=8))
    for i in range(20):
        rows.append(_row(i + 200, date="2023-08-04", code=f"{i + 301:06d}", kline20=-25, kline60=-40, ret20=-20))
    frame = pd.DataFrame(rows)
    rulebooks = build_walkforward_promote_rulebooks(
        frame,
        valid_blocks=["H2023_2"],
        score_quantile_min=0.0,
        min_rows=20,
        min_stocks=10,
    )
    pack = {
        "valid_block": "H2023_2",
        "kline_features": {"kline_return_20d": -25.0, "kline_return_60d": -40.0, "kline_atr20_pct": 2.0},
        "chip_features": {"lower_support": 0.25, "upper_overhang": 0.7, "cost_band_width": 0.8},
        "news_features": {"news_count_30d": 1, "news_missing_rate": 0.2, "news_evidence_quality": 0.8, "news_warning_score": 0.1, "news_opportunity_score": 0.5},
        "financial_report_features": {"financial_report_join_status": "event_window_matched", "financial_report_missing_rate": 0.1, "financial_report_event_count": 1, "financial_quality_risk_score": 0.1, "financial_surprise_score": 0.2},
        "peer_context_features": {"tushare_industry_positive_breadth_20d": 0.7, "tushare_industry_relative_return_20d": 1.0},
        "book_skill_candidates": [{"strategy_id": "CORE_TREND_001"}],
    }

    context = render_promote_context(pack, rulebooks["H2023_2"])

    assert "walk_forward_prior_only" in context
    assert "kline_reversal_friction_confirmed=" in context
    assert "return_20d" not in context
    assert "pool_excess" not in context
    assert "-20" not in context


def test_attach_promote_contexts_sets_none_without_prior_data() -> None:
    packs = [{"valid_block": "H2023_2", "kline_features": {}, "news_features": {}, "financial_report_features": {}}]
    attach_promote_contexts(packs, {"H2023_2": {"status": "insufficient_prior_data", "valid_block": "H2023_2"}})
    assert packs[0]["promote_context"] == "none"
