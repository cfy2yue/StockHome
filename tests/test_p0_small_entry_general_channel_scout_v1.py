from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_p0_small_entry_general_channel_scout_v1 import (
    assert_no_future_fields,
    attach_flags,
    build_agent_preview,
    build_rulebook,
    enrich_summary,
    evaluate_rules,
    has_strategy,
    promotion_status,
)


def _detail() -> pd.DataFrame:
    rows = []
    for block, ret in [("H2024_1", 4.0), ("H2025_1", 5.0), ("H2026_1", 7.0)]:
        for idx in range(10):
            rows.append(
                {
                    "date": f"2026-01-{idx + 1:02d}",
                    "code": f"000{idx:03d}",
                    "name": f"测试{idx}",
                    "target_block": block,
                    "frequency": "weekly_tuesday",
                    "operation_action": "small_buy_hold",
                    "triggered_skill_ids": "PPS-M-003",
                    "return_20d": ret if idx < 8 else -2.0,
                    "all_triggered_grounded": True,
                    "weak_skill_count": 0,
                }
            )
    return pd.DataFrame(rows)


def _joined() -> pd.DataFrame:
    detail = _detail()
    rows = []
    for _, row in detail.iterrows():
        rows.append(
            {
                "date": row["date"],
                "code": row["code"],
                "news_count_30d": 3,
                "news_warning_score": 0.0,
                "news_opportunity_score": 0.0,
                "news_missing_rate": 0.0,
                "official_confirmation_score": 1.0,
                "announcement_materiality_score": 0.7,
                "financial_report_event_count": 0,
                "financial_report_missing_rate": 1.0,
                "financial_report_join_status": "no_event_in_window",
                "peer_group_positive_breadth_20d": 0.2,
                "peer_relative_to_group_20d": -1.0,
                "tushare_industry_positive_breadth_20d": 0.2,
                "tushare_industry_relative_return_20d": -1.0,
                "tushare_area_positive_breadth_20d": 0.2,
                "prior_return_20d": -8.0,
                "rsi14": 32.0,
                "drawdown60": -12.0,
                "close_above_ma200": False,
                "ma200_slope20": 0.0,
                "lower_support": 0.2,
                "upper_overhang": 0.2,
                "winner_rate_pct": 20.0,
            }
        )
    return pd.DataFrame(rows)


def test_has_strategy_uses_exact_tokens() -> None:
    assert has_strategy("PPS-M-003;PPS-Q-017", "PPS-M-003")
    assert not has_strategy("PPS-M-0039;PPS-Q-017", "PPS-M-003")


def test_general_channel_scout_can_promote_stable_candidate() -> None:
    data = attach_flags(_detail(), _joined())
    metrics, block_metrics = evaluate_rules(data, build_rulebook())
    summary = enrich_summary(metrics, block_metrics)

    row = summary[summary["rule_id"].eq("pps_m003_tuesday_clean_chip")].iloc[0]

    assert row["prior_evaluable_blocks"] == 2
    assert row["h2026_rows"] == 10
    assert row["promotion_status"] in {"yellow_candidate_for_more_panel", "diagnostic_or_reject"}


def test_promotion_status_green_requires_prior_and_h2026() -> None:
    row = {
        "prior_evaluable_blocks": 2,
        "prior_selected_rows_sum": 40,
        "prior_delta_pos_hit": 1.0,
        "prior_delta_avg_hit": 1.0,
        "h2026_rows": 30,
        "h2026_pos20": 0.7,
        "h2026_avg20_pp": 6.0,
        "h2026_loss_gt5": 0.1,
    }
    assert promotion_status(row) == "green_candidate_for_ds_sample"


def test_agent_preview_excludes_future_keys() -> None:
    data = attach_flags(_detail(), _joined())
    metrics, block_metrics = evaluate_rules(data, build_rulebook())
    summary = enrich_summary(metrics, block_metrics)
    preview = build_agent_preview(summary, data, build_rulebook(), max_rows=5)

    for item in preview:
        assert_no_future_fields(item)
    with pytest.raises(ValueError):
        assert_no_future_fields({"return_20d": 1.0})
