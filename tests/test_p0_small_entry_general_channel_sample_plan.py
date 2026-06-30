from __future__ import annotations

import pandas as pd

from scripts.audit_p0_small_entry_general_channel_scout_v1 import attach_flags, build_rulebook
from scripts.build_p0_small_entry_general_channel_sample_plan_v1 import build_sample_plan


def _detail() -> pd.DataFrame:
    rows = []
    for block, date_prefix in [("H2024_1", "2024-03"), ("H2025_1", "2025-03"), ("H2026_1", "2026-03")]:
        for idx in range(8):
            rows.append(
                {
                    "date": f"{date_prefix}-{idx + 1:02d}",
                    "code": f"000{idx:03d}",
                    "name": f"测试{idx}",
                    "target_block": block,
                    "frequency": "weekly_tuesday",
                    "operation_action": "small_buy_hold",
                    "triggered_skill_ids": "PPS-M-003;PPS-Q-017",
                    "return_20d": 5.0,
                    "book_score": 1.2,
                    "triggered_skill_count": 2,
                    "grounded_skill_count": 2,
                    "weak_skill_count": 0,
                    "grounded_skill_ids": "PPS-M-003;PPS-Q-017",
                    "weak_skill_ids": "",
                    "all_triggered_grounded": True,
                }
            )
    return pd.DataFrame(rows)


def _joined() -> pd.DataFrame:
    rows = []
    for _, row in _detail().iterrows():
        rows.append(
            {
                "date": row["date"],
                "code": row["code"],
                "news_count_30d": 3,
                "news_warning_score": 0.0,
                "news_opportunity_score": 0.5,
                "news_missing_rate": 0.0,
                "official_confirmation_score": 1.0,
                "announcement_materiality_score": 0.5,
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


def test_general_channel_sample_plan_is_safe_and_stratified() -> None:
    data = attach_flags(_detail(), _joined())
    rules = {rule["rule_id"]: rule for rule in build_rulebook()}

    plan, audit = build_sample_plan(
        data,
        rules,
        rule_ids=["news_financial_clean_chip_pullback", "pps_m003_tuesday"],
        blocks={"H2024_1", "H2025_1", "H2026_1"},
        max_per_rule_block=2,
        max_rows=12,
        exclude_keys=set(),
    )

    assert len(plan) == 12
    assert not plan.duplicated(["date", "code"]).any()
    assert {"return_20d", "positive_20d", "loss_gt5"}.isdisjoint(plan.columns)
    assert "return_20d" in audit.columns
    assert set(plan["task_mode"]) == {"single_stock"}
    assert set(plan["operation_action"]) == {"small_buy_hold"}


def test_general_channel_sample_plan_respects_exclude_keys() -> None:
    data = attach_flags(_detail(), _joined())
    rules = {rule["rule_id"]: rule for rule in build_rulebook()}
    excluded = {(str(data.iloc[0]["date"]), str(data.iloc[0]["code"]).zfill(6))}

    plan, _ = build_sample_plan(
        data,
        rules,
        rule_ids=["news_financial_clean_chip_pullback"],
        blocks={"H2024_1"},
        max_per_rule_block=8,
        max_rows=8,
        exclude_keys=excluded,
    )

    assert excluded.isdisjoint(set(zip(plan["date"].astype(str), plan["code"].astype(str).str.zfill(6))))
    assert len(plan) == 7
