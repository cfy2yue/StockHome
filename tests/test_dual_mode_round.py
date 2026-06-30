from __future__ import annotations

import pandas as pd

from src.agent_training.dual_mode_round import build_dual_mode_evidence_packs, build_walkforward_evidence_packs, dual_mode_metrics, dual_mode_step_metrics, load_ground_truth, select_dual_mode_rows, _portfolio_score, _apply_portfolio_row_gate


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2023-07-04",
                "code": "1",
                "name": "测试A",
                "gt_status": "evaluated",
                "return_20d": 5.0,
                "relative_strength_rank": 0.9,
                "counter_score": 8,
                "close_above_ma200": True,
                "news_count_30d": 2,
                "news_risk_event_score_30d": 0,
                "triggered_skills": "CORE_TREND_001",
                "data_gaps": "",
            },
            {
                "date": "2023-07-07",
                "code": "2",
                "name": "测试B",
                "gt_status": "evaluated",
                "return_20d": -3.0,
                "relative_strength_rank": 0.4,
                "counter_score": 4,
                "close_above_ma200": False,
                "news_count_30d": 4,
                "news_risk_event_score_30d": 2,
                "atr20_pct": 6,
                "triggered_skills": "",
                "data_gaps": "financial_publish_date_missing",
            },
            {
                "date": "2024-01-05",
                "code": "3",
                "name": "测试C",
                "gt_status": "evaluated",
                "return_20d": 8.0,
                "relative_strength_rank": 0.8,
                "counter_score": 7,
            },
            {
                "date": "2024-07-05",
                "code": "4",
                "name": "测试D",
                "gt_status": "evaluated",
                "return_20d": 2.0,
                "relative_strength_rank": 0.7,
                "counter_score": 6,
            },
            {
                "date": "2023-07-11",
                "code": "1",
                "name": "测试A",
                "gt_status": "evaluated",
                "return_20d": -1.0,
                "relative_strength_rank": 1.0,
                "counter_score": 9,
                "close_above_ma200": True,
                "news_count_30d": 0,
                "prior_return_20d": 150,
                "rsi14": 90,
                "data_gaps": "financial_publish_date_missing",
                "triggered_skills": "DOW-B-017",
            },
            {
                "date": "2023-07-14",
                "code": "5",
                "name": "测试E",
                "gt_status": "evaluated",
                "return_20d": 3.0,
                "relative_strength_rank": 0.7,
                "counter_score": 6,
                "close_above_ma200": True,
                "news_count_30d": 1,
            },
        ]
    )


def test_select_dual_mode_rows_separates_modes() -> None:
    rows = select_dual_mode_rows(_frame(), limit_per_mode=1, valid_block="H2023_2")
    assert set(rows) == {"portfolio_pool", "single_stock"}
    assert len(rows["portfolio_pool"]) == 1
    assert len(rows["single_stock"]) == 1
    assert rows["single_stock"].iloc[0]["code"] == "2"


def test_load_ground_truth_can_use_custom_event_features(tmp_path) -> None:
    gt_path = tmp_path / "gt.csv"
    event_path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "date": "2025-04-25",
                "code": "000089",
                "name": "测试A",
                "gt_status": "evaluated",
                "return_20d": 1.0,
            }
        ]
    ).to_csv(gt_path, index=False)
    pd.DataFrame(
        [
            {
                "code": "000089",
                "decision_date": "2025-04-25",
                "available_at": "2025-04-25 15:00:00",
                "event_count": 3,
                "news_warning_score": 0.5,
                "announcement_materiality_score": 0.7,
                "news_timestamp_quality": 0.7,
                "news_evidence_quality": 0.9,
            }
        ]
    ).to_csv(event_path, index=False)

    joined = load_ground_truth([gt_path], event_features_path=event_path, financial_report_features_path=None)
    plain = load_ground_truth([gt_path], event_features_path=None, financial_report_features_path=None)

    assert joined.iloc[0]["news_event_table_join_status"] == "event_window_matched"
    assert joined.iloc[0]["news_warning_score"] == 0.5
    assert "news_event_table_join_status" not in plain.columns


def test_load_ground_truth_can_use_custom_financial_report_features_asof(tmp_path) -> None:
    gt_path = tmp_path / "gt.csv"
    financial_path = tmp_path / "financial.csv"
    pd.DataFrame(
        [
            {"date": "2025-04-25", "code": "000089", "gt_status": "evaluated", "return_20d": 1.0},
            {"date": "2025-04-26", "code": "000089", "gt_status": "evaluated", "return_20d": 2.0},
        ]
    ).to_csv(gt_path, index=False)
    pd.DataFrame(
        [
            {
                "code": "000089",
                "decision_date": "2025-04-26",
                "available_at": "2025-04-26 00:00:00",
                "financial_report_event_count": 1,
                "financial_report_materiality_score": 0.8,
                "financial_quality_risk_score": 0.7,
                "financial_surprise_score": -0.3,
                "financial_disclosure_quality_score": 0.6,
                "financial_report_missing_rate": 0.0,
                "financial_report_latest_period": "20241231",
                "financial_report_event_types": "annual_report",
                "source_type": "paid_standardized",
                "source_name": "unit_test",
            }
        ]
    ).to_csv(financial_path, index=False)

    joined = load_ground_truth([gt_path], event_features_path=None, financial_report_features_path=financial_path)
    by_date = {row["date"]: row for _, row in joined.iterrows()}

    assert by_date["2025-04-25"]["financial_report_join_status"] == "no_event_in_window"
    assert by_date["2025-04-25"]["financial_report_missing_rate"] == 1.0
    assert by_date["2025-04-26"]["financial_report_join_status"] == "event_window_matched"
    assert by_date["2025-04-26"]["financial_quality_risk_score"] == 0.7


def test_load_ground_truth_can_merge_point_in_time_kline_and_peer_features(tmp_path) -> None:
    gt_path = tmp_path / "gt.csv"
    kline_path = tmp_path / "kline.csv"
    peer_path = tmp_path / "peer.csv"
    pd.DataFrame(
        [
            {"date": "2025-04-25", "code": "000089", "gt_status": "evaluated", "return_20d": 1.0},
        ]
    ).to_csv(gt_path, index=False)
    pd.DataFrame(
        [
            {"date": "2025-04-25", "code": "000089", "kline_return_20d": -12.0, "kline_atr20_pct": 3.0},
        ]
    ).to_csv(kline_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-04-25",
                "code": "000089",
                "tushare_industry": "测试行业",
                "tushare_industry_relative_return_20d": 2.5,
                "tushare_industry_positive_breadth_20d": 0.7,
            },
        ]
    ).to_csv(peer_path, index=False)

    joined = load_ground_truth(
        [gt_path],
        event_features_path=None,
        financial_report_features_path=None,
        kline_features_path=kline_path,
        tushare_peer_features_path=peer_path,
    )

    assert joined.iloc[0]["kline_return_20d"] == -12.0
    assert joined.iloc[0]["tushare_industry"] == "测试行业"
    assert joined.iloc[0]["tushare_industry_relative_return_20d"] == 2.5


def test_load_ground_truth_can_merge_point_in_time_chip_core_features(tmp_path) -> None:
    gt_path = tmp_path / "gt.csv"
    chip_path = tmp_path / "chip.csv"
    pd.DataFrame(
        [
            {"date": "2025-04-25", "code": "000089", "gt_status": "evaluated", "return_20d": 1.0},
        ]
    ).to_csv(gt_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-04-25",
                "code": "000089",
                "lower_support": 0.28,
                "chip_concentration": 0.12,
                "cost_band_width": 0.40,
                "upper_overhang": 0.20,
                "winner_rate_pct": 35.0,
                "neg_winner_rate": -35.0,
                "chip_core_source_type": "paid_standardized",
                "chip_core_source_name": "unit_test_chip",
            },
        ]
    ).to_csv(chip_path, index=False)

    joined = load_ground_truth(
        [gt_path],
        event_features_path=None,
        financial_report_features_path=None,
        chip_core_features_path=chip_path,
    )

    assert joined.iloc[0]["lower_support"] == 0.28
    assert joined.iloc[0]["chip_core_source_name"] == "unit_test_chip"


def test_positive_confirmation_row_gate_filters_hard_conflicts() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-03",
                "code": "000001",
                "news_count_30d": 2,
                "news_missing_rate": 0.1,
                "news_evidence_quality": 0.8,
                "news_warning_score": 0.1,
                "news_opportunity_score": 0.5,
                "financial_report_join_status": "no_event_in_window",
                "financial_report_missing_rate": 1.0,
                "financial_report_event_count": 0,
                "tushare_industry_positive_breadth_20d": 0.7,
                "tushare_industry_relative_return_20d": 1.5,
                "lower_support": 0.2,
                "upper_overhang": 0.4,
                "cost_band_width": 0.6,
                "kline_return_20d": 1.0,
                "kline_return_60d": 3.0,
                "kline_atr20_pct": 2.0,
                "triggered_skills": "CORE_TREND_001",
            },
            {
                "date": "2025-01-03",
                "code": "000002",
                "news_count_30d": 2,
                "news_missing_rate": 0.1,
                "news_evidence_quality": 0.8,
                "news_warning_score": 0.1,
                "news_opportunity_score": 0.5,
                "financial_report_join_status": "no_event_in_window",
                "financial_report_missing_rate": 1.0,
                "financial_report_event_count": 0,
                "tushare_industry_positive_breadth_20d": 0.3,
                "tushare_industry_relative_return_20d": -2.0,
                "lower_support": 0.2,
                "upper_overhang": 1.7,
                "cost_band_width": 1.8,
                "kline_return_20d": -25.0,
                "kline_return_60d": -40.0,
                "kline_atr20_pct": 13.0,
                "triggered_skills": "CORE_TREND_001",
            },
        ]
    )

    selected = _apply_portfolio_row_gate(frame, "positive_confirmation_min2_no_hard")

    assert selected["code"].tolist() == ["000001"]


def test_kline_reversal_friction_sampler_requires_confirmation_and_chip_support() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-03",
                "code": "000001",
                "news_count_30d": 1,
                "news_missing_rate": 0.1,
                "news_evidence_quality": 0.8,
                "news_warning_score": 0.1,
                "news_opportunity_score": 0.5,
                "financial_report_join_status": "event_window_matched",
                "financial_report_missing_rate": 0.1,
                "financial_report_event_count": 1,
                "financial_quality_risk_score": 0.1,
                "financial_surprise_score": 0.2,
                "lower_support": 0.25,
                "upper_overhang": 0.6,
                "cost_band_width": 0.8,
                "kline_return_20d": -22.0,
                "kline_return_60d": -18.0,
                "kline_atr20_pct": 4.0,
                "triggered_skills": "CORE_TREND_001",
            },
            {
                "date": "2025-01-03",
                "code": "000002",
                "news_count_30d": 0,
                "news_missing_rate": 1.0,
                "financial_report_join_status": "code_not_in_feature_table",
                "lower_support": 0.05,
                "upper_overhang": 2.0,
                "kline_return_20d": -24.0,
                "kline_return_60d": -40.0,
                "kline_atr20_pct": 13.0,
            },
        ]
    )

    selected = _apply_portfolio_row_gate(frame, "kline_reversal_friction_confirmed")

    assert selected["code"].tolist() == ["000001"]


def test_financial_event_quality_sampler_requires_matched_low_risk_event() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-03",
                "code": "000001",
                "news_count_30d": 1,
                "news_missing_rate": 0.2,
                "news_evidence_quality": 0.8,
                "news_warning_score": 0.1,
                "news_opportunity_score": 0.4,
                "financial_report_join_status": "event_window_matched",
                "financial_report_missing_rate": 0.1,
                "financial_report_event_count": 1,
                "financial_quality_risk_score": 0.2,
                "financial_surprise_score": 0.1,
                "tushare_industry_positive_breadth_20d": 0.6,
                "tushare_industry_relative_return_20d": 0.5,
                "lower_support": 0.2,
                "upper_overhang": 0.8,
                "cost_band_width": 0.9,
                "kline_return_20d": 1.0,
                "kline_return_60d": 2.0,
                "kline_atr20_pct": 3.0,
            },
            {
                "date": "2025-01-03",
                "code": "000002",
                "news_count_30d": 1,
                "news_missing_rate": 0.2,
                "news_evidence_quality": 0.8,
                "news_warning_score": 0.1,
                "news_opportunity_score": 0.4,
                "financial_report_join_status": "event_window_matched",
                "financial_report_missing_rate": 0.1,
                "financial_report_event_count": 1,
                "financial_quality_risk_score": 0.8,
                "financial_surprise_score": -0.2,
                "lower_support": 0.2,
                "upper_overhang": 0.8,
                "cost_band_width": 0.9,
                "kline_return_20d": 1.0,
                "kline_return_60d": 2.0,
                "kline_atr20_pct": 3.0,
            },
        ]
    )

    selected = _apply_portfolio_row_gate(frame, "financial_event_quality_pc2")

    assert selected["code"].tolist() == ["000001"]


def test_select_dual_mode_rows_diversifies_codes() -> None:
    rows = select_dual_mode_rows(_frame(), limit_per_mode=2, valid_block="H2023_2", portfolio_date_gate="all_dates", decision_frequency="twice_weekly")
    assert rows["portfolio_pool"]["code"].astype(str).nunique() == 2


def test_select_dual_mode_rows_penalizes_overheat_without_evidence() -> None:
    rows = select_dual_mode_rows(_frame(), limit_per_mode=1, valid_block="H2023_2")
    assert str(rows["portfolio_pool"].iloc[0]["date"]) != "2023-07-11"


def test_build_dual_mode_evidence_packs_excludes_future_returns() -> None:
    packs = build_dual_mode_evidence_packs(
        _frame(),
        limit_per_mode=1,
        agent_policy_version="test_dual_v0",
        step=1,
        train_blocks=["H2023_1"],
        valid_block="H2023_2",
        memory_context="memory",
    )
    assert {pack["task_mode"] for pack in packs} == {"portfolio_pool", "single_stock"}
    assert all("return_20d" not in pack for pack in packs)
    assert all("task_mode_requirement" in pack for pack in packs)


def test_dual_mode_metrics_by_task_mode() -> None:
    cards = [
        {
            "task_mode": "portfolio_pool",
            "decision_date": "2023-07-04",
            "code": "000001",
            "simulated_weight_change": 1.0,
            "data_missing_flags": "",
        },
        {
            "task_mode": "single_stock",
            "decision_date": "2023-07-07",
            "code": "000002",
            "simulated_weight_change": 0.0,
            "data_missing_flags": "financial_publish_date_missing",
        },
    ]
    metrics = dual_mode_metrics(cards, _frame())
    by_mode = {row["task_mode"]: row for _, row in metrics.iterrows()}
    assert by_mode["portfolio_pool"]["avg_return_20d_exposure"] == 5.0
    assert by_mode["portfolio_pool"]["schema_pass_rate"] == 1.0
    assert by_mode["portfolio_pool"]["cash_adjusted_avg_return_20d"] == 5.0
    assert pd.isna(by_mode["single_stock"]["avg_return_20d_exposure"])
    assert by_mode["single_stock"]["cash_adjusted_avg_return_20d"] > 0
    assert by_mode["single_stock"]["data_missing_flag_cards"] == 1


def test_dual_mode_raw_exposure_requires_increase_action() -> None:
    cards = [
        {
            "task_mode": "portfolio_pool",
            "decision_date": "2023-07-04",
            "code": "000001",
            "simulated_action": "降低研究暴露",
            "simulated_weight_change": 0.05,
            "data_missing_flags": "",
        },
        {
            "task_mode": "portfolio_pool",
            "decision_date": "2023-07-07",
            "code": "000002",
            "simulated_action": "增加研究暴露",
            "simulated_weight_change": 1.0,
            "data_missing_flags": "",
        },
    ]
    metrics = dual_mode_metrics(cards, _frame())
    row = metrics.iloc[0]
    assert row["exposure_cards"] == 1
    assert row["avg_return_20d_exposure"] == -3.0


def test_build_walkforward_evidence_packs_preserves_time_order() -> None:
    packs = build_walkforward_evidence_packs(
        _frame(),
        limit_per_mode=1,
        agent_policy_version="test_walk_v0",
        valid_blocks=["H2023_2", "H2024_1"],
    )
    assert {pack["valid_block"] for pack in packs} == {"H2023_2", "H2024_1"}
    by_block = {pack["valid_block"]: pack for pack in packs if pack["task_mode"] == "portfolio_pool"}
    assert by_block["H2023_2"]["train_blocks"] == "H2023_1"
    assert by_block["H2024_1"]["train_blocks"] == "H2023_1+H2023_2"
    assert by_block["H2024_1"]["step"] == 2


def test_dual_mode_step_metrics_splits_by_block_and_mode() -> None:
    cards = [
        {
            "agent_policy_version": "test_walk_v0",
            "step": 1,
            "train_blocks": "H2023_1",
            "valid_block": "H2023_2",
            "task_mode": "portfolio_pool",
            "decision_date": "2023-07-04",
            "code": "000001",
            "simulated_weight_change": 1.0,
            "data_missing_flags": "",
        },
        {
            "agent_policy_version": "test_walk_v0",
            "step": 2,
            "train_blocks": "H2023_1+H2023_2",
            "valid_block": "H2024_1",
            "task_mode": "single_stock",
            "decision_date": "2024-01-05",
            "code": "000003",
            "simulated_weight_change": 1.0,
            "data_missing_flags": "",
        },
    ]
    metrics = dual_mode_step_metrics(cards, _frame())
    keys = {(row["valid_block"], row["task_mode"]): row for _, row in metrics.iterrows()}
    assert keys[("H2023_2", "portfolio_pool")]["avg_return_20d_exposure"] == 5.0
    assert keys[("H2024_1", "single_stock")]["positive_20d_rate_exposure"] == 1.0
    assert keys[("H2024_1", "single_stock")]["cash_adjusted_avg_return_20d"] == 8.0

def test_walkforward_pack_records_portfolio_gate_and_frequency() -> None:
    packs = build_walkforward_evidence_packs(
        _frame(),
        limit_per_mode=1,
        agent_policy_version="test_gate_v0",
        valid_blocks=["H2023_2"],
        portfolio_preset="pullback_recovery",
        portfolio_date_gate="pool_pullback",
        portfolio_row_gate="news_risk_low",
        decision_frequency="every_2_weeks",
    )
    portfolio_packs = [pack for pack in packs if pack["task_mode"] == "portfolio_pool"]
    assert portfolio_packs
    assert "candidate=dual_mode_portfolio_pool:pullback_recovery:pool_pullback:news_risk_low:every_2_weeks" in portfolio_packs[0]["python_signal_summary"]


def test_peer_confirmed_pullback_and_news_risk_gate_do_not_refill_bad_rows() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2023-01-06", "code": "10", "gt_status": "evaluated", "relative_strength_rank": 0.5, "counter_score": 6, "prior_return_20d": -20, "rsi14": 40, "news_count_30d": 1},
            {"date": "2023-02-10", "code": "11", "gt_status": "evaluated", "relative_strength_rank": 0.5, "counter_score": 6, "prior_return_20d": 0, "rsi14": 45, "news_count_30d": 1},
            {"date": "2023-03-10", "code": "12", "gt_status": "evaluated", "relative_strength_rank": 0.5, "counter_score": 6, "prior_return_20d": 20, "rsi14": 50, "news_count_30d": 1},
            {"date": "2023-07-14", "code": "20", "gt_status": "evaluated", "relative_strength_rank": 0.9, "counter_score": 8, "prior_return_20d": -5, "rsi14": 45, "news_count_30d": 1, "peer_relative_to_group_20d": 1, "peer_group_positive_breadth_20d": 0.8, "news_risk_event_score_30d": 1, "news_warning_score_30d": 0},
            {"date": "2023-07-28", "code": "21", "gt_status": "evaluated", "relative_strength_rank": 0.8, "counter_score": 7, "prior_return_20d": 0, "rsi14": 50, "news_count_30d": 1, "peer_relative_to_group_20d": 1, "peer_group_positive_breadth_20d": 0.7, "news_risk_event_score_30d": 2, "news_warning_score_30d": 0},
        ]
    )
    rows = select_dual_mode_rows(
        frame,
        limit_per_mode=3,
        valid_block="H2023_2",
        train_blocks=["H2023_1"],
        portfolio_preset="peer_confirmed_pullback",
        portfolio_date_gate="all_dates",
        portfolio_row_gate="news_risk_low",
        decision_frequency="twice_weekly",
    )
    assert rows["portfolio_pool"].empty

def test_portfolio_date_gate_does_not_refill_filtered_dates() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2023-01-06", "code": "10", "gt_status": "evaluated", "relative_strength_rank": 0.5, "counter_score": 6, "prior_return_20d": -20, "rsi14": 40, "news_count_30d": 1},
            {"date": "2023-02-10", "code": "11", "gt_status": "evaluated", "relative_strength_rank": 0.5, "counter_score": 6, "prior_return_20d": 0, "rsi14": 45, "news_count_30d": 1},
            {"date": "2023-03-10", "code": "12", "gt_status": "evaluated", "relative_strength_rank": 0.5, "counter_score": 6, "prior_return_20d": 20, "rsi14": 50, "news_count_30d": 1},
            {"date": "2023-07-14", "code": "20", "gt_status": "evaluated", "relative_strength_rank": 0.9, "counter_score": 8, "prior_return_20d": -25, "rsi14": 35, "news_count_30d": 1},
            {"date": "2023-07-28", "code": "21", "gt_status": "evaluated", "relative_strength_rank": 1.0, "counter_score": 8, "prior_return_20d": 40, "rsi14": 70, "news_count_30d": 1},
            {"date": "2023-08-11", "code": "22", "gt_status": "evaluated", "relative_strength_rank": 0.95, "counter_score": 8, "prior_return_20d": 50, "rsi14": 75, "news_count_30d": 1},
        ]
    )
    rows = select_dual_mode_rows(
        frame,
        limit_per_mode=3,
        valid_block="H2023_2",
        train_blocks=["H2023_1"],
        portfolio_date_gate="pool_pullback",
        decision_frequency="every_2_weeks",
    )
    portfolio = rows["portfolio_pool"]
    assert len(portfolio) == 1
    assert set(portfolio["code"].astype(str)) == {"20"}


def test_cross_channel_row_gate_requires_multiple_confirmations() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2023-07-14",
                "code": "20",
                "gt_status": "evaluated",
                "return_20d": 3.0,
                "news_missing_rate": 0.1,
                "news_evidence_quality": 0.8,
                "news_count_30d": 2,
                "news_warning_score": 0.1,
                "news_opportunity_score": 0.4,
                "financial_report_missing_rate": 0.1,
                "financial_report_event_count": 1,
                "financial_quality_risk_score": 0.2,
                "tushare_industry_positive_breadth_20d": 0.7,
                "tushare_industry_relative_return_20d": 0.5,
                "lower_support": 0.20,
                "upper_overhang": 0.7,
                "triggered_skills": "PPS-Q-017",
            },
            {
                "date": "2023-07-14",
                "code": "21",
                "gt_status": "evaluated",
                "return_20d": -2.0,
                "news_missing_rate": 1.0,
                "news_evidence_quality": 0.0,
                "news_count_30d": 0,
                "financial_report_missing_rate": 1.0,
                "financial_report_event_count": 0,
                "financial_quality_risk_score": 0.9,
                "tushare_industry_positive_breadth_20d": 0.2,
                "tushare_industry_relative_return_20d": -3.0,
                "lower_support": 0.05,
                "upper_overhang": 2.0,
                "triggered_skills": "",
            },
        ]
    )
    rows = select_dual_mode_rows(
        frame,
        limit_per_mode=3,
        valid_block="H2023_2",
        portfolio_preset="rev_plus_chip_core",
        portfolio_date_gate="all_dates",
        portfolio_row_gate="cross_channel_min3",
        decision_frequency="twice_weekly",
    )
    assert set(rows["portfolio_pool"]["code"].astype(str)) == {"20"}


def test_reversal_ranker_v1_scores_increase_for_recent_losers() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2024-01-05", "code": "1", "gt_status": "evaluated", "prior_return_20d": 30, "peer_relative_to_group_20d": 5, "return_20d": -2},
            {"date": "2024-01-05", "code": "2", "gt_status": "evaluated", "prior_return_20d": -20, "peer_relative_to_group_20d": -4, "return_20d": 4},
            {"date": "2024-01-05", "code": "3", "gt_status": "evaluated", "prior_return_20d": 0, "peer_relative_to_group_20d": 0, "return_20d": 1},
            {"date": "2024-01-05", "code": "4", "gt_status": "evaluated", "prior_return_20d": 10, "peer_relative_to_group_20d": 2, "return_20d": 0},
            {"date": "2024-01-05", "code": "5", "gt_status": "evaluated", "prior_return_20d": -5, "peer_relative_to_group_20d": -1, "return_20d": 2},
        ]
    )
    scores = _portfolio_score(frame, "reversal_ranker_v1")
    assert float(scores.loc[frame["code"].astype(str).eq("2")].iloc[0]) > float(scores.loc[frame["code"].astype(str).eq("1")].iloc[0])


def test_reversal_ranker_v1_evidence_pack_carries_quant_tool_summaries() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2024-01-05", "code": "1", "name": "A", "gt_status": "evaluated", "prior_return_20d": 30, "peer_relative_to_group_20d": 5, "return_20d": -2},
            {"date": "2024-01-05", "code": "2", "name": "B", "gt_status": "evaluated", "prior_return_20d": -20, "peer_relative_to_group_20d": -4, "return_20d": 4},
            {"date": "2024-01-05", "code": "3", "name": "C", "gt_status": "evaluated", "prior_return_20d": 0, "peer_relative_to_group_20d": 0, "return_20d": 1},
            {"date": "2024-01-05", "code": "4", "name": "D", "gt_status": "evaluated", "prior_return_20d": 10, "peer_relative_to_group_20d": 2, "return_20d": 0},
            {"date": "2024-01-05", "code": "5", "name": "E", "gt_status": "evaluated", "prior_return_20d": -5, "peer_relative_to_group_20d": -1, "return_20d": 2},
        ]
    )
    packs = build_dual_mode_evidence_packs(
        frame,
        limit_per_mode=1,
        agent_policy_version="test_reversal_v1",
        step=1,
        train_blocks=["H2024_1"],
        valid_block="H2024_1",
        portfolio_preset="reversal_ranker_v1",
        portfolio_date_gate="all_dates",
        decision_frequency="twice_weekly",
    )
    portfolio = [pack for pack in packs if pack["task_mode"] == "portfolio_pool"]
    assert portfolio
    summaries = portfolio[0]["quant_tool_summaries"]
    assert summaries
    assert summaries[0]["tool_id"] == "portfolio_reversal_ranker"
    assert summaries[0]["usable_in_agent_default"] is False
    assert "return_20d" not in portfolio[0]
    assert "return_20d" not in summaries[0]


def test_rev_plus_chip_core_scores_recent_losers_with_chip_support() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2024-01-05", "code": "1", "gt_status": "evaluated", "kline_return_20d": 30, "kline_return_60d": 40, "corr_peer_avg_return_20d": 5, "lower_support": 0.05, "chip_concentration": 0.05, "cost_band_width": 0.10, "upper_overhang": 0.05, "winner_rate_pct": 80, "neg_winner_rate": -80, "return_20d": -2},
            {"date": "2024-01-05", "code": "2", "gt_status": "evaluated", "kline_return_20d": -20, "kline_return_60d": -30, "corr_peer_avg_return_20d": -4, "lower_support": 0.35, "chip_concentration": 0.30, "cost_band_width": 0.70, "upper_overhang": 0.40, "winner_rate_pct": 20, "neg_winner_rate": -20, "return_20d": 4},
            {"date": "2024-01-05", "code": "3", "gt_status": "evaluated", "kline_return_20d": 0, "kline_return_60d": 0, "corr_peer_avg_return_20d": 0, "lower_support": 0.15, "chip_concentration": 0.12, "cost_band_width": 0.25, "upper_overhang": 0.15, "winner_rate_pct": 50, "neg_winner_rate": -50, "return_20d": 1},
            {"date": "2024-01-05", "code": "4", "gt_status": "evaluated", "kline_return_20d": 10, "kline_return_60d": 12, "corr_peer_avg_return_20d": 2, "lower_support": 0.12, "chip_concentration": 0.10, "cost_band_width": 0.18, "upper_overhang": 0.12, "winner_rate_pct": 60, "neg_winner_rate": -60, "return_20d": 0},
            {"date": "2024-01-05", "code": "5", "gt_status": "evaluated", "kline_return_20d": -5, "kline_return_60d": -8, "corr_peer_avg_return_20d": -1, "lower_support": 0.20, "chip_concentration": 0.16, "cost_band_width": 0.35, "upper_overhang": 0.22, "winner_rate_pct": 40, "neg_winner_rate": -40, "return_20d": 2},
        ]
    )
    scores = _portfolio_score(frame, "rev_plus_chip_core")
    assert float(scores.loc[frame["code"].astype(str).eq("2")].iloc[0]) > float(scores.loc[frame["code"].astype(str).eq("1")].iloc[0])


def test_rev_plus_chip_core_evidence_pack_carries_default_quant_tool_and_chip_summary() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2024-01-05", "code": "1", "name": "A", "gt_status": "evaluated", "kline_return_20d": 30, "kline_return_60d": 40, "corr_peer_avg_return_20d": 5, "lower_support": 0.05, "chip_concentration": 0.05, "cost_band_width": 0.10, "upper_overhang": 0.05, "winner_rate_pct": 80, "neg_winner_rate": -80, "chip_core_source_name": "unit_test", "return_20d": -2},
            {"date": "2024-01-05", "code": "2", "name": "B", "gt_status": "evaluated", "kline_return_20d": -20, "kline_return_60d": -30, "corr_peer_avg_return_20d": -4, "lower_support": 0.35, "chip_concentration": 0.30, "cost_band_width": 0.70, "upper_overhang": 0.40, "winner_rate_pct": 20, "neg_winner_rate": -20, "chip_core_source_name": "unit_test", "return_20d": 4},
            {"date": "2024-01-05", "code": "3", "name": "C", "gt_status": "evaluated", "kline_return_20d": 0, "kline_return_60d": 0, "corr_peer_avg_return_20d": 0, "lower_support": 0.15, "chip_concentration": 0.12, "cost_band_width": 0.25, "upper_overhang": 0.15, "winner_rate_pct": 50, "neg_winner_rate": -50, "chip_core_source_name": "unit_test", "return_20d": 1},
            {"date": "2024-01-05", "code": "4", "name": "D", "gt_status": "evaluated", "kline_return_20d": 10, "kline_return_60d": 12, "corr_peer_avg_return_20d": 2, "lower_support": 0.12, "chip_concentration": 0.10, "cost_band_width": 0.18, "upper_overhang": 0.12, "winner_rate_pct": 60, "neg_winner_rate": -60, "chip_core_source_name": "unit_test", "return_20d": 0},
            {"date": "2024-01-05", "code": "5", "name": "E", "gt_status": "evaluated", "kline_return_20d": -5, "kline_return_60d": -8, "corr_peer_avg_return_20d": -1, "lower_support": 0.20, "chip_concentration": 0.16, "cost_band_width": 0.35, "upper_overhang": 0.22, "winner_rate_pct": 40, "neg_winner_rate": -40, "chip_core_source_name": "unit_test", "return_20d": 2},
        ]
    )
    packs = build_dual_mode_evidence_packs(
        frame,
        limit_per_mode=1,
        agent_policy_version="test_rev_chip_core_v1",
        step=1,
        train_blocks=["H2024_1"],
        valid_block="H2024_1",
        portfolio_preset="rev_plus_chip_core",
        portfolio_date_gate="all_dates",
        decision_frequency="twice_weekly",
    )
    portfolio = [pack for pack in packs if pack["task_mode"] == "portfolio_pool"]
    assert portfolio
    summaries = portfolio[0]["quant_tool_summaries"]
    assert summaries[0]["tool_id"] == "portfolio_rev_chip_core_ranker"
    assert summaries[0]["usable_in_agent_default"] is True
    assert portfolio[0]["chip_features"]["lower_support"] == 0.35
    assert "return_20d" not in portfolio[0]
    assert "return_20d" not in summaries[0]
