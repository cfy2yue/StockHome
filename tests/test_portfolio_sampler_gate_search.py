from __future__ import annotations

import pandas as pd

from scripts.run_portfolio_sampler_gate_search import (
    BANK_RETURN_20D,
    apply_date_gate,
    apply_row_gate,
    build_gate_thresholds,
    diagnostics_table,
    metrics_for_selected,
    prepare_features,
)


def _row(**overrides) -> dict:
    row = {
        "code": "1",
        "date": "2024-01-05",
        "gt_status": "evaluated",
        "relative_strength_rank": 0.7,
        "counter_score": 2.0,
        "close_above_ma200": True,
        "prior_return_20d": 5.0,
        "rsi14": 55.0,
        "atr20_pct": 2.0,
        "kline_return_20d": 5.0,
        "kline_return_60d": 3.0,
        "kline_return_120d": 2.0,
        "kline_volatility_ratio_20_60": 1.0,
        "news_missing_rate": 0.2,
        "news_count_30d": 2,
        "financial_report_missing_rate": 0.2,
        "financial_report_event_count": 1,
        "peer_relative_to_group_20d": 0.0,
        "peer_group_positive_breadth_20d": 0.6,
        "triggered_skills": "PPS-Q-017",
        "data_gaps": "",
    }
    row.update(overrides)
    return row


def test_prepare_features_builds_confirmation_and_risk_flags() -> None:
    frame = pd.DataFrame(
        [
            _row(code="000001"),
            _row(
                code="000002",
                news_missing_rate=1.0,
                news_count_30d=0,
                financial_report_missing_rate=1.0,
                financial_report_event_count=0,
                peer_group_positive_breadth_20d=0.2,
                kline_return_20d=40.0,
                kline_return_60d=-30.0,
                kline_volatility_ratio_20_60=2.0,
                prior_return_20d=65.0,
                triggered_skills="",
                data_gaps="financial_publish_date_missing",
            ),
        ]
    )

    prepared = prepare_features(frame).sort_values("code")

    strong = prepared.iloc[0]
    weak = prepared.iloc[1]
    assert strong["_news_ok"] == 1
    assert strong["_financial_ok"] == 1
    assert strong["_peer_ok"] == 1
    assert strong["_kline_safe"] == 1
    assert strong["_triggered_skill_present"] == 1
    assert strong["_confirmation_count"] == 5
    assert strong["_risk_gap_count"] == 0
    assert weak["_overheat"] == 1
    assert weak["_financial_gap_flag"] == 1
    assert weak["_confirmation_count"] == 0
    assert weak["_risk_gap_count"] == 5


def test_cross_channel_min3_requires_three_confirming_channels() -> None:
    frame = prepare_features(
        pd.DataFrame(
            [
                _row(code="000001"),
                _row(
                    code="000002",
                    news_missing_rate=1.0,
                    news_count_30d=0,
                    financial_report_missing_rate=1.0,
                    financial_report_event_count=0,
                    peer_group_positive_breadth_20d=0.2,
                    triggered_skills="",
                ),
            ]
        )
    )

    filtered = apply_row_gate(frame, "cross_channel_min3")

    assert filtered["code"].tolist() == ["000001"]


def test_date_gate_uses_train_thresholds_for_pool_pullback() -> None:
    train = prepare_features(
        pd.DataFrame(
            [
                _row(date="2023-01-03", prior_return_20d=-10.0),
                _row(date="2023-01-10", prior_return_20d=0.0),
                _row(date="2023-01-17", prior_return_20d=10.0),
            ]
        )
    )
    thresholds = build_gate_thresholds(train)
    valid = prepare_features(
        pd.DataFrame(
            [
                _row(code="000001", date="2024-01-02", prior_return_20d=-15.0),
                _row(code="000002", date="2024-01-09", prior_return_20d=20.0),
            ]
        )
    )

    filtered = apply_date_gate(valid, thresholds, "pool_pullback_q40")

    assert filtered["date"].tolist() == ["2024-01-02"]


def test_metrics_for_empty_selection_counts_cash_defense() -> None:
    selected = pd.DataFrame(columns=["code", "date", "return_20d"])

    metrics = metrics_for_selected(selected, expected_decision_dates=3)

    assert metrics["decision_dates"] == 0
    assert metrics["expected_decision_dates"] == 3
    assert metrics["decision_coverage"] == 0.0
    assert metrics["cash_blended_avg_return_20d"] == round(BANK_RETURN_20D, 4)
    assert metrics["cash_blended_positive_20d_rate"] == 1.0


def test_diagnostics_requires_stability_loss_and_hit_block_gates() -> None:
    base = {
        "score_profile": "kline_multiscale_quality",
        "date_gate": "low_overheat_q50",
        "row_gate": "cross_channel_min3",
        "decision_frequency": "weekly_friday",
        "top_n": 10,
        "panel_blocks": 12,
        "hit_blocks_pos60": 9,
        "decision_coverage_mean": 0.5,
        "raw_positive_20d_rate_mean": 0.62,
        "raw_positive_20d_rate_std": 0.18,
        "avg_return_20d_mean": 2.0,
        "avg_return_20d_std": 1.0,
        "loss_20d_over_5_rate_mean": 0.12,
        "stability_score_mean": 0.5,
        "h2026_pos_rate_mean": 0.60,
        "h2026_avg_return_mean": 1.0,
        "top_stock_share_mean": 0.05,
        "unique_codes_mean": 40,
    }
    aggregate = pd.DataFrame(
        [
            base,
            {**base, "score_profile": "unstable", "raw_positive_20d_rate_std": 0.21},
            {**base, "score_profile": "high_loss", "loss_20d_over_5_rate_mean": 0.21},
            {**base, "score_profile": "thin_hits", "hit_blocks_pos60": 8},
        ]
    )

    diagnostics = diagnostics_table(aggregate, pd.DataFrame())
    statuses = dict(zip(diagnostics["score_profile"], diagnostics["promotion_status"]))

    assert statuses["kline_multiscale_quality"] == "candidate_for_agent_sampler"
    assert statuses["unstable"] == "observe_unstable"
    assert statuses["high_loss"] == "observe_loss_too_high"
    assert statuses["thin_hits"] == "observe_not_enough_hit_blocks"
