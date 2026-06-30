from __future__ import annotations

import json

import pandas as pd

from scripts.run_date_regime_gate_experiment import (
    all_dates_rule,
    apply_rule_mask,
    aggregate_selected,
    block_stability_metrics,
    build_daily_regime_table,
    build_global_regime_features,
    build_rule_candidates,
    choose_best_rule,
    diagnostics_table,
    evaluate_gate_on_daily_table,
    rule_training_score,
    write_rule_outcomes,
)


def _daily_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [f"2024-01-{day:02d}" for day in range(1, 7)],
            "pool_avg_prior20": [10, 8, 6, 4, 2, 0],
            "pool_overheat_ratio": [0.8, 0.7, 0.3, 0.2, 0.1, 0.0],
            "pool_news_risk_avg": [1, 1, 0, 0, 0, 0],
            "pool_fin_quality_risk_avg": [0, 0, 0, 0, 0, 0],
            "pool_atr20_avg": [3, 3, 2, 2, 1, 1],
            "pool_k60_deep_drawdown_ratio": [0, 0, 0, 0, 0, 0],
            "pool_peer_breadth": [0.1, 0.2, 0.6, 0.7, 0.8, 0.9],
            "pool_regime_score": [1, 1, 3, 3, 4, 4],
            "pool_news_coverage": [0.5] * 6,
            "pool_financial_coverage": [0.5] * 6,
            "pool_kline_safe_ratio": [0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
            "pool_above_ma200_ratio": [0.4] * 6,
            "return_20d": [-8.0, -3.0, 2.0, 4.0, 6.0, 8.0],
        }
    )


def _stock_frame() -> pd.DataFrame:
    rows = []
    for code, k20, above, prior, news_missing in [
        ("000001", 5.0, 1.0, 4.0, 0.0),
        ("000002", -12.0, 0.0, 70.0, 1.0),
    ]:
        rows.append(
            {
                "code": code,
                "date": "2024-01-05",
                "_k20": k20,
                "_prior20": prior,
                "_above_ma200": above,
                "_overheat": 1.0 if prior >= 60 else 0.0,
                "_news_ok": 1.0 if news_missing < 0.8 else 0.0,
                "_financial_ok": 1.0,
                "_peer_ok": 1.0,
                "_kline_safe": 1.0 if -15 <= k20 <= 25 else 0.0,
                "_confirmation_count": 3.0,
                "_news_risk": 0.0,
                "_fin_quality_risk": 0.0,
                "_atr20": 2.0,
                "_k60": -5.0,
                "_candidate_score": 1.0,
                "return_20d": 2.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_daily_regime_table_merges_global_market_features() -> None:
    frame = _stock_frame()
    global_features = build_global_regime_features(frame)

    table = build_daily_regime_table(frame, row_gate="none", frequency="twice_weekly", top_n=1, global_regime_features=global_features)
    rules = build_rule_candidates(table, quantiles=[0.5])

    assert table.iloc[0]["global_stock_count"] == 2
    assert table.iloc[0]["global_kline_positive_breadth_20d"] == 0.5
    assert any(rule["rule_id"].startswith("global_kline_positive_breadth_20d") for rule in rules)


def test_build_rule_candidates_uses_train_quantiles_and_masks() -> None:
    table = _daily_table()
    rules = build_rule_candidates(table, quantiles=[0.5])
    peer_rule = next(rule for rule in rules if rule["rule_id"] == "pool_peer_breadth_ge_q50")

    mask = apply_rule_mask(table, peer_rule)

    assert rules[0]["rule_id"] == "all_dates"
    assert mask.tolist() == [False, False, False, True, True, True]
    assert "train_q50" in peer_rule["rule_text"]


def test_evaluate_gate_counts_cash_for_skipped_dates() -> None:
    table = _daily_table()
    rule = {"rule_id": "peer_high", "rule_text": "peer high", "conditions": [{"feature": "pool_peer_breadth", "op": ">=", "threshold": 0.6}]}

    metrics = evaluate_gate_on_daily_table(table, rule)

    assert metrics["expected_dates"] == 6
    assert metrics["active_dates"] == 4
    assert metrics["raw_positive_20d_rate"] == 1.0
    assert metrics["decision_coverage"] == 0.6667
    assert metrics["cash_blended_positive_20d_rate"] == 1.0


def test_choose_best_rule_prefers_viable_train_score_and_falls_back() -> None:
    table = _daily_table()
    baseline = all_dates_rule()
    peer_rule = {"rule_id": "peer_high", "rule_text": "peer high", "conditions": [{"feature": "pool_peer_breadth", "op": ">=", "threshold": 0.6}]}
    train_evals = []
    for rule in [baseline, peer_rule]:
        metrics = evaluate_gate_on_daily_table(table, rule)
        train_evals.append((rule, metrics, rule_training_score(metrics)))

    best_rule, _, _ = choose_best_rule(train_evals, min_active_dates=3, min_coverage=0.3)
    fallback_rule, _, _ = choose_best_rule(train_evals, min_active_dates=99, min_coverage=0.3)

    assert best_rule["rule_id"] == "peer_high"
    assert fallback_rule["rule_id"] == "all_dates"


def test_block_stability_can_filter_unstable_train_rule() -> None:
    rule = all_dates_rule()
    unstable = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-07-01", "2023-07-02"],
            "time_block": ["H2023_1", "H2023_1", "H2023_2", "H2023_2"],
            "return_20d": [5.0, 6.0, -5.0, -6.0],
        }
    )
    metrics = evaluate_gate_on_daily_table(unstable, rule)
    metrics.update(block_stability_metrics(unstable, rule, hit_threshold=0.55))

    best_rule, _, _ = choose_best_rule([(rule, metrics, rule_training_score(metrics))], min_active_dates=1, min_coverage=0.1, min_block_hit_ratio=0.75)

    assert metrics["train_block_count"] == 2
    assert metrics["train_block_hit_count"] == 1
    assert metrics["train_block_hit_ratio"] == 0.5
    assert best_rule["rule_id"] == "all_dates"


def test_diagnostics_requires_lift_and_low_loss_for_candidate(tmp_path) -> None:
    rows = []
    for panel, block in [("p1", "H2023_2"), ("p1", "H2024_2"), ("p2", "H2025_1"), ("p2", "H2026_1")]:
        rows.append(
            {
                "panel": panel,
                "valid_block": block,
                "strategy_id": "s_good",
                "score_profile": "kline_multiscale_quality",
                "row_gate": "cross_channel_min2",
                "decision_frequency": "weekly_friday",
                "top_n": 10,
                "selected_rule_id": "pool_peer_breadth_ge_q50",
                "selected_valid_decision_coverage": 0.5,
                "selected_valid_avg_return_20d": 2.0,
                "selected_valid_raw_positive_20d_rate": 0.65,
                "selected_valid_std_return_20d": 1.0,
                "selected_valid_loss_20d_over_5_rate": 0.10,
                "selected_valid_cash_blended_avg_return_20d": 1.0,
                "selected_valid_cash_blended_positive_20d_rate": 0.8,
                "baseline_valid_avg_return_20d": 1.0,
                "baseline_valid_raw_positive_20d_rate": 0.55,
                "baseline_valid_cash_blended_avg_return_20d": 0.5,
                "delta_valid_raw_positive_20d_rate": 0.10,
                "delta_valid_avg_return_20d": 1.0,
                "delta_valid_cash_blended_avg_return_20d": 0.5,
                "delta_valid_loss_20d_over_5_rate": -0.05,
            }
        )
    selected = pd.DataFrame(rows)
    aggregate = aggregate_selected(selected)
    diagnostics = diagnostics_table(aggregate)
    outcome_path = tmp_path / "outcomes.jsonl"

    write_rule_outcomes(outcome_path, diagnostics)
    first = json.loads(outcome_path.read_text(encoding="utf-8").splitlines()[0])

    assert diagnostics.iloc[0]["promotion_status"] == "candidate_for_agent_regime_gate"
    assert first["usable_in_agent_default"] is True
    assert first["research_only"] is True
