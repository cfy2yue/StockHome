from __future__ import annotations

import pandas as pd

from scripts.run_date_regime_gate_experiment import (
    _apply_exposure_to_portfolio_metrics,
    _summarize_exposure_scope,
    run_exposure_guard_experiment,
)
from src.agent_training.date_regime_gate import (
    EXPOSURE_GUARD_PRESETS,
    apply_exposure_gate_to_table,
    auditor_checks_exposure_gate,
    build_daily_regime_features,
    exposure_scale_from_score,
    fit_exposure_gate_spec,
)
from src.agent_training.dual_mode_round import select_dual_mode_rows


def _stock_frame() -> pd.DataFrame:
    rows = []
    for i, (date, k20, above) in enumerate(
        [
            ("2024-01-02", -5.0, 0.0),
            ("2024-01-02", 8.0, 1.0),
            ("2024-01-03", -12.0, 0.0),
            ("2024-01-03", 2.0, 1.0),
            ("2024-07-02", -3.0, 1.0),
            ("2024-07-02", 6.0, 1.0),
        ]
    ):
        rows.append(
            {
                "date": date,
                "code": f"{i:06d}",
                "gt_status": "evaluated",
                "prior_return_20d": k20,
                "kline_return_20d": k20,
                "kline_return_60d": k20 / 2,
                "close_above_ma200": above,
                "rsi14": 50,
                "return_20d": k20 / 3,
                "relative_strength_rank": 0.5,
                "counter_score": 5,
            }
        )
    return pd.DataFrame(rows)


def test_build_daily_regime_features_decision_time_only() -> None:
    table = build_daily_regime_features(_stock_frame(), include_reversal_ic_proxy=True)
    assert "global_above_ma200_rate" in table.columns
    assert "cross_section_std_prior20" in table.columns
    assert table["global_above_ma200_rate"].between(0, 1).all()


def test_fit_exposure_gate_spec_respects_train_blocks() -> None:
    table = build_daily_regime_features(_stock_frame(), include_reversal_ic_proxy=True)
    table["time_block"] = "H2024_1"
    spec = fit_exposure_gate_spec(table, preset="moderate", train_blocks=["H2024_1"])
    gated = apply_exposure_gate_to_table(table, spec)
    assert set(gated["exposure_label"].unique()) <= {"deploy", "half", "abstain"}
    checks = auditor_checks_exposure_gate(train_table=table, feature_cols=spec.used_features)
    assert checks["h2026_not_in_train"] is True
    assert checks["no_future_feature_cols"] is True


def test_exposure_scale_mapping() -> None:
    spec = fit_exposure_gate_spec(
        pd.DataFrame(
            {
                "time_block": ["H2024_1"] * 4,
                "global_above_ma200_rate": [0.1, 0.3, 0.6, 0.9],
                "global_kline_positive_breadth_20d": [0.2, 0.4, 0.5, 0.8],
                "global_weak_breadth_ratio": [0.8, 0.6, 0.4, 0.1],
                "global_overheat_ratio": [0.7, 0.5, 0.3, 0.1],
            }
        ),
        preset="moderate",
        train_blocks=["H2024_1"],
    )
    low = exposure_scale_from_score(-2.0, spec)
    high = exposure_scale_from_score(2.0, spec)
    assert low <= high


def test_summarize_exposure_scope_scales_metrics() -> None:
    port = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "topk_pool_excess_gross": [2.0, -1.0],
            "topk_pool_excess_net": [1.5, -1.5],
            "topk_pool_excess_net_flat": [0.5, -2.5],
            "active_selected": [5.0, 5.0],
            "turnover": [0.1, 0.2],
        }
    )
    exposure = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "exposure_scale": [1.0, 0.0],
            "exposure_label": ["deploy", "abstain"],
        }
    )
    merged = _apply_exposure_to_portfolio_metrics(port, exposure)
    summary = _summarize_exposure_scope(merged)
    assert summary["n_abstain_days"] == 1
    assert summary["decision_coverage"] == 0.5


def test_select_dual_mode_rows_exposure_guard_default_off() -> None:
    frame = _stock_frame()
    base = select_dual_mode_rows(frame, limit_per_mode=1, valid_block="H2024_1", portfolio_date_gate="all_dates")
    assert "exposure_scale" not in base["portfolio_pool"].columns


def test_select_dual_mode_rows_exposure_guard_enabled() -> None:
    frame = _stock_frame()
    rows = select_dual_mode_rows(
        frame,
        limit_per_mode=2,
        valid_block="H2024_1",
        train_blocks=["H2023_1", "H2023_2"],
        portfolio_date_gate="all_dates",
        portfolio_exposure_regime_gate="exposure_guard_v1",
        decision_frequency="twice_weekly",
    )
    pool = rows["portfolio_pool"]
    if not pool.empty:
        assert "exposure_scale" in pool.columns
        assert pool["exposure_scale"].between(0, 1).all()


def test_exposure_guard_presets_defined() -> None:
    assert set(EXPOSURE_GUARD_PRESETS) >= {"conservative", "moderate", "balanced", "aggressive"}
