from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_lightweight_ml_channel_experiment import (
    FUTURE_OR_LABEL_FIELDS,
    add_correlation_peer_features,
    add_regime_features,
    add_tushare_peer_features,
    build_date_gate_specs,
    DateGateSpec,
    fit_additive_bin_model,
    merge_correlation_peer_features,
    merge_tushare_peer_features,
    apply_date_gate,
    run_experiment,
    score_frame,
)


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    blocks = [
        ("H2023_1", "2023-04-04"),
        ("H2023_2", "2023-08-04"),
        ("H2024_1", "2024-04-04"),
        ("H2024_2", "2024-08-04"),
        ("H2025_1", "2025-04-04"),
        ("H2025_2", "2025-08-04"),
        ("H2026_1", "2026-04-04"),
    ]
    for block, date in blocks:
        for idx in range(260):
            strong = idx % 4 == 0
            rows.append(
                {
                    "date": date,
                    "code": f"{idx + 1:06d}",
                    "time_block": block,
                    "total_score": 8 if strong else 4,
                    "trend_score": 7 if strong else 3,
                    "prior_return_20d": -12 if strong else 8,
                    "kline_return_20d": -12 if strong else 8,
                    "kline_rsi14": 38 if strong else 72,
                    "kline_atr20_pct": 2 if strong else 7,
                    "news_warning_score": 0 if strong else 1,
                    "financial_report_missing_rate": 0 if strong else 1,
                    "return_20d": 6 if strong else -3,
                    "gt_status": "evaluated",
                    "rating": "暂时剔除",
                }
            )
    return pd.DataFrame(rows)


def test_additive_bin_model_scores_without_future_fields() -> None:
    frame = _synthetic_frame()
    features = ["total_score", "kline_return_20d", "news_warning_score", "return_20d", "rating"]
    model = fit_additive_bin_model(frame, [feature for feature in features if feature not in FUTURE_OR_LABEL_FIELDS], feature_group="test")
    assert model.rules
    assert "return_20d" not in model.selected_features
    assert "rating" not in model.selected_features

    scored = score_frame(frame, model)
    assert "ml_score" in scored.columns
    strong_score = scored.loc[scored["total_score"] == 8, "ml_score"].mean()
    weak_score = scored.loc[scored["total_score"] == 4, "ml_score"].mean()
    assert strong_score > weak_score


def test_additive_bin_model_handles_nullable_numeric_features() -> None:
    frame = _synthetic_frame().head(240).copy()
    frame["nullable_peer_feature"] = pd.Series([pd.NA if idx % 5 == 0 else float(idx % 7) for idx in range(len(frame))], dtype="Float64")
    model = fit_additive_bin_model(frame, ["nullable_peer_feature"], feature_group="nullable_test")
    scored = score_frame(frame, model)
    assert "ml_score" in scored
    assert pd.to_numeric(scored["ml_score"], errors="coerce").notna().all()


def test_run_experiment_writes_all_rolling_blocks() -> None:
    outputs = run_experiment(_synthetic_frame())
    step = outputs["step_metrics"]
    aggregate = outputs["aggregate"]
    assert not step.empty
    assert set(step["target_block"]) == {"H2024_2", "H2025_1", "H2025_2", "H2026_1"}
    assert not aggregate.empty
    assert "promotion_status" in aggregate.columns


def test_correlation_peer_features_are_time_safe(tmp_path: Path) -> None:
    daily_root = tmp_path / "daily"
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    for code, slope in [("000001", 1.0), ("000002", 1.05), ("000003", -1.0)]:
        path = daily_root / code
        path.mkdir(parents=True)
        close = [10 + slope * idx * 0.01 for idx in range(len(dates))]
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close}).to_csv(path / "daily.csv", index=False)
    frame = pd.DataFrame(
        {
            "date": ["2023-06-30"] * 3,
            "code": ["000001", "000002", "000003"],
            "kline_return_20d": [2.0, 3.0, -2.0],
        }
    )
    enriched = add_correlation_peer_features(frame, daily_root)
    assert "corr_peer_avg_return_20d" in enriched.columns
    row = enriched[enriched["code"] == "000001"].iloc[0]
    assert row["corr_peer_count"] >= 1
    assert row["corr_peer_avg_return_20d"] > 0


def test_corr_peer_cache_merge_and_regime_gate() -> None:
    frame = _synthetic_frame().head(20).copy()
    features = pd.DataFrame(
        {
            "date": [frame.iloc[0]["date"]],
            "code": [frame.iloc[0]["code"]],
            "corr_peer_avg_return_20d": [3.0],
            "corr_peer_positive_breadth_20d": [0.8],
            "corr_peer_relative_return_20d": [1.0],
            "corr_peer_avg_corr": [0.6],
            "corr_peer_count": [10],
        }
    )
    merged = merge_correlation_peer_features(frame, features)
    assert pd.to_numeric(merged["corr_peer_avg_return_20d"], errors="coerce").notna().sum() == 1

    with_regime = add_regime_features(frame)
    assert "regime_prior_positive_breadth_20d" in with_regime.columns
    specs = build_date_gate_specs(with_regime)
    assert specs[0].name == "all_dates"
    gate = DateGateSpec("positive_breadth_ge_zero", "regime_prior_positive_breadth_20d", ">=", 0.0)
    filtered = apply_date_gate(with_regime, gate)
    assert len(filtered) == len(with_regime)


def test_tushare_peer_features_exclude_self_and_cache_merge(tmp_path: Path) -> None:
    stock_basic = tmp_path / "stock_basic.csv"
    pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003"],
            "industry": ["银行", "银行", "地产"],
            "area": ["深圳", "深圳", "北京"],
        }
    ).to_csv(stock_basic, index=False)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-05"]),
            "code": ["000001", "000002", "000003"],
            "kline_return_20d": [2.0, -4.0, 6.0],
            "kline_ma_gap_close_200": [1.0, -2.0, 3.0],
            "news_warning_score": [0.1, 0.5, 0.2],
            "news_opportunity_score": [0.2, 0.8, 0.4],
            "self_news_intensity": [0.3, 0.9, 0.1],
        }
    )
    enriched = add_tushare_peer_features(frame, stock_basic_path=stock_basic)
    row = enriched[enriched["code"] == "000001"].iloc[0]
    assert row["tushare_industry_group_size"] == 1
    assert row["tushare_industry_avg_return_20d"] == -4.0
    assert row["tushare_industry_relative_return_20d"] == 6.0
    assert row["tushare_industry_positive_breadth_20d"] == 0.0
    assert round(float(row["tushare_industry_news_attention_gap"]), 4) == -0.6

    features = enriched[["date", "code", "tushare_industry", "tushare_area", "tushare_industry_avg_return_20d"]]
    merged = merge_tushare_peer_features(frame[["date", "code"]], features)
    assert pd.to_numeric(merged["tushare_industry_avg_return_20d"], errors="coerce").notna().sum() == 2
