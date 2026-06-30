from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_kline_channel_exploration import (
    add_peer_kline_features,
    build_daily_kline_features,
    merge_kline_features,
    run_exploration,
)


def _daily_rows(days: int = 150) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    close = pd.Series([10 + idx * 0.08 + ((-1) ** idx) * 0.25 for idx in range(days)])
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": [1000000 + idx * 1000 for idx in range(days)],
            "amount": [10000000 + idx * 10000 for idx in range(days)],
            "pct_chg": close.pct_change().fillna(0) * 100,
        }
    )


def test_multiscale_daily_features_are_time_safe(tmp_path: Path) -> None:
    for code, offset in [("000001", 0.0), ("000002", 1.0)]:
        code_dir = tmp_path / code
        code_dir.mkdir()
        daily = _daily_rows()
        daily["close"] = daily["close"] + offset
        daily.to_csv(code_dir / "daily.csv", index=False)

    features = build_daily_kline_features(tmp_path, allowed_codes={"000001", "000002"})
    assert {
        "kline_return_3d",
        "kline_return_5d",
        "kline_return_60d",
        "kline_efficiency_ratio_20d",
        "kline_direction_reversal_rate_20d",
        "kline_oscillation_cross_count_20d",
        "kline_range_width_pct_60d",
    }.issubset(features.columns)

    decisions = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "date": pd.to_datetime(["2023-06-30", "2023-06-30"]),
            "sector_group": ["all", "all"],
            "return_20d": [2.0, -1.0],
            "time_block": ["H2023_1", "H2023_1"],
        }
    )
    merged = merge_kline_features(decisions, features)
    assert pd.to_numeric(merged["kline_return_5d"], errors="coerce").notna().all()
    assert pd.to_numeric(merged["kline_return_60d"], errors="coerce").notna().all()
    assert pd.to_numeric(merged["kline_oscillation_cross_count_20d"], errors="coerce").notna().all()
    assert pd.to_numeric(merged["kline_efficiency_ratio_20d"], errors="coerce").notna().all()
    assert pd.to_numeric(merged["kline_direction_reversal_rate_20d"], errors="coerce").notna().all()
    assert "return_20d_x" not in merged.columns


def test_peer_kline_features_and_exploration_keep_validation_split() -> None:
    frame = pd.DataFrame(
        {
            "code": ["000001", "000002", "000003", "000004", "000005", "000006"],
            "date": pd.to_datetime(["2023-05-05", "2023-05-05", "2024-10-11", "2024-10-11", "2025-10-10", "2025-10-10"]),
            "sector_group": ["all", "all", "all", "all", "all", "all"],
            "time_block": ["H2023_1", "H2023_1", "H2024_2", "H2024_2", "H2025_2", "H2025_2"],
            "return_20d": [3, -2, 4, -1, 5, -3],
            "kline_return_5d": [2, -1, 3, -2, 4, -3],
            "kline_return_20d": [5, -2, 6, -1, 8, -4],
            "kline_return_60d": [8, -3, 9, -2, 10, -5],
            "kline_return_120d": [10, -4, 11, -3, 12, -6],
            "kline_volatility_20d": [2, 3, 2, 3, 2, 3],
            "kline_ma_gap_close_200": [1, -1, 1, -1, 1, -1],
        }
    )

    enriched = add_peer_kline_features(frame)
    assert "peer_kline_relative_to_group_20d" in enriched
    assert "peer_kline_group_positive_breadth_20d" in enriched
    result = run_exploration(enriched, max_depth=1, min_samples=1)
    assert set(result["rule_id"]).issuperset({"baseline_all", "multiscale_price_plus_peer_depth1"})
    assert result.loc[result["rule_id"] == "baseline_all", "train_sample_count"].iloc[0] == 2
