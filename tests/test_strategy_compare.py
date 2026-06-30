from __future__ import annotations

import pandas as pd

from src.backtest.strategy_compare import comparison_summary, write_strategy_comparison


def test_strategy_comparison_quantifies_delta(tmp_path):
    rows = [
        {
            "rating": "放入观察",
            "triggered_skills": "PPS-Q-017;PPS-Q-019;DOW-B-017",
            "prior_return_20d": 10,
            "relative_strength_rank": 0.8,
            "close_above_ma200": True,
            "return_5d": 1,
            "return_10d": 2,
            "return_20d": 8,
        },
        {
            "rating": "放入观察",
            "triggered_skills": "PPS-Q-017",
            "prior_return_20d": -5,
            "relative_strength_rank": 0.2,
            "close_above_ma200": False,
            "return_5d": -2,
            "return_10d": -3,
            "return_20d": -6,
        },
    ]
    for split in ["epoch1", "epoch2", "test"]:
        path = tmp_path / split
        path.mkdir()
        pd.DataFrame(rows).to_csv(path / "ground_truth.csv", index=False)
    result = write_strategy_comparison(tmp_path)
    summary = comparison_summary(result, "test")
    assert (tmp_path / "strategy_comparison.md").exists()
    assert summary["avg_return_20d_delta"] > 0
    assert summary["loss_20d_over_5_rate_delta"] < 0
