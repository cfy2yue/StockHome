from pathlib import Path

import pandas as pd

from src.backtest.pool_walkforward import write_pool_walkforward_report


def test_pool_walkforward_writes_oos_report(tmp_path: Path):
    path = tmp_path / "epoch2"
    path.mkdir()
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=120)
    for date in dates:
        for idx in range(12):
            strong = idx < 4
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "code": f"600{idx:03d}",
                    "gt_status": "evaluated",
                    "return_20d": 6 if strong else -1,
                    "total_score": 8 if strong else 3,
                    "relative_strength_rank": 0.9 if strong else 0.2,
                    "close_above_ma200": strong,
                    "counter_score": 8,
                    "news_risk_event_score_30d": 0,
                    "news_negative_materiality_30d": 0,
                    "peer_relative_to_group_20d": 4 if strong else -1,
                    "peer_group_positive_breadth_20d": 0.7 if strong else 0.2,
                    "peer_group_above_ma200_rate": 0.7 if strong else 0.2,
                    "atr20_pct": 3 if strong else 8,
                    "drawdown60": -5 if strong else -25,
                    "prior_return_20d": 5 if strong else -8,
                }
            )
    pd.DataFrame(rows).to_csv(path / "ground_truth.csv", index=False)
    result = write_pool_walkforward_report(tmp_path, folds=3)
    assert not result.empty
    assert (tmp_path / "pool_walkforward_report.csv").exists()
    text = (tmp_path / "pool_walkforward_report.md").read_text(encoding="utf-8")
    assert "OOS" in text
