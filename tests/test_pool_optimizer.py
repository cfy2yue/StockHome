from pathlib import Path

import pandas as pd

from src.backtest.pool_optimizer import write_pool_optimizer_report


def test_pool_optimizer_writes_search_valid_test_report(tmp_path: Path):
    for split in ["epoch2", "test"]:
        path = tmp_path / split
        path.mkdir()
        rows = []
        dates = pd.bdate_range("2025-01-01", periods=12)
        for date in dates:
            for idx in range(15):
                good = idx < 3
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "code": f"600{idx:03d}",
                        "gt_status": "evaluated",
                        "return_20d": 9 if good else -2,
                        "total_score": 8 if good else 3,
                        "relative_strength_rank": 0.9 if good else 0.2,
                        "close_above_ma200": good,
                        "counter_score": 8,
                        "news_risk_event_score_30d": 0,
                        "peer_relative_to_group_20d": 5 if good else -1,
                        "peer_group_positive_breadth_20d": 0.8 if good else 0.2,
                    }
                )
        pd.DataFrame(rows).to_csv(path / "ground_truth.csv", index=False)
    result = write_pool_optimizer_report(tmp_path)
    assert not result.empty
    assert (tmp_path / "pool_optimizer_report.csv").exists()
    assert "valid20日均值" in (tmp_path / "pool_optimizer_report.md").read_text(encoding="utf-8")
