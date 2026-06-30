from pathlib import Path

import pandas as pd

from src.backtest.pool_selection import write_pool_selection_report


def test_pool_selection_report_compares_topn_with_pool_baseline(tmp_path: Path):
    for split in ["epoch2", "test"]:
        path = tmp_path / split
        path.mkdir()
        rows = []
        for day in range(3):
            for idx in range(25):
                rows.append(
                    {
                        "date": f"2025-01-0{day + 1}",
                        "code": f"600{idx:03d}",
                        "gt_status": "evaluated",
                        "return_20d": 10 if idx < 5 else -2,
                        "total_score": 9 if idx < 5 else 3,
                        "relative_strength_rank": 0.9 if idx < 5 else 0.2,
                        "close_above_ma200": idx < 5,
                        "peer_relative_to_group_20d": 5 if idx < 5 else -1,
                        "news_risk_event_score_30d": 0,
                    }
                )
        pd.DataFrame(rows).to_csv(path / "ground_truth.csv", index=False)
    result = write_pool_selection_report(tmp_path)
    assert not result.empty
    assert (tmp_path / "pool_selection_report.md").exists()
    test_top5 = result[(result["split"] == "test") & (result["strategy"] == "总分Top5")].iloc[0]
    assert test_top5["avg_return_20d"] == 10
