from pathlib import Path

import pandas as pd

from src.backtest.rebound_validation import write_rebound_validation_report


def test_rebound_validation_locks_train_rule_and_tests_once(tmp_path: Path):
    for split in ["epoch2", "test"]:
        path = tmp_path / split
        path.mkdir()
        rows = []
        dates = pd.bdate_range("2024-01-01", periods=140 if split == "epoch2" else 40)
        for date in dates:
            for idx in range(12):
                rebound = idx < 3
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "code": f"600{idx:03d}",
                        "gt_status": "evaluated",
                        "return_20d": 10 if rebound else -2,
                        "total_score": 5,
                        "counter_score": 8 if rebound else 3,
                        "news_risk_event_score_30d": 0 if rebound else 5,
                        "news_negative_materiality_30d": 0,
                        "news_opportunity_event_score_30d": 0,
                        "news_conflict_intensity_30d": 0,
                        "relative_strength_rank": 0.4,
                        "close_above_ma200": not rebound,
                        "ma200_slope20": 1 if rebound else -1,
                        "atr20_pct": 3 if rebound else 9,
                        "drawdown60": -30 if rebound else -5,
                        "prior_return_20d": -18 if rebound else 4,
                        "peer_relative_to_group_20d": 0,
                        "peer_group_positive_breadth_20d": 0.75 if rebound else 0.2,
                        "peer_group_above_ma200_rate": 0.7 if rebound else 0.2,
                    }
                )
        pd.DataFrame(rows).to_csv(path / "ground_truth.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "test",
                "strategy": "全候选池等权基线",
                "avg_return_20d": 1,
                "positive_20d_rate": 0.5,
                "stability_score": -3,
            }
        ]
    ).to_csv(tmp_path / "pool_selection_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "test",
                "gate_name": "20日滚动长期持有基线",
                "avg_return_20d": 1,
                "positive_20d_rate": 0.5,
                "stability_score": -5,
            }
        ]
    ).to_csv(tmp_path / "baseline_comparison.csv", index=False)

    result = write_rebound_validation_report(tmp_path, folds=4)

    assert len(result) >= 1
    assert (tmp_path / "rebound_validation.csv").exists()
    assert (tmp_path / "rebound_validation.md").exists()
    row = result.sort_values("test_avg_return_20d", ascending=False).iloc[0]
    assert row["test_avg_return_20d"] >= 8
    assert row["test_vs_pool_avg_return_delta"] > 0
