from pathlib import Path

import pandas as pd

from src.backtest.rebound_diagnostics import write_rebound_diagnostics_report


def test_rebound_diagnostics_writes_fold_report(tmp_path: Path):
    path = tmp_path / "epoch2"
    path.mkdir()
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=120)
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

    result = write_rebound_diagnostics_report(tmp_path, folds=3)

    assert not result.empty
    assert (tmp_path / "rebound_diagnostics.csv").exists()
    text = (tmp_path / "rebound_diagnostics.md").read_text(encoding="utf-8")
    assert "反弹型候选池诊断" in text
    assert "策略族汇总" in text
