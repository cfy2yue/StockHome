from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.gate_optimizer import write_gate_optimization_report


def test_gate_optimizer_writes_baseline_and_gate_report(tmp_path: Path):
    rows = []
    for idx in range(80):
        rows.append(
            {
                "date": f"2025-01-{(idx % 28) + 1:02d}",
                "rating": "放入观察" if idx % 2 == 0 else "暂时剔除",
                "sector_group": "star_technology",
                "triggered_skills": "PPS-Q-017;PPS-Q-019;DOW-B-017" if idx % 3 == 0 else "DOW-B-004",
                "prior_return_20d": 12 if idx % 3 == 0 else -2,
                "relative_strength_rank": 0.8 if idx % 3 == 0 else 0.2,
                "close_above_ma200": idx % 3 == 0,
                "total_score": 5.5,
                "trend_score": 8 if idx % 3 == 0 else 3,
                "book_score": 6,
                "counter_score": 8,
                "return_20d": 12 if idx % 3 == 0 else -4,
            }
        )
    for split in ["epoch2", "test"]:
        path = tmp_path / split
        path.mkdir()
        pd.DataFrame(rows).to_csv(path / "ground_truth.csv", index=False)
    result = write_gate_optimization_report(tmp_path)
    assert not result.empty
    assert (tmp_path / "baseline_comparison.csv").exists()
    assert (tmp_path / "gate_optimization.md").exists()
    assert "强动量低反证" in (tmp_path / "gate_optimization.md").read_text(encoding="utf-8")
