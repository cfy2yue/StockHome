from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.tree_gate import write_tree_gate_report


def test_tree_gate_writes_explainable_report(tmp_path: Path):
    rows = []
    for idx in range(120):
        strong = idx % 4 == 0
        rows.append(
            {
                "date": f"2025-01-{(idx % 28) + 1:02d}",
                "total_score": 6 if strong else 4,
                "trend_score": 8 if strong else 3,
                "book_score": 7 if strong else 4,
                "counter_score": 8,
                "prior_return_20d": 15 if strong else -3,
                "relative_strength_rank": 0.85 if strong else 0.25,
                "rsi14": 65 if strong else 45,
                "macd_hist": 0.2 if strong else -0.1,
                "volume_ratio20": 1.2,
                "drawdown60": -5 if strong else -20,
                "ma200_slope20": 0.5 if strong else -0.3,
                "atr20_pct": 4,
                "return_20d": 12 if strong else -5,
            }
        )
    for split in ["epoch2", "test"]:
        path = tmp_path / split
        path.mkdir()
        pd.DataFrame(rows).to_csv(path / "ground_truth.csv", index=False)
    result = write_tree_gate_report(tmp_path)
    assert not result.empty
    assert (tmp_path / "tree_gate_optimization.csv").exists()
    assert "tree_gate_depth" in (tmp_path / "tree_gate_optimization.md").read_text(encoding="utf-8")
