from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.engine import run_backtest
from src.backtest.reusable_rules import build_reusable_rule_candidates
from src.backtest.reporting import write_summary_report
from src.backtest.weight_optimizer import optimize_weights


def test_light_backtest_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    stocks = [
        {"code": "600001", "name": "训练有色", "set": "train", "sector_group": "nonferrous_materials"},
        {"code": "688001", "name": "训练科创", "set": "train", "sector_group": "star_technology", "board": "star"},
    ]
    test_stocks = [{"code": "688002", "name": "测试科创", "set": "test", "sector_group": "star_technology", "board": "star"}]
    for stock in stocks + test_stocks:
        _write_stock_cache(data_dir, stock["code"])

    epoch1 = run_backtest(stocks, data_dir, {}, output_dir, "epoch1")
    weights = optimize_weights(epoch1["ground_truth"], {})
    epoch2 = run_backtest(stocks, data_dir, weights, output_dir, "epoch2")
    test = run_backtest(test_stocks, data_dir, weights, output_dir, "test")
    rules = build_reusable_rule_candidates(epoch1["ground_truth"], epoch2["ground_truth"], test["ground_truth"], output_dir / "reusable_rules")
    write_summary_report(output_dir, epoch1["ground_truth"], epoch2["ground_truth"], test["ground_truth"])

    assert not epoch1["decisions"].empty
    assert not epoch1["ground_truth"].empty
    assert set(epoch1["decisions"]["rating"]).issubset({"继续深挖", "放入观察", "暂时剔除", "信息不足"})
    assert (output_dir / "epoch1" / "decisions_summary.csv").exists()
    assert (output_dir / "test" / "ground_truth.csv").exists()
    assert "peer_relative_to_group_20d" in epoch1["decisions"].columns
    assert "peer_group_positive_breadth_20d" in epoch1["ground_truth"].columns
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "final_reusable_rules.yaml").exists()
    assert isinstance(rules, list)
    text = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert "不输出买卖指令" in text


def test_reusable_rule_schema_contains_audit_fields(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"triggered_skills": "PPS-Q-017", "gt_pass": True, "rating": "继续深挖", "return_5d": 4, "return_10d": 5, "return_20d": 9},
            {"triggered_skills": "PPS-Q-017", "gt_pass": True, "rating": "继续深挖", "return_5d": 5, "return_10d": 6, "return_20d": 10},
            {"triggered_skills": "PPS-Q-017", "gt_pass": False, "rating": "继续深挖", "return_5d": -6, "return_10d": -4, "return_20d": -2},
        ]
    )
    rules = build_reusable_rule_candidates(df, df, pd.DataFrame(), tmp_path / "reusable_rules")
    assert rules
    rule = rules[0]
    assert rule["derived_from"]["strategy_id"] == "PPS-Q-017"
    assert rule["formula"]
    assert rule["anti_leakage_checks"]["ground_truth_excluded_from_scoring"] is True
    assert "不得生成买卖指令" in rule["decision_effect"]


def _write_stock_cache(data_dir: Path, code: str) -> None:
    stock_dir = data_dir / code
    stock_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2025-01-02", periods=280)
    base = 10.0
    rows = []
    for i, date in enumerate(dates):
        close = base + i * 0.03 + (0.8 if i % 37 == 0 else 0)
        rows.append(
            {
                "date": date.date().isoformat(),
                "open": close - 0.05,
                "high": close + 0.12,
                "low": close - 0.12,
                "close": close,
                "volume": 1000000 + i * 1000,
                "amount": (1000000 + i * 1000) * close,
            }
        )
    pd.DataFrame(rows).to_csv(stock_dir / "daily.csv", index=False)
    financial = [
        {
            "report_period": "2025Q1",
            "publish_date": "2025-04-30",
            "yoypni": 12,
            "roe": 11,
            "debt_to_assets": 45,
            "current_ratio": 1.2,
        }
    ]
    (stock_dir / "financial.json").write_text(yaml.safe_dump(financial, allow_unicode=True), encoding="utf-8")
