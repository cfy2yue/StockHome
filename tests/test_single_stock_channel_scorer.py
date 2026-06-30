from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_channel_score_loader_drops_future_result_fields(tmp_path: Path) -> None:
    module = _load_script("audit_single_stock_channel_scorer_v1")
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "code": "1",
                "logistic_channel_outcome__prob_hard_counter": 0.91,
                "logistic_channel_outcome__prob_soft_gap": 0.10,
                "logistic_channel_outcome__prob_positive_support": 0.02,
                "logistic_channel_outcome__prob_neutral": 0.03,
                "return_20d": -9.0,
                "pool_excess_20d": -4.0,
                "rule_outcome_label": "hard_counter",
            }
        ]
    ).to_csv(csv_path, index=False)

    scores, features = module.load_safe_channel_scores(csv_path)

    assert scores.loc[0, "code"] == "000001"
    assert "channel_hard_counter_prob" in scores.columns
    assert "channel_counter_gap_prob" in scores.columns
    assert "channel_hard_counter_yellow_flag" in features
    assert "return_20d" not in scores.columns
    assert "pool_excess_20d" not in scores.columns
    assert "rule_outcome_label" not in scores.columns


def test_agent_preview_contains_only_research_grades_and_no_future_fields() -> None:
    module = _load_script("audit_single_stock_channel_scorer_v1")
    target = pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "code": "000001",
                "time_block": "H2026_1",
                "channel_hard_counter_prob": 0.96,
                "channel_soft_gap_prob": 0.20,
                "channel_positive_support_prob": 0.01,
                "channel_neutral_prob": 0.02,
                "channel_counter_gap_prob": 0.95,
                "channel_soft_or_hard_prob": 1.16,
                "channel_positive_gap_prob": -0.95,
                "channel_hard_counter_yellow_flag": 0.0,
                "channel_hard_counter_high_flag": 1.0,
                "channel_soft_gap_dominant_flag": 0.0,
                "channel_score_coverage": 1.0,
                "return_20d": -8.0,
                "single_stock_label": "reduce_or_exclude",
            }
        ]
    )
    opp = pd.DataFrame({"ml_score": [0.01]})
    risk = pd.DataFrame({"risk_score": [0.90]})

    preview = module.build_agent_preview(target, opp, risk, opp_threshold=0.10, risk_threshold=0.50)

    assert preview.loc[0, "research_grade"] in {"继续深挖", "放入观察", "暂时剔除", "信息不足"}
    assert preview.loc[0, "research_grade"] == "暂时剔除"
    assert "return_20d" not in preview.columns
    assert "single_stock_label" not in preview.columns
    assert "买入" not in preview.to_string()
    assert "卖出" not in preview.to_string()


def test_agent_preview_does_not_remove_on_low_hard_counter_risk_only() -> None:
    module = _load_script("audit_single_stock_channel_scorer_v1")
    target = pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "code": "000001",
                "time_block": "H2026_1",
                "channel_hard_counter_prob": 0.30,
                "channel_soft_gap_prob": 0.20,
                "channel_positive_support_prob": 0.01,
                "channel_neutral_prob": 0.49,
                "channel_score_coverage": 1.0,
            }
        ]
    )
    opp = pd.DataFrame({"ml_score": [0.01]})
    risk = pd.DataFrame({"risk_score": [0.90]})

    preview = module.build_agent_preview(target, opp, risk, opp_threshold=0.10, risk_threshold=0.50)

    assert preview.loc[0, "research_grade"] == "放入观察"
