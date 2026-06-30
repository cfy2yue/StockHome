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


def test_decision_point_table_excludes_future_fields_and_builds_key_points() -> None:
    module = _load_script("build_decision_point_table")
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-07",
                "code": "000001",
                "name": "样本A",
                "time_block": "H2025_1",
                "return_20d": 9.9,
                "financial_quality_risk_score": 0.7,
                "news_missing_rate": 0.1,
                "prior_return_20d": -12.0,
            },
            {
                "date": "2025-01-10",
                "code": "000002",
                "name": "样本B",
                "time_block": "H2025_1",
                "return_20d": -3.0,
                "news_warning_score": 0.7,
                "news_missing_rate": 0.2,
                "prior_return_20d": 2.0,
            },
        ]
    )

    table = module.build_decision_point_table(frame)

    assert "return_20d" not in table.columns
    assert "gt_status" not in table.columns
    assert table["research_only"].all()
    assert table["not_investment_instruction"].all()
    assert (table["decision_frequency"] == "key_points_only").any()
    assert table["trigger_reason"].astype(str).str.contains("financial_quality_risk_score|news_warning_score|prior_return_20d").any()


def test_single_stock_labels_and_agent_tool_outcome_do_not_leak_future_fields() -> None:
    module = _load_script("run_quant_tool_minimal_experiment")

    positive = pd.Series({"return_5d": 1.0, "return_10d": 2.0, "return_20d": 5.0})
    weak = pd.Series({"return_5d": -4.5, "return_10d": -1.0, "return_20d": 1.0})
    missing = pd.Series({"return_5d": pd.NA, "return_10d": 0.0, "return_20d": 0.0})

    assert module.label_single_stock(positive) == "increase_research"
    assert module.label_single_stock(weak) == "reduce_or_exclude"
    assert module.label_single_stock(missing) == "insufficient"

    outcome = {
        "tool_id": "single_stock_risk_opportunity_score_minimal_v1",
        "tool_version": "quant_tool_minimal_v1",
        "task_mode": "single_stock_watch",
        "policy_profile": "mid_horizon_research",
        "decision_frequency": "scheduled_twice_weekly",
        "score": 0.1,
        "score_quantile": "offline_rank_top24",
        "confidence": 0.3,
        "action_hint": "仅作反证或灰色参考",
        "top_features": ["prior_return_20d", "rsi14"],
        "missing_flags": [],
        "counter_evidence": ["latest_time_block_failed"],
        "source_ref_ids": ["local_gt_cache"],
        "train_valid_test_blocks": "rolling_H2024_2_to_H2026_1",
        "promotion_status": "observe_latest_block_failed",
        "research_only": True,
        "not_investment_instruction": True,
    }

    module.sanitize_tool_outcome_for_agent(outcome)


def test_agent_tool_outcome_rejects_future_result_terms() -> None:
    module = _load_script("run_quant_tool_minimal_experiment")
    outcome = {
        "tool_id": "bad",
        "return_20d": 1.2,
    }

    try:
        module.sanitize_tool_outcome_for_agent(outcome)
    except ValueError as exc:
        assert "future/result" in str(exc)
    else:
        raise AssertionError("expected future-field rejection")
