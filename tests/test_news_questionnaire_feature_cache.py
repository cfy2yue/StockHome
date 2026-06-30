from __future__ import annotations

import os

import pandas as pd

from scripts.build_news_questionnaire_feature_cache import (
    agent_preview_rows,
    build_coverage,
    build_feature_cache,
)
from src.world_model.news_questionnaire import load_news_questionnaire


def test_build_feature_cache_drops_future_fields_and_keeps_latest(tmp_path) -> None:
    config = load_news_questionnaire()
    older = tmp_path / "news_questionnaire_flash_old_scores.csv"
    newer = tmp_path / "news_questionnaire_flash_new_scores.csv"
    row = {
        "decision_date": "2026-01-06",
        "code": "1",
        "questionnaire_version": "news_semantic_questionnaire_v1",
        "mainline_summary": "old",
        "missing_or_conflict_notes": "old note",
        "ds_news_risk_score": 0.3,
        "ds_news_uncertainty_score": 0.4,
        "ds_news_net_score": -0.2,
        "return_20d": 12.3,
        "gt_status": "evaluated",
    }
    pd.DataFrame([row]).to_csv(older, index=False)
    pd.DataFrame([{**row, "mainline_summary": "new", "ds_news_risk_score": 0.7, "ds_news_uncertainty_score": 0.65}]).to_csv(newer, index=False)
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    features, meta = build_feature_cache([older, newer], config=config)

    assert len(features) == 1
    result = features.iloc[0]
    assert result["code"] == "000001"
    assert result["ds_news_mainline_summary"] == "new"
    assert bool(result["ds_news_risk_guard"]) is True
    assert bool(result["ds_news_uncertainty_guard"]) is True
    assert bool(result["usable_as_positive_alpha_default"]) is False
    assert "return_20d" not in features.columns
    assert "gt_status" not in features.columns
    assert meta["dropped_future_columns"] == ["gt_status", "return_20d"]


def test_agent_preview_omits_future_fields() -> None:
    features = pd.DataFrame(
        [
            {
                "decision_date": "2026-01-06",
                "code": "000001",
                "news_semantic_questionnaire_version": "news_semantic_questionnaire_v1",
                "default_agent_use": "risk_uncertainty_explanation_not_positive_alpha",
                "usable_as_positive_alpha_default": False,
                "ds_news_positive_alpha_status": "not_accepted_default",
                "ds_news_risk_score": 0.7,
                "ds_news_uncertainty_score": 0.65,
                "ds_news_quality_score": 0.5,
                "ds_news_net_score": -0.4,
                "ds_news_mainline_summary": "risk",
                "source_score_file": "unit.csv",
                "research_only": True,
                "not_investment_instruction": True,
                "return_20d": 9.9,
            }
        ]
    )

    preview = agent_preview_rows(features)

    assert len(preview) == 1
    assert "return_20d" not in preview[0]
    assert preview[0]["forbidden_use"] == "do_not_use_as_positive_alpha_or_order_instruction"


def test_build_coverage_uses_only_date_code_time_block(tmp_path) -> None:
    joined = tmp_path / "joined.csv"
    pd.DataFrame(
        [
            {"date": "2026-01-06", "code": "1", "time_block": "H2026_1", "return_20d": 10.0},
            {"date": "2026-01-07", "code": "2", "time_block": "H2026_1", "return_20d": -5.0},
        ]
    ).to_csv(joined, index=False)
    features = pd.DataFrame([{"decision_date": "2026-01-06", "code": "000001"}])

    coverage = build_coverage(features, joined)

    block = coverage[coverage["scope"].eq("H2026_1")].iloc[0]
    assert block["gt_rows"] == 2
    assert block["questionnaire_matched_rows"] == 1
    assert block["coverage"] == 0.5
