from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_financial_report_strata.py"


def _module():
    spec = importlib.util.spec_from_file_location("analyze_financial_report_strata", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-04-30",
                "code": "000001",
                "name": "样本A",
                "return_20d": -8.0,
                "prior_return_20d": 25.0,
                "rsi14": 74.0,
                "relative_strength_rank": 0.7,
                "news_missing_rate": 0.1,
                "triggered_skills": "CORE_TREND_001",
                "financial_report_event_count": 1,
                "financial_report_join_status": "event_window_matched",
                "financial_report_materiality_score": 0.8,
                "financial_quality_risk_score": 0.5,
                "financial_surprise_score": -0.6,
                "financial_disclosure_quality_score": 0.7,
                "financial_report_event_types": "annual_report;performance_forecast",
            },
            {
                "date": "2025-05-06",
                "code": "000002",
                "name": "样本B",
                "return_20d": 6.0,
                "prior_return_20d": 3.0,
                "rsi14": 55.0,
                "relative_strength_rank": 0.6,
                "news_missing_rate": 0.2,
                "triggered_skills": "QUALITY_001",
                "financial_report_event_count": 1,
                "financial_report_join_status": "event_window_matched",
                "financial_report_materiality_score": 0.7,
                "financial_quality_risk_score": 0.1,
                "financial_surprise_score": 0.7,
                "financial_disclosure_quality_score": 0.8,
                "financial_report_event_types": "quarterly_report",
            },
            {
                "date": "2025-05-08",
                "code": "000003",
                "name": "样本C",
                "return_20d": -3.0,
                "prior_return_20d": 1.0,
                "rsi14": 50.0,
                "relative_strength_rank": 0.1,
                "news_missing_rate": 0.9,
                "triggered_skills": "",
                "financial_report_event_count": 3,
                "financial_report_join_status": "event_window_matched",
                "financial_report_materiality_score": 0.6,
                "financial_quality_risk_score": 0.65,
                "financial_surprise_score": 0.5,
                "financial_disclosure_quality_score": 0.4,
                "financial_report_event_types": "audit_report;financial_inquiry",
            },
        ]
    )


def test_add_strata_columns_marks_financial_flags() -> None:
    module = _module()

    enriched = module.add_strata_columns(_sample_frame())

    assert enriched.loc[0, "time_block"] == "H2025_1"
    assert bool(enriched.loc[0, "negative_surprise_overheat_flag"]) is True
    assert bool(enriched.loc[1, "positive_surprise_low_risk_flag"]) is True
    assert bool(enriched.loc[2, "positive_surprise_weak_context_flag"]) is True
    assert bool(enriched.loc[2, "quality_risk_high_flag"]) is True
    assert enriched.loc[0, "event_type_primary"] == "performance_forecast"


def test_candidate_ds_plan_excludes_future_return() -> None:
    module = _module()
    enriched = module.add_strata_columns(_sample_frame())

    sample_audit, ds_plan = module.build_candidate_samples(enriched, max_samples_per_rule=2)

    assert "return_20d" in sample_audit.columns
    assert "return_20d" not in ds_plan.columns
    assert set(ds_plan["candidate_rule"]) >= {
        "financial_negative_surprise_overheat_guard_v1",
        "financial_positive_surprise_low_risk_candidate_v1",
    }


def test_diversified_guard_plan_excludes_future_return_and_caps_stock_count() -> None:
    module = _module()
    rows = []
    for idx in range(6):
        rows.append(
            {
                "date": f"2025-05-{idx + 1:02d}",
                "code": f"{idx + 1:06d}",
                "name": f"样本{idx}",
                "return_20d": -5.0 + idx,
                "prior_return_20d": 25.0,
                "rsi14": 72.0,
                "relative_strength_rank": 0.7,
                "news_missing_rate": 0.9,
                "triggered_skills": "QUALITY_001",
                "financial_report_event_count": 1,
                "financial_report_join_status": "event_window_matched",
                "financial_report_materiality_score": 0.8,
                "financial_quality_risk_score": 0.7,
                "financial_surprise_score": -0.6,
                "financial_disclosure_quality_score": 0.7,
                "financial_report_event_types": "annual_report",
            }
        )
    for idx in range(6, 10):
        rows.append(
            {
                "date": f"2025-05-{idx + 1:02d}",
                "code": f"{idx + 1:06d}",
                "name": f"中性{idx}",
                "return_20d": 2.0,
                "prior_return_20d": 2.0,
                "rsi14": 50.0,
                "relative_strength_rank": 0.5,
                "news_missing_rate": 0.2,
                "triggered_skills": "QUALITY_001",
                "financial_report_event_count": 1,
                "financial_report_join_status": "event_window_matched",
                "financial_report_materiality_score": 0.6,
                "financial_quality_risk_score": 0.2,
                "financial_surprise_score": 0.0,
                "financial_disclosure_quality_score": 0.8,
                "financial_report_event_types": "quarterly_report",
            }
        )
    enriched = module.add_strata_columns(pd.DataFrame(rows))

    audit, ds_plan = module.build_diversified_guard_samples(enriched, target_rows=8, max_per_stock=1, max_per_block=0)

    assert len(ds_plan) == 8
    assert "return_20d" in audit.columns
    assert "return_20d" not in ds_plan.columns
    assert ds_plan["code"].value_counts().max() == 1
    assert "financial_report_neutral_control_v1" in set(ds_plan["candidate_rule"])


def test_rule_metrics_keep_small_n_as_observe() -> None:
    module = _module()
    enriched = module.add_strata_columns(_sample_frame())

    rules = module.build_rule_candidate_metrics(enriched)

    assert set(rules["status"]) == {"observe_small_n"}
    assert rules["baseline_rows"].eq(3).all()
