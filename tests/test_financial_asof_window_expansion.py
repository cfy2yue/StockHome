from __future__ import annotations

import pytest

import pandas as pd

import scripts.audit_financial_asof_window_expansion as audit_module
from scripts.audit_financial_asof_window_expansion import (
    DEFAULT_JOINED_GT_CACHE_PATH,
    add_financial_flags,
    assert_no_future_fields,
    build_agent_previews,
    aggregate_metrics,
    evaluate_rules,
    load_or_build_window_detail,
    load_base_frame,
    policy_status,
    select_rule_rows,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "name": "A",
                "time_block": "H2026_1",
                "return_20d": 5.0,
                "window_days": 90,
                "scope": "all_pool",
                "financial_report_join_status": "event_window_matched",
                "financial_report_event_count": 2,
                "financial_report_materiality_score": 0.8,
                "financial_quality_risk_score": 0.1,
                "financial_surprise_score": 0.3,
                "financial_disclosure_quality_score": 0.85,
                "financial_report_missing_rate": 0.0,
                "financial_report_event_types": "annual_report",
            },
            {
                "date": "2026-01-02",
                "code": "000002",
                "name": "B",
                "time_block": "H2026_1",
                "return_20d": -6.0,
                "window_days": 90,
                "scope": "all_pool",
                "financial_report_join_status": "event_window_matched",
                "financial_report_event_count": 1,
                "financial_report_materiality_score": 0.9,
                "financial_quality_risk_score": 0.6,
                "financial_surprise_score": -0.4,
                "financial_disclosure_quality_score": 0.7,
                "financial_report_missing_rate": 0.0,
                "financial_report_event_types": "financial_inquiry",
            },
        ]
    )


def test_financial_flags_separate_positive_and_risk_rules() -> None:
    flagged = add_financial_flags(_frame())

    first = flagged.iloc[0]
    second = flagged.iloc[1]
    assert first["financial_quality_low_risk"]
    assert first["financial_positive_surprise_low_risk"]
    assert first["financial_multi_event_review"]
    assert not first["financial_high_risk_guard"]
    assert second["financial_high_risk_guard"]


def test_select_rule_rows_uses_required_flags() -> None:
    flagged = add_financial_flags(_frame())
    selected = select_rule_rows(flagged, {"required_flags": ["financial_high_risk_guard"]})

    assert selected["code"].tolist() == ["000002"]


def test_agent_preview_does_not_contain_future_fields() -> None:
    flagged = add_financial_flags(_frame())
    metrics = evaluate_rules(flagged)
    aggregate = aggregate_metrics(metrics)
    previews = build_agent_previews(aggregate)

    assert previews
    text = str(previews)
    assert "return_20d" not in text
    assert "pool_excess_20d" not in text
    assert "gt_status" not in text
    for preview in previews:
        assert_no_future_fields(preview)


def test_policy_status_rejects_too_few_samples() -> None:
    status = policy_status(
        {
            "total_selected_rows": 10,
            "h2026_selected_rows": 2,
            "max_top_stock_concentration": 0.1,
            "direction": "positive",
            "prior_pool_excess_20d": 1.0,
            "h2026_pool_excess_20d": 1.0,
            "prior_positive_rate_lift": 0.1,
            "h2026_positive_rate_lift": 0.1,
            "prior_loss_gt5_lift": -0.1,
            "h2026_loss_gt5_lift": -0.1,
        }
    )

    assert status == "reject_too_few_samples"


def test_assert_no_future_fields_rejects_result_key() -> None:
    with pytest.raises(ValueError):
        assert_no_future_fields({"positive_20d_rate": 0.7})


def test_default_base_loader_rebuilds_joined_cache_through_ground_truth_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    source_frame = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "1",
                "name": "A",
                "gt_status": "evaluated",
                "return_20d": 1.0,
            }
        ]
    )

    def fake_load_ground_truth(paths, **kwargs):
        calls["paths"] = list(paths)
        calls["kwargs"] = kwargs
        return source_frame.copy()

    def fake_ranker_details(frame, **kwargs):
        return {"score_quantile": pd.Series([1.0], index=frame.index)}

    monkeypatch.setattr(audit_module, "load_ground_truth", fake_load_ground_truth)
    monkeypatch.setattr(audit_module, "_portfolio_ranker_details", fake_ranker_details)

    loaded = load_base_frame(
        DEFAULT_JOINED_GT_CACHE_PATH,
        ground_truth_sources=[audit_module.ROOT / "dummy_ground_truth.csv"],
    )

    assert calls["paths"]
    assert calls["kwargs"]["kline_features_path"] == audit_module.DEFAULT_KLINE_FEATURES_PATH
    assert loaded["code"].tolist() == ["000001"]


def test_window_detail_cache_reuses_matching_source_fingerprints(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    joined_path = tmp_path / "joined.csv"
    financial_path = tmp_path / "financial.csv"
    detail_cache = tmp_path / "detail.csv.gz"
    joined_path.write_text("joined", encoding="utf-8")
    financial_path.write_text("financial", encoding="utf-8")
    base = pd.DataFrame([{"date": "2026-01-02", "code": "000001", "return_20d": 1.0}])
    features = pd.DataFrame([{"code": "000001", "available_at": "2026-01-01 00:00:00"}])
    detail = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "000001",
                "window_days": 90,
                "scope": "all_pool",
                "return_20d": 1.0,
            }
        ]
    )
    calls = {"count": 0}

    def fake_build_window_detail(*args, **kwargs):
        calls["count"] += 1
        return detail.copy()

    monkeypatch.setattr(audit_module, "build_window_detail", fake_build_window_detail)

    built, status = load_or_build_window_detail(
        base,
        features,
        windows=[90],
        high_ranker_quantile=0.8,
        joined_cache_path=joined_path,
        financial_features_path=financial_path,
        detail_cache_path=detail_cache,
    )
    assert status == "miss_rebuilt"
    assert calls["count"] == 1
    assert detail_cache.exists()

    def fail_build_window_detail(*args, **kwargs):
        raise AssertionError("cache was not reused")

    monkeypatch.setattr(audit_module, "build_window_detail", fail_build_window_detail)
    loaded, status = load_or_build_window_detail(
        base,
        features,
        windows=[90],
        high_ranker_quantile=0.8,
        joined_cache_path=joined_path,
        financial_features_path=financial_path,
        detail_cache_path=detail_cache,
    )
    assert status == "hit"
    assert loaded["code"].tolist() == ["000001"]
    assert built["window_days"].tolist() == [90]
