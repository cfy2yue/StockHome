from __future__ import annotations

import json

import pandas as pd

from src.world_model.financial_report_channel import (
    build_financial_report_events,
    build_financial_report_feature_table,
    merge_financial_report_features_asof,
)


def test_financial_report_channel_builds_time_safe_events(tmp_path):
    cache = tmp_path / "tushare_pro"
    fina = cache / "tables" / "fina_indicator"
    anns = cache / "tables" / "anns_d"
    fina.mkdir(parents=True)
    anns.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20260425",
                "end_date": "20260331",
                "eps": "0.67",
                "roe": "2.65",
                "netprofit_yoy": "3.02",
                "op_yoy": "2.83",
                "ocf_yoy": "-76.80",
            }
        ]
    ).to_csv(fina / "000001.SZ.csv", index=False)
    pd.DataFrame(
        [
            {
                "ann_date": "20250430",
                "ts_code": "000002.SZ",
                "name": "样本公司",
                "title": "2024年年度报告",
                "url": "https://example.test/report.pdf",
            },
            {
                "ann_date": "20250430",
                "ts_code": "000003.SZ",
                "name": "非财报",
                "title": "关于召开股东大会的公告",
                "url": "https://example.test/meeting.pdf",
            },
        ]
    ).to_csv(anns / "20250430_20250430.csv", index=False)

    events = build_financial_report_events(cache)

    assert len(events) == 2
    assert set(events["financial_report_event_type"]) == {"quarterly_metrics", "annual_report"}
    assert set(events["available_at_guard_status"]) == {"date_only_next_day_conservative"}
    assert "2026-04-26 00:00:00" in set(events["available_at"])
    assert "2025-05-01 00:00:00" in set(events["available_at"])
    assert events["financial_report_missing_rate"].between(0, 1).all()


def test_financial_report_feature_table_rolls_up_same_day_events(tmp_path):
    cache = tmp_path / "tushare_pro"
    anns = cache / "tables" / "anns_d"
    anns.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ann_date": "20250430",
                "ts_code": "000002.SZ",
                "title": "2024年年度报告",
                "url": "https://example.test/report.pdf",
            },
            {
                "ann_date": "20250430",
                "ts_code": "000002.SZ",
                "title": "2024年年度审计报告",
                "url": "https://example.test/audit.pdf",
            },
        ]
    ).to_csv(anns / "20250430_20250430.csv", index=False)

    features = build_financial_report_feature_table(build_financial_report_events(cache))

    assert len(features) == 1
    row = features.iloc[0]
    assert row["decision_date"] == "2025-05-01"
    assert row["financial_report_event_count"] == 2
    assert "annual_report" in row["financial_report_event_types"]
    assert "audit_report" in row["financial_report_event_types"]


def test_financial_report_channel_builds_forecast_and_express_events(tmp_path):
    cache = tmp_path / "tushare_pro"
    forecast = cache / "tables" / "forecast"
    express = cache / "tables" / "express"
    forecast.mkdir(parents=True)
    express.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000004.SZ",
                "ann_date": "20260420",
                "end_date": "20260331",
                "type": "预增",
                "p_change_min": "20",
                "p_change_max": "40",
            }
        ]
    ).to_csv(forecast / "000004.SZ.csv", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000005.SZ",
                "ann_date": "20260422",
                "end_date": "20260331",
                "revenue": "1000",
                "n_income": "80",
                "yoy_sales": "12",
                "yoy_tp": "-5",
            }
        ]
    ).to_csv(express / "000005.SZ.csv", index=False)

    events = build_financial_report_events(cache)

    assert set(events["financial_report_event_type"]) == {"performance_forecast", "performance_express"}
    assert set(events["interface"]) == {"forecast", "express"}
    assert "2026-04-21 00:00:00" in set(events["available_at"])
    assert events["financial_report_missing_rate"].between(0, 1).all()


def test_financial_report_channel_builds_fina_audit_risk_events(tmp_path):
    cache = tmp_path / "tushare_pro"
    audit = cache / "tables" / "fina_audit"
    audit.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000006.SZ",
                "ann_date": "20260428",
                "end_date": "20251231",
                "audit_result": "标准无保留意见",
                "audit_agency": "样本会计师事务所",
            },
            {
                "ts_code": "000007.SZ",
                "ann_date": "20260429",
                "end_date": "20251231",
                "audit_result": "保留意见",
                "audit_agency": "样本会计师事务所",
            },
        ]
    ).to_csv(audit / "sample.csv", index=False)

    events = build_financial_report_events(cache).sort_values("ts_code").reset_index(drop=True)

    assert set(events["financial_report_event_type"]) == {"audit_opinion"}
    assert set(events["interface"]) == {"fina_audit"}
    assert events.loc[0, "financial_quality_risk_score"] == 0.0
    assert events.loc[1, "financial_quality_risk_score"] == 0.9
    assert events.loc[1, "financial_surprise_score"] == -0.9
    assert "2026-04-30 00:00:00" in set(events["available_at"])


def test_financial_report_channel_builds_statement_variance_events(tmp_path):
    cache = tmp_path / "tushare_pro"
    income = cache / "tables" / "income"
    cashflow = cache / "tables" / "cashflow"
    balancesheet = cache / "tables" / "balancesheet"
    income.mkdir(parents=True)
    cashflow.mkdir(parents=True)
    balancesheet.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000008.SZ",
                "ann_date": "20250425",
                "end_date": "20240331",
                "total_revenue": "1000",
                "n_income_attr_p": "100",
            },
            {
                "ts_code": "000008.SZ",
                "ann_date": "20260425",
                "end_date": "20250331",
                "total_revenue": "800",
                "n_income_attr_p": "-20",
            },
        ]
    ).to_csv(income / "000008.SZ.csv", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000008.SZ",
                "ann_date": "20260425",
                "end_date": "20250331",
                "n_cashflow_act": "-50",
                "c_fr_sale_sg": "700",
            }
        ]
    ).to_csv(cashflow / "000008.SZ.csv", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000008.SZ",
                "ann_date": "20260425",
                "end_date": "20250331",
                "total_assets": "1000",
                "total_liab": "880",
                "total_hldr_eqy_exc_min_int": "120",
            }
        ]
    ).to_csv(balancesheet / "000008.SZ.csv", index=False)

    events = build_financial_report_events(cache)

    assert {"income_statement", "cashflow_statement", "balance_sheet"}.issubset(set(events["financial_report_event_type"]))
    income_event = events[events["financial_report_event_type"].eq("income_statement") & events["report_period"].eq("20250331")].iloc[0]
    income_metrics = json.loads(income_event["key_metrics_json"])
    assert income_metrics["n_income_attr_p_yoy"] == -120.0
    assert income_event["financial_quality_risk_score"] > 0
    cashflow_event = events[events["financial_report_event_type"].eq("cashflow_statement")].iloc[0]
    assert cashflow_event["financial_quality_risk_score"] >= 0.3
    balance_event = events[events["financial_report_event_type"].eq("balance_sheet")].iloc[0]
    balance_metrics = json.loads(balance_event["key_metrics_json"])
    assert balance_metrics["liab_to_assets"] == 0.88
    assert balance_event["financial_quality_risk_score"] >= 0.45
    assert set(events["available_at"]) == {"2025-04-26 00:00:00", "2026-04-26 00:00:00"}


def test_merge_financial_report_features_asof_blocks_future_reports():
    decisions = pd.DataFrame(
        [
            {"date": "2025-04-30", "code": "000002"},
            {"date": "2025-05-02", "code": "000002"},
            {"date": "2025-05-02", "code": "000003"},
        ]
    )
    features = pd.DataFrame(
        [
            {
                "ts_code": "000002.SZ",
                "code": "000002",
                "decision_date": "2025-05-01",
                "available_at": "2025-05-01 00:00:00",
                "financial_report_event_count": 1,
                "financial_report_materiality_score": 0.8,
                "financial_quality_risk_score": 0.4,
                "financial_surprise_score": -0.2,
                "financial_disclosure_quality_score": 0.8,
                "financial_report_missing_rate": 0.0,
                "financial_report_latest_period": "20241231",
                "financial_report_event_types": "annual_report",
                "source_type": "paid_standardized",
                "source_name": "tushare_pro",
            }
        ]
    )

    merged = merge_financial_report_features_asof(decisions, features, window_days=90)

    first = merged[merged["date"].eq("2025-04-30")].iloc[0]
    second = merged[merged["date"].eq("2025-05-02")].iloc[0]
    missing_code = merged[merged["code"].eq("000003")].iloc[0]
    assert first["financial_report_join_status"] == "no_event_in_window"
    assert first["financial_report_missing_rate"] == 1.0
    assert second["financial_report_join_status"] == "event_window_matched"
    assert second["financial_report_event_count"] == 1
    assert second["financial_report_latest_period"] == "20241231"
    assert missing_code["financial_report_join_status"] == "code_not_in_feature_table"
