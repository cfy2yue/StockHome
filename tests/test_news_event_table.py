from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.world_model.news_event_table import (
    build_event_feature_table,
    build_local_public_news_event_table,
    build_news_event_outputs,
    build_news_event_table,
    combine_news_event_tables,
    merge_event_features_asof,
)


def test_build_news_event_table_normalizes_announcements_and_news(tmp_path: Path) -> None:
    cache = tmp_path / "tushare"
    anns = cache / "tables" / "anns_d"
    news = cache / "tables" / "news"
    anns.mkdir(parents=True)
    news.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20260105",
                "name": "平安银行",
                "title": "重大诉讼公告",
                "url": "https://example.test/a",
            }
        ]
    ).to_csv(anns / "20260105_20260105.csv", index=False)
    pd.DataFrame(
        [
            {
                "pub_time": "2026-01-05 20:30:00",
                "title": "政策支持行业扩产",
                "content": "国务院政策支持行业扩产合作",
                "src": "sina",
            }
        ]
    ).to_csv(news / "sina_20260105_20260105.csv", index=False)

    events = build_news_event_table(cache)

    assert len(events) == 2
    ann = events[events["interface"].eq("anns_d")].iloc[0]
    assert ann["available_at"] == "2026-01-05 15:00:00"
    assert ann["available_at_guard_status"] == "date_only_close_assumed"
    assert ann["risk_score"] > 0
    assert ann["announcement_materiality_score"] >= 0.7
    item = events[events["interface"].eq("news")].iloc[0]
    assert item["available_at"] == "2026-01-05 20:30:00"
    assert item["policy_score"] > 0


def test_event_feature_table_uses_latest_available_at_for_same_day() -> None:
    events = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "code": "000001",
                "event_date": "2026-01-05",
                "available_at": "2026-01-05 15:00:00",
                "risk_score": 0.0,
                "opportunity_score": 0.0,
                "policy_score": 0.0,
                "official_confirmation_score": 1.0,
                "announcement_materiality_score": 0.7,
                "news_timestamp_quality": 0.7,
                "news_evidence_quality": 0.8,
            },
            {
                "ts_code": "000001.SZ",
                "code": "000001",
                "event_date": "2026-01-05",
                "available_at": "2026-01-05 21:00:00",
                "risk_score": 1.0,
                "opportunity_score": 0.0,
                "policy_score": 0.0,
                "official_confirmation_score": 1.0,
                "announcement_materiality_score": 1.0,
                "news_timestamp_quality": 1.0,
                "news_evidence_quality": 1.0,
            },
        ]
    )

    features = build_event_feature_table(events)

    assert len(features) == 1
    assert features.iloc[0]["available_at"] == "2026-01-05 21:00:00"
    assert features.iloc[0]["news_warning_score"] == 1.0


def test_build_news_event_outputs_writes_empty_schema_files(tmp_path: Path) -> None:
    events, features = build_news_event_outputs(tmp_path / "tushare")

    assert events.empty
    assert features.empty
    assert (tmp_path / "tushare" / "derived" / "news_event_table.csv").exists()
    assert (tmp_path / "tushare" / "derived" / "news_world_model_event_features.csv").exists()


def test_build_local_public_news_event_table_reads_yaml_cache(tmp_path: Path) -> None:
    stock_dir = tmp_path / "000001"
    stock_dir.mkdir()
    (stock_dir / "news.json").write_text(
        """
meta:
  code: '000001'
  name: 平安银行
events:
- title: 公司获得重大订单
  content: 公司获得重大订单并签订合作协议
  datetime: '2026-06-23 10:30:00'
  source: 东方财富
  source_type: 公开新闻
  provider: eastmoney_stock_news
  code: '000001'
- 公告标题: 平安银行:重大诉讼公告
  公告日期: '2026-06-24'
  source: 东方财富公告大全
  source_type: 官方公告
  provider: eastmoney_stock_notice
  code: '000001'
  公告类型: 诉讼
""",
        encoding="utf-8",
    )

    events = build_local_public_news_event_table(tmp_path)
    by_title = {row["title"]: row for _, row in events.iterrows()}

    assert len(events) == 2
    assert set(events["source_type"]) == {"public_aggregator"}
    assert by_title["公司获得重大订单"]["available_at"] == "2026-06-23 10:30:00"
    assert by_title["平安银行:重大诉讼公告"]["available_at"] == "2026-06-24 15:00:00"
    assert by_title["平安银行:重大诉讼公告"]["available_at_guard_status"] == "date_only_close_assumed"
    assert by_title["平安银行:重大诉讼公告"]["official_confirmation_score"] > 0.8


def test_combine_news_event_tables_dedupes_event_id(tmp_path: Path) -> None:
    stock_dir = tmp_path / "000001"
    stock_dir.mkdir()
    (stock_dir / "news.json").write_text(
        """
events:
- title: 公司获得重大订单
  datetime: '2026-06-23 10:30:00'
  provider: eastmoney_stock_news
  code: '000001'
""",
        encoding="utf-8",
    )
    local = build_local_public_news_event_table(tmp_path)
    combined = combine_news_event_tables(local, local)

    assert len(local) == 1
    assert len(combined) == 1


def test_merge_event_features_asof_respects_available_at_and_window() -> None:
    decisions = pd.DataFrame(
        [
            {"date": "2026-01-05", "code": "1", "name": "测试A"},
            {"date": "2026-01-06", "code": "1", "name": "测试A"},
            {"date": "2026-02-20", "code": "1", "name": "测试A"},
        ]
    )
    features = pd.DataFrame(
        [
            {
                "code": "000001",
                "decision_date": "2026-01-05",
                "available_at": "2026-01-05 15:00:00",
                "event_count": 2,
                "source_type": "paid_standardized",
                "source_name": "tushare_pro",
                "news_warning_score": 0.8,
                "news_opportunity_score": 0.0,
                "policy_background_score": 0.1,
                "official_confirmation_score": 1.0,
                "announcement_materiality_score": 0.9,
                "news_timestamp_quality": 0.7,
                "news_evidence_quality": 0.9,
            },
            {
                "code": "000001",
                "decision_date": "2026-01-05",
                "available_at": "2026-01-05 21:00:00",
                "event_count": 9,
                "source_type": "public_aggregator",
                "source_name": "eastmoney_stock_news",
                "news_warning_score": 1.0,
                "news_opportunity_score": 0.0,
                "policy_background_score": 0.0,
                "official_confirmation_score": 1.0,
                "announcement_materiality_score": 1.0,
                "news_timestamp_quality": 1.0,
                "news_evidence_quality": 1.0,
            },
        ]
    )

    merged = merge_event_features_asof(decisions, features, window_days=30)
    by_date = {row["date"]: row for _, row in merged.iterrows()}

    assert by_date["2026-01-05"]["event_count"] == 2
    assert by_date["2026-01-05"]["news_warning_score"] == 0.8
    assert by_date["2026-01-05"]["news_missing_rate"] == 0.0
    assert by_date["2026-01-05"]["source_type"] == "paid_standardized"
    assert by_date["2026-01-06"]["event_count"] == 11
    assert by_date["2026-01-06"]["source_type"] == "paid_standardized+public_aggregator"
    assert by_date["2026-02-20"]["news_missing_rate"] == 1.0
    assert by_date["2026-02-20"]["news_event_table_join_status"] == "no_event_in_window"
