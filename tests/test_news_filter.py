from __future__ import annotations

from src.backtest.news_filter import filter_news_for_decision


def test_news_filter_excludes_future_intraday_news():
    events = [
        {"标题": "公司获得重大订单", "发布时间": "2025-09-01 14:30:00", "来源": "公开新闻"},
        {"标题": "公司收到监管处罚", "发布时间": "2025-09-01 16:30:00", "来源": "公开新闻"},
    ]
    out = filter_news_for_decision(events, "2025-09-01")
    assert [event.title for event in out] == ["公司获得重大订单"]


def test_news_filter_date_only_available_next_day():
    events = [
        {"公告标题": "公司业绩预增公告", "公告日期": "2025-09-01", "来源": "交易所公告"},
    ]
    same_day = filter_news_for_decision(events, "2025-09-01")
    next_day = filter_news_for_decision(events, "2025-09-02")
    assert same_day == []
    assert next_day[0].source_type == "官方公告"
    assert next_day[0].materiality_score >= 7


def test_news_filter_dedupes_and_sorts_by_materiality():
    events = [
        {"标题": "公司一般新闻", "发布时间": "2025-09-01 10:00:00", "来源": "公开新闻"},
        {"标题": "公司一般新闻", "发布时间": "2025-09-01 10:00:00", "来源": "公开新闻"},
        {"标题": "公司股东减持计划", "发布时间": "2025-09-01 09:00:00", "来源": "公开新闻"},
    ]
    out = filter_news_for_decision(events, "2025-09-01")
    assert len(out) == 2
    assert out[0].event_type == "减持"
