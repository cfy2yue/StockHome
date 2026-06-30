from __future__ import annotations

from src.backtest.news_vector import NEWS_VECTOR_DIMENSIONS, vectorize_news_events


def test_news_vector_has_fixed_32_dimensions_and_time_window():
    events = [
        {"标题": "公司获得重大订单", "发布时间": "2025-09-01 14:30:00", "来源": "公开新闻"},
        {"标题": "公司股东减持计划", "发布时间": "2025-09-01 09:00:00", "来源": "公开新闻"},
        {"标题": "公司业绩预增公告", "公告日期": "2025-09-01", "来源": "交易所公告"},
        {"标题": "公司收到监管处罚", "发布时间": "2025-09-01 16:30:00", "来源": "公开新闻"},
    ]
    vector = vectorize_news_events(events, "2025-09-01")
    assert len(NEWS_VECTOR_DIMENSIONS) == 32
    assert set(vector) == set(NEWS_VECTOR_DIMENSIONS)
    assert vector["news_count_30d"] == 2
    assert vector["news_positive_materiality_30d"] > 0
    assert vector["news_negative_materiality_30d"] > 0
    assert vector["news_conflict_intensity_30d"] > 0
    assert vector["news_official_count_30d"] == 0


def test_news_vector_date_only_announcement_available_next_day():
    events = [{"标题": "公司业绩预增公告", "公告日期": "2025-09-01", "来源": "交易所公告"}]
    same_day = vectorize_news_events(events, "2025-09-01")
    next_day = vectorize_news_events(events, "2025-09-02")
    assert same_day["news_count_30d"] == 0
    assert next_day["news_official_count_30d"] == 1
