from scripts.prepare_backtest_news import _has_requested_sources


def test_has_requested_sources_uses_event_providers():
    events = [
        {"provider": "eastmoney_stock_news", "title": "新闻"},
        {"provider": "eastmoney_stock_notice", "公告标题": "公告"},
    ]
    assert _has_requested_sources(events, {"eastmoney_news"})
    assert _has_requested_sources(events, {"eastmoney_news", "eastmoney_notice"})
    assert not _has_requested_sources(events[:1], {"eastmoney_news", "eastmoney_notice"})
