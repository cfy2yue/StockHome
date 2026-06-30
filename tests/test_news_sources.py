from src.backtest.news_sources import merge_events


def test_merge_events_dedupes_and_sorts():
    events = merge_events(
        [
            {"code": "000001", "title": "公司获得订单", "datetime": "2026-06-01 10:00:00", "provider": "a"},
            {"code": "000001", "title": "公司获得订单", "datetime": "2026-06-01 10:00:00", "provider": "a"},
        ],
        [{"code": "000001", "公告标题": "公司业绩预增公告", "公告日期": "2026-06-02", "provider": "b"}],
    )
    assert len(events) == 2
    assert events[0]["公告标题"] == "公司业绩预增公告"
