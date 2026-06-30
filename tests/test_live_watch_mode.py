from __future__ import annotations

from pathlib import Path

from src.data.schemas import FetchResult
from src.live_watch import (
    LiveWatchConfig,
    LiveWatchSession,
    recommendation_from_research_grade,
    render_live_watch_markdown,
    stance_from_research_grade,
)


class FakeLiveAdapter:
    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.news_calls = 0

    def quote_realtime(self, code: str) -> FetchResult:
        if not self.ok:
            return FetchResult(False, "fake quote", None, error="offline", fetched_at="2026-06-28 10:00:00")
        return FetchResult(
            True,
            "fake quote",
            [{"price": 10.5, "last_close": 10.0, "open": 10.1, "high": 10.6, "low": 10.0}],
            fetched_at="2026-06-28 10:00:00",
        )

    def kline_intraday(self, code: str, frequency: str = "5m", limit: int = 80) -> FetchResult:
        if not self.ok:
            return FetchResult(False, "fake intraday", None, error="offline", fetched_at="2026-06-28 10:00:00")
        rows = [
            {"datetime": "2026-06-28 09:35:00", "close": 10.2, "high": 10.25, "low": 10.0},
            {"datetime": "2026-06-28 09:40:00", "close": 10.5, "high": 10.6, "low": 10.3},
        ]
        return FetchResult(True, "fake intraday", rows, fetched_at="2026-06-28 10:00:00")

    def kline_today_daily(self, code: str, limit: int = 40) -> FetchResult:
        if not self.ok:
            return FetchResult(False, "fake daily", None, error="offline", fetched_at="2026-06-28 10:00:00")
        return FetchResult(True, "fake daily", [{"close": 10.0}, {"close": 10.5}], fetched_at="2026-06-28 10:00:00")

    def stock_news(self, code: str) -> FetchResult:
        self.news_calls += 1
        if not self.ok:
            return FetchResult(False, "fake news", None, error="offline", fetched_at="2026-06-28 10:00:00")
        return FetchResult(True, "fake news", [{"标题": "公司获得重要订单"}], fetched_at="2026-06-28 10:00:00")

    def stock_announcements(self, code: str) -> FetchResult:
        if not self.ok:
            return FetchResult(False, "fake announcements", None, error="offline", fetched_at="2026-06-28 10:00:00")
        return FetchResult(True, "fake announcements", [{"公告标题": "项目获批公告"}], fetched_at="2026-06-28 10:00:00")

    def current_quantitative(self, code: str) -> FetchResult:
        if not self.ok:
            return FetchResult(False, "fake quant", None, error="offline", fetched_at="2026-06-28 10:00:00")
        return FetchResult(True, "fake quant", [{"名称": "测试股份"}], fetched_at="2026-06-28 10:00:00")

    def financial_indicator(self, code: str) -> FetchResult:
        if not self.ok:
            return FetchResult(False, "fake financial", None, error="offline", fetched_at="2026-06-28 10:00:00")
        return FetchResult(True, "fake financial", [{"报告期": "2026Q1"}], fetched_at="2026-06-28 10:00:00")


def _config(tmp_path: Path) -> LiveWatchConfig:
    return LiveWatchConfig(code="000001", name="测试股份", cache_dir=tmp_path / "cache", report_dir=tmp_path / "reports")


def test_live_watch_returns_research_grade_and_uses_daily_cache(tmp_path: Path) -> None:
    adapter = FakeLiveAdapter(ok=True)
    session = LiveWatchSession(_config(tmp_path), adapter=adapter, sleeper=lambda _: None)
    first = session.run_once()
    second = session.run_once()
    assert first.research_grade in {"继续深挖", "放入观察", "暂时剔除", "信息不足"}
    assert second.research_grade in {"继续深挖", "放入观察", "暂时剔除", "信息不足"}
    assert adapter.news_calls == 1
    text = render_live_watch_markdown(first)
    assert "研究分级" in text
    assert "明确建议" in text
    assert "明确观点" in text
    assert "仓位计划" in text
    assert "价格触发" in text
    assert "风控线" in text
    assert "升级条件" in text
    assert "研究辅助型操作建议" in text
    assert "不保证收益" in text


def test_live_watch_degrades_to_insufficient_when_sources_fail(tmp_path: Path) -> None:
    session = LiveWatchSession(_config(tmp_path), adapter=FakeLiveAdapter(ok=False), sleeper=lambda _: None)
    decision = session.run_once()
    assert decision.research_grade == "信息不足"
    assert "先补数据" in decision.user_stance
    assert decision.data_missing_flags


def test_stance_from_grade_contains_action_plan() -> None:
    assert "试探买入" in recommendation_from_research_grade("继续深挖")
    assert "暂不新增买入/加仓" in recommendation_from_research_grade("放入观察")
    assert "减仓或卖出" in recommendation_from_research_grade("暂时剔除")
    assert "暂不交易" in recommendation_from_research_grade("信息不足")
    assert "小仓试错" in stance_from_research_grade("继续深挖")
    assert "暂不买入/加仓" in stance_from_research_grade("放入观察")
    assert "降低仓位或退出" in stance_from_research_grade("暂时剔除")
    assert "不给买入或加仓建议" in stance_from_research_grade("信息不足")
