from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class NewsFetchResult:
    events: list[dict[str, Any]]
    errors: list[str]


def fetch_eastmoney_stock_news(code: str) -> NewsFetchResult:
    try:
        import akshare as ak

        df = ak.stock_news_em(symbol=code)
    except Exception as exc:
        return NewsFetchResult([], [f"{code}: eastmoney_stock_news failed: {exc}"])
    events = []
    for _, row in df.iterrows():
        events.append(
            {
                "title": _text(row.get("新闻标题")),
                "content": _text(row.get("新闻内容")),
                "datetime": _text(row.get("发布时间")),
                "source": _text(row.get("文章来源") or "东方财富个股新闻"),
                "source_type": "公开新闻",
                "url": _text(row.get("新闻链接")),
                "provider": "eastmoney_stock_news",
                "code": code,
            }
        )
    return NewsFetchResult([event for event in events if event["title"]], [])


def fetch_eastmoney_stock_notices(code: str, begin_date: str, end_date: str) -> NewsFetchResult:
    try:
        from akshare.stock_fundamental import stock_notice

        df = stock_notice._stock_notice_report(security=code, symbol="全部", begin_date=begin_date, end_date=end_date)
    except Exception as exc:
        return NewsFetchResult([], [f"{code}: eastmoney_stock_notice failed: {exc}"])
    events = []
    for _, row in df.iterrows():
        events.append(
            {
                "公告标题": _text(row.get("公告标题")),
                "公告日期": _text(row.get("公告日期")),
                "source": "东方财富公告大全",
                "source_type": "官方公告",
                "公告类型": _text(row.get("公告类型")),
                "url": _text(row.get("网址")),
                "provider": "eastmoney_stock_notice",
                "code": _text(row.get("代码") or code),
                "name": _text(row.get("名称")),
            }
        )
    return NewsFetchResult([event for event in events if event["公告标题"]], [])


def merge_events(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    merged = []
    for events in groups:
        for event in events:
            key = (
                _text(event.get("code")),
                _text(event.get("title") or event.get("公告标题")),
                _text(event.get("datetime") or event.get("公告日期") or event.get("date")),
                _text(event.get("provider")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(event)
    return sorted(merged, key=_event_time, reverse=True)


def _event_time(event: dict[str, Any]) -> pd.Timestamp:
    value = event.get("datetime") or event.get("发布时间") or event.get("公告日期") or event.get("date")
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp("1900-01-01")
    return pd.Timestamp(ts)


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()
