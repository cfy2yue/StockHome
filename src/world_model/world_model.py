from __future__ import annotations

from .announcement_collector import collect_announcements
from .market_regime import collect_market_regime
from .news_collector import collect_stock_news
from .sector_events import collect_sector_events


def build_world_model(code: str, name: str, dry_run: bool = False) -> dict:
    news = collect_stock_news(code, name, dry_run)
    ann = collect_announcements(code, name, dry_run)
    market = collect_market_regime(dry_run)
    sector = collect_sector_events(None, dry_run)
    risks = []
    if not news.ok:
        risks.append("新闻模块当前公开接口获取失败，本轮新闻信息不足")
    if not ann.ok:
        risks.append("公告模块获取失败，重大公告需人工核验")
    if not market.ok:
        risks.append("市场环境数据不足")
    return {
        "score": 4 if risks else 6,
        "news": news,
        "announcements": ann,
        "market": market,
        "sector": sector,
        "risks": risks,
        "summary": "最新信息层需要人工复核" if risks else "最新信息层已获取基础样例",
    }
