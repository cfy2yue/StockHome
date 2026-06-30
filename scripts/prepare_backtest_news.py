from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.io import load_universe, read_yaml
from src.backtest.news_sources import fetch_eastmoney_stock_news, fetch_eastmoney_stock_notices, merge_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch free public news/announcement events for backtest cache.")
    parser.add_argument("--universe", default="config/backtest_scale_200_universe.yaml")
    parser.add_argument("--data-dir", default="data/backtest_scale_200")
    parser.add_argument("--sources", default="eastmoney_news", help="Comma list: eastmoney_news,eastmoney_notice")
    parser.add_argument("--notice-start", default="2026-01-01")
    parser.add_argument("--notice-end", default="2026-06-23")
    parser.add_argument("--max-stocks", type=int, default=0, help="0 means all stocks in universe.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based inclusive stock index in the universe.")
    parser.add_argument("--end-index", type=int, default=0, help="1-based inclusive stock index; 0 means no explicit end.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--skip-existing", action="store_true", help="Skip only when requested providers already exist in news.json.")
    args = parser.parse_args()

    universe = load_universe(args.universe)
    stocks = universe["train"] + universe["test"]
    start = max(1, args.start_index)
    end = args.end_index if args.end_index > 0 else len(stocks)
    stocks = stocks[start - 1 : end]
    if args.max_stocks > 0:
        stocks = stocks[: args.max_stocks]
    sources = {item.strip() for item in args.sources.split(",") if item.strip()}
    data_dir = Path(args.data_dir)
    errors: list[str] = []
    written = 0
    skipped = 0

    for idx, stock in enumerate(stocks, 1):
        code = str(stock["code"]).zfill(6)
        stock_dir = data_dir / code
        stock_dir.mkdir(parents=True, exist_ok=True)
        news_path = stock_dir / "news.json"
        existing = _read_existing(news_path)
        if args.skip_existing and _has_requested_sources(existing, sources):
            skipped += 1
            continue
        groups = [existing]
        if "eastmoney_news" in sources:
            result = fetch_eastmoney_stock_news(code)
            groups.append(result.events)
            errors.extend(result.errors)
        if "eastmoney_notice" in sources:
            result = fetch_eastmoney_stock_notices(code, args.notice_start, args.notice_end)
            groups.append(result.events)
            errors.extend(result.errors)
        events = merge_events(*groups)
        payload = {
            "meta": {
                "code": code,
                "name": stock.get("name", ""),
                "sources": sorted(sources),
                "providers_in_events": sorted(_providers(events)),
                "notice_start": args.notice_start if "eastmoney_notice" in sources else "",
                "notice_end": args.notice_end if "eastmoney_notice" in sources else "",
                "note": "免费公开源缓存；回测时仍按决策日时间窗过滤。",
            },
            "events": events,
        }
        news_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        written += 1
        absolute_idx = start + idx - 1
        print(f"- [{absolute_idx}/{start + len(stocks) - 1}] {code} news_events={len(events)}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    report_path = data_dir / "news_fetch_report.md"
    report_path.write_text(_report(written, skipped, errors, sources, start, start + len(stocks) - 1), encoding="utf-8")
    print("A股研究Agent")
    print(f"新闻缓存完成：{written} 支，报告：{report_path}")


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = read_yaml(path)
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        return data["events"]
    return data if isinstance(data, list) else []


def _has_requested_sources(events: list[dict], sources: set[str]) -> bool:
    providers = _providers(events)
    expected = set()
    if "eastmoney_news" in sources:
        expected.add("eastmoney_stock_news")
    if "eastmoney_notice" in sources:
        expected.add("eastmoney_stock_notice")
    return bool(expected) and expected.issubset(providers)


def _providers(events: list[dict]) -> set[str]:
    return {str(event.get("provider", "")).strip() for event in events if event.get("provider")}


def _report(written: int, skipped: int, errors: list[str], sources: set[str], start: int, end: int) -> str:
    lines = [
        "# 新闻缓存采集报告",
        "",
        f"- 请求来源：{', '.join(sorted(sources))}",
        f"- 股票序号范围：{start}-{end}",
        f"- 写入股票数：{written}",
        f"- 跳过股票数：{skipped}",
        f"- 失败记录数：{len(errors)}",
        "- 边界：只使用免费公开源；回测时按决策日时间窗过滤；失败源跳过并记录。",
    ]
    if errors:
        lines += ["", "## 失败记录", ""]
        lines.extend(f"- {item}" for item in errors)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
