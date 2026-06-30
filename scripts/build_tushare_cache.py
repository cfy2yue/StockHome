from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tushare_pro_adapter import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_REQUEST_INTERVAL_SECONDS,
    SOURCE_NAME,
    SOURCE_TYPE,
    TushareCacheConfig,
    TushareCallRecord,
    TushareProAdapter,
    load_call_records,
    table_path,
    write_cache_manifest,
    write_coverage_outputs,
)
from src.world_model.news_event_table import build_news_event_outputs


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_START_DATE = "20210101"
DEFAULT_END_DATE = "20260625"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a credential-safe offline Tushare Pro cache.")
    parser.add_argument("--execute", action="store_true", help="Actually call Tushare Pro. Without this flag, only writes a dry-run plan/report.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--token-path", default=str(ROOT / "tushare_token.txt"))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--request-interval-seconds", type=float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
    parser.add_argument("--request-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--interfaces",
        nargs="+",
        default=["stock_basic", "trade_cal"],
        choices=[
            "stock_basic",
            "trade_cal",
            "daily_by_trade_date",
            "daily_by_stock",
            "daily_basic_by_trade_date",
            "cyq_perf_by_trade_date",
            "margin_detail_by_trade_date",
            "adj_factor",
            "suspend_d",
            "stk_limit",
            "fina_indicator",
            "forecast",
            "express",
            "income",
            "cashflow",
            "balancesheet",
            "fina_audit",
            "anns_d",
            "news",
            "major_news",
        ],
    )
    parser.add_argument("--max-trade-dates", type=int, default=0, help="Cap trade_date loops. 0 means do not loop trade dates unless explicitly set.")
    parser.add_argument("--max-stocks", type=int, default=0, help="Cap per-stock loops. 0 means do not loop stocks unless explicitly set.")
    parser.add_argument("--stock-offset", type=int, default=0, help="Start per-stock loops from this sorted stock-code offset for resumable shards.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip requests whose target cache file already exists.")
    parser.add_argument("--ann-shard-days", type=int, default=0, help="Split anns_d into inclusive date shards. 0 means one request for start/end.")
    parser.add_argument("--max-ann-shards", type=int, default=0, help="Cap anns_d shard loops. Required when --ann-shard-days > 0.")
    parser.add_argument("--news-srcs", default="sina", help="Comma-separated Tushare news src values. Empty string means no src filter.")
    parser.add_argument("--major-news-srcs", default="新浪财经", help="Comma-separated Tushare major_news src values. Empty string means no src filter.")
    parser.add_argument("--news-start-datetime", default="", help="Optional news start datetime, e.g. 2026-01-05 00:00:00. Defaults to start-date 00:00:00.")
    parser.add_argument("--news-end-datetime", default="", help="Optional news end datetime, e.g. 2026-01-06 23:59:59. Defaults to end-date 23:59:59.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    notes = _plan_notes(args)
    records: list[TushareCallRecord] = []

    if not args.execute:
        _write_plan(cache_dir, args, notes)
        _write_fetch_plan_manifest(cache_dir, args, notes)
        write_coverage_outputs(cache_dir, REPORT_DIR, records, dry_run=True, notes=notes)
        _print_summary(cache_dir, dry_run=True, records=records)
        return

    config = TushareCacheConfig(
        cache_dir=cache_dir,
        token_path=Path(args.token_path),
        request_interval_seconds=args.request_interval_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    adapter = TushareProAdapter(config)
    try:
        _execute_interfaces(adapter, args)
    finally:
        adapter.write_records()
        records = load_call_records(cache_dir)
        write_cache_manifest(cache_dir, records=records, dry_run=False, notes=notes)
        write_coverage_outputs(cache_dir, REPORT_DIR, records, dry_run=False, notes=notes)
    _print_summary(cache_dir, dry_run=False, records=records)


def _execute_interfaces(adapter: TushareProAdapter, args: argparse.Namespace) -> None:
    skip_existing = bool(getattr(args, "skip_existing", False))
    if "stock_basic" in args.interfaces:
        _call_and_write(
            adapter,
            "stock_basic",
            "stock_basic",
            "list_status_L",
            skip_existing=skip_existing,
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs",
        )

    if "trade_cal" in args.interfaces or any(
        name in args.interfaces
        for name in (
            "daily_by_trade_date",
            "daily_basic_by_trade_date",
            "cyq_perf_by_trade_date",
            "margin_detail_by_trade_date",
            "suspend_d",
            "stk_limit",
        )
    ):
        _call_and_write(adapter, "trade_cal", "trade_cal", f"{args.start_date}_{args.end_date}", skip_existing=skip_existing, exchange="", start_date=args.start_date, end_date=args.end_date)

    if "cyq_perf_by_trade_date" in args.interfaces:
        trade_dates = _cached_trade_dates(adapter.config.cache_dir, args.start_date, args.end_date)
        if args.max_trade_dates <= 0:
            raise RuntimeError("cyq_perf_by_trade_date requires --max-trade-dates to avoid accidental large pulls")
        for trade_date in trade_dates[: args.max_trade_dates]:
            _call_and_write(adapter, "cyq_perf", "cyq_perf", f"trade_date_{trade_date}", skip_existing=skip_existing, trade_date=trade_date)

    if "margin_detail_by_trade_date" in args.interfaces:
        trade_dates = _cached_trade_dates(adapter.config.cache_dir, args.start_date, args.end_date)
        if args.max_trade_dates <= 0:
            raise RuntimeError("margin_detail_by_trade_date requires --max-trade-dates to avoid accidental large pulls")
        for trade_date in trade_dates[: args.max_trade_dates]:
            _call_and_write(adapter, "margin_detail", "margin_detail", f"trade_date_{trade_date}", skip_existing=skip_existing, trade_date=trade_date)

    if "daily_by_trade_date" in args.interfaces:
        trade_dates = _cached_trade_dates(adapter.config.cache_dir, args.start_date, args.end_date)
        if args.max_trade_dates <= 0:
            raise RuntimeError("daily_by_trade_date requires --max-trade-dates to avoid accidental large pulls")
        for trade_date in trade_dates[: args.max_trade_dates]:
            _call_and_write(adapter, "daily", "daily", f"trade_date_{trade_date}", skip_existing=skip_existing, trade_date=trade_date)

    if "daily_basic_by_trade_date" in args.interfaces:
        trade_dates = _cached_trade_dates(adapter.config.cache_dir, args.start_date, args.end_date)
        if args.max_trade_dates <= 0:
            raise RuntimeError("daily_basic_by_trade_date requires --max-trade-dates to avoid accidental large pulls")
        for trade_date in trade_dates[: args.max_trade_dates]:
            _call_and_write(adapter, "daily_basic", "daily_basic", f"trade_date_{trade_date}", skip_existing=skip_existing, trade_date=trade_date)

    if "suspend_d" in args.interfaces:
        trade_dates = _cached_trade_dates(adapter.config.cache_dir, args.start_date, args.end_date)
        if args.max_trade_dates <= 0:
            raise RuntimeError("suspend_d requires --max-trade-dates to avoid accidental large pulls")
        for trade_date in trade_dates[: args.max_trade_dates]:
            _call_and_write(adapter, "suspend_d", "suspend_d", f"trade_date_{trade_date}", skip_existing=skip_existing, trade_date=trade_date)

    if "stk_limit" in args.interfaces:
        trade_dates = _cached_trade_dates(adapter.config.cache_dir, args.start_date, args.end_date)
        if args.max_trade_dates <= 0:
            raise RuntimeError("stk_limit requires --max-trade-dates to avoid accidental large pulls")
        for trade_date in trade_dates[: args.max_trade_dates]:
            _call_and_write(adapter, "stk_limit", "stk_limit", f"trade_date_{trade_date}", skip_existing=skip_existing, trade_date=trade_date)

    stock_codes = _stock_slice(_cached_stock_codes(adapter.config.cache_dir), args)
    if "daily_by_stock" in args.interfaces:
        if args.max_stocks <= 0:
            raise RuntimeError("daily_by_stock requires --max-stocks to avoid accidental large pulls")
        for ts_code in stock_codes[: args.max_stocks]:
            _call_and_write(adapter, "daily", "daily_by_stock", ts_code, skip_existing=skip_existing, ts_code=ts_code, start_date=args.start_date, end_date=args.end_date)

    if "adj_factor" in args.interfaces:
        if args.max_stocks <= 0:
            raise RuntimeError("adj_factor requires --max-stocks to avoid accidental large pulls")
        for ts_code in stock_codes[: args.max_stocks]:
            _call_and_write(adapter, "adj_factor", "adj_factor", ts_code, skip_existing=skip_existing, ts_code=ts_code, start_date=args.start_date, end_date=args.end_date)

    if "fina_indicator" in args.interfaces:
        if args.max_stocks <= 0:
            raise RuntimeError("fina_indicator requires --max-stocks to avoid accidental large pulls")
        for ts_code in stock_codes[: args.max_stocks]:
            _call_and_write(adapter, "fina_indicator", "fina_indicator", ts_code, skip_existing=skip_existing, ts_code=ts_code, start_date=args.start_date, end_date=args.end_date)
        _derive_financial_disclosure_calendar(adapter.config.cache_dir)

    stock_financial_interfaces = {
        "forecast": "forecast",
        "express": "express",
        "income": "income",
        "cashflow": "cashflow",
        "balancesheet": "balancesheet",
        "fina_audit": "fina_audit",
    }
    for interface, table_name in stock_financial_interfaces.items():
        if interface not in args.interfaces:
            continue
        if args.max_stocks <= 0:
            raise RuntimeError(f"{interface} requires --max-stocks to avoid accidental large pulls")
        for ts_code in stock_codes[: args.max_stocks]:
            _call_and_write(adapter, interface, table_name, ts_code, skip_existing=skip_existing, ts_code=ts_code, start_date=args.start_date, end_date=args.end_date)

    if "anns_d" in args.interfaces:
        if args.ann_shard_days > 0:
            if args.max_ann_shards <= 0:
                raise RuntimeError("anns_d sharded pulls require --max-ann-shards to avoid accidental large pulls")
            for start_date, end_date in _date_shards(args.start_date, args.end_date, args.ann_shard_days)[: args.max_ann_shards]:
                _call_and_write(
                    adapter,
                    "anns_d",
                    "anns_d",
                    f"{start_date}_{end_date}",
                    skip_existing=skip_existing,
                    start_date=start_date,
                    end_date=end_date,
                )
        else:
            _call_and_write(
                adapter,
                "anns_d",
                "anns_d",
                f"{args.start_date}_{args.end_date}",
                skip_existing=skip_existing,
                start_date=args.start_date,
                end_date=args.end_date,
            )

    if "news" in args.interfaces:
        for src in _split_sources(args.news_srcs):
            partition = f"{_safe_partition(src or 'all')}_{args.start_date}_{args.end_date}"
            _call_and_write(
                adapter,
                "news",
                "news",
                partition,
                skip_existing=skip_existing,
                src=src,
                start_date=_news_start(args),
                end_date=_news_end(args),
            )

    if "major_news" in args.interfaces:
        for src in _split_sources(args.major_news_srcs):
            partition = f"{_safe_partition(src or 'all')}_{args.start_date}_{args.end_date}"
            _call_and_write(
                adapter,
                "major_news",
                "major_news",
                partition,
                skip_existing=skip_existing,
                src=src,
                start_date=_news_start(args),
                end_date=_news_end(args),
            )

    if any(name in args.interfaces for name in ["anns_d", "news", "major_news"]):
        build_news_event_outputs(adapter.config.cache_dir)


def _call_and_write(adapter: TushareProAdapter, interface: str, table_name: str, partition: str, *, skip_existing: bool = False, **params: Any) -> bool:
    if skip_existing and table_path(adapter.config.cache_dir, table_name, partition).exists():
        return True
    try:
        frame = adapter.call(interface, **params)
    except Exception:
        return False
    adapter.write_table(table_name, frame, partition=partition)
    return True


def _stock_slice(stock_codes: list[str], args: argparse.Namespace) -> list[str]:
    offset = max(0, int(getattr(args, "stock_offset", 0) or 0))
    limit = int(getattr(args, "max_stocks", 0) or 0)
    if limit <= 0:
        return stock_codes[offset:]
    return stock_codes[offset : offset + limit]


def _cached_trade_dates(cache_dir: Path, start_date: str, end_date: str) -> list[str]:
    paths = sorted((Path(cache_dir) / "tables" / "trade_cal").glob("*.csv"))
    if not paths:
        return []
    frame = pd.concat([pd.read_csv(path, dtype=str) for path in paths], ignore_index=True)
    if "cal_date" not in frame:
        return []
    if "is_open" in frame:
        frame = frame[frame["is_open"].astype(str).eq("1")]
    dates = frame["cal_date"].astype(str)
    return sorted(date for date in dates.unique().tolist() if start_date <= date <= end_date)


def _cached_stock_codes(cache_dir: Path) -> list[str]:
    paths = sorted((Path(cache_dir) / "tables" / "stock_basic").glob("*.csv"))
    if not paths:
        return []
    frame = pd.concat([pd.read_csv(path, dtype=str) for path in paths], ignore_index=True)
    if "ts_code" not in frame:
        return []
    return sorted(frame["ts_code"].dropna().astype(str).unique().tolist())


def _derive_financial_disclosure_calendar(cache_dir: Path) -> Path:
    rows = []
    for path in sorted((Path(cache_dir) / "tables" / "fina_indicator").glob("*.csv")):
        frame = pd.read_csv(path, dtype=str)
        for _, row in frame.iterrows():
            ts_code = str(row.get("ts_code", "")).strip()
            report_period = str(row.get("end_date", "")).strip()
            disclosure_date = str(row.get("ann_date", "")).strip()
            if ts_code and report_period and disclosure_date:
                rows.append(
                    {
                        "ts_code": ts_code,
                        "report_period": report_period,
                        "disclosure_date": disclosure_date,
                        "available_at": f"{disclosure_date} 15:00:00",
                        "source_type": SOURCE_TYPE,
                        "source_name": SOURCE_NAME,
                        "interface": "fina_indicator",
                    }
                )
    out = Path(cache_dir) / "derived" / "financial_disclosure_calendar.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["ts_code", "report_period", "disclosure_date", "available_at", "source_type", "source_name", "interface"]).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def _split_sources(raw: str) -> list[str]:
    if raw == "":
        return [""]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _news_start(args: argparse.Namespace) -> str:
    if args.news_start_datetime:
        return args.news_start_datetime
    return f"{args.start_date[:4]}-{args.start_date[4:6]}-{args.start_date[6:8]} 00:00:00"


def _news_end(args: argparse.Namespace) -> str:
    if args.news_end_datetime:
        return args.news_end_datetime
    return f"{args.end_date[:4]}-{args.end_date[4:6]}-{args.end_date[6:8]} 23:59:59"


def _date_shards(start_date: str, end_date: str, shard_days: int) -> list[tuple[str, str]]:
    if shard_days <= 0:
        return [(start_date, end_date)]
    start = datetime.strptime(start_date, "%Y%m%d").date()
    end = datetime.strptime(end_date, "%Y%m%d").date()
    if end < start:
        raise ValueError("end_date must be >= start_date")
    shards: list[tuple[str, str]] = []
    current = start
    while current <= end:
        shard_end = min(current + timedelta(days=shard_days - 1), end)
        shards.append((current.strftime("%Y%m%d"), shard_end.strftime("%Y%m%d")))
        current = shard_end + timedelta(days=1)
    return shards


def _safe_partition(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _write_plan(cache_dir: Path, args: argparse.Namespace, notes: list[str]) -> Path:
    path = cache_dir / "fetch_plan.md"
    lines = [
        "# Tushare Pro Fetch Plan",
        "",
        "本计划不会读取或输出 token。实际执行必须显式加 `--execute`。",
        "",
        f"- source_type: `{SOURCE_TYPE}`",
        f"- source_name: `{SOURCE_NAME}`",
        f"- start_date: `{args.start_date}`",
        f"- end_date: `{args.end_date}`",
        f"- interfaces: `{','.join(args.interfaces)}`",
        f"- request_interval_seconds: `{max(DEFAULT_REQUEST_INTERVAL_SECONDS, float(args.request_interval_seconds))}`",
        f"- request_timeout_seconds: `{max(5.0, float(args.request_timeout_seconds))}`",
        f"- max_trade_dates: `{args.max_trade_dates}`",
        f"- max_stocks: `{args.max_stocks}`",
        f"- stock_offset: `{args.stock_offset}`",
        f"- skip_existing: `{bool(args.skip_existing)}`",
        f"- ann_shard_days: `{args.ann_shard_days}`",
        f"- max_ann_shards: `{args.max_ann_shards}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_fetch_plan_manifest(cache_dir: Path, args: argparse.Namespace, notes: list[str]) -> Path:
    table_files = sorted(path.relative_to(cache_dir).as_posix() for path in (cache_dir / "tables").glob("**/*.csv")) if (cache_dir / "tables").exists() else []
    manifest = {
        "source_type": SOURCE_TYPE,
        "source_name": SOURCE_NAME,
        "dry_run": True,
        "purpose": "fetch_plan_only_no_api_call",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "credential_policy": "token not read for dry-run fetch plans; token never written to manifest/log/report/cache metadata",
        "planned_interfaces": list(args.interfaces),
        "planned_start_date": args.start_date,
        "planned_end_date": args.end_date,
        "planned_max_stocks": int(args.max_stocks),
        "planned_stock_offset": int(args.stock_offset),
        "planned_skip_existing": bool(args.skip_existing),
        "planned_request_timeout_seconds": max(5.0, float(args.request_timeout_seconds)),
        "planned_max_trade_dates": int(args.max_trade_dates),
        "planned_ann_shard_days": int(args.ann_shard_days),
        "planned_max_ann_shards": int(args.max_ann_shards),
        "existing_table_files_count": len(table_files),
        "notes": notes,
    }
    path = cache_dir / "fetch_plan_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _plan_notes(args: argparse.Namespace) -> list[str]:
    notes = [
        "回测和 DeepSeek 决策点只读本地缓存，不在线请求。",
        "token/key 不写入代码、日志、报告、prompt、ledger、缓存元数据或 Git。",
        "默认请求间隔不低于 0.7 秒，遵守 100 次/分钟限制。",
    ]
    if not args.execute:
        notes.append("当前为 dry-run，未调用 Tushare Pro。")
    trade_date_interfaces = {
        "daily_by_trade_date",
        "daily_basic_by_trade_date",
        "cyq_perf_by_trade_date",
        "margin_detail_by_trade_date",
        "suspend_d",
        "stk_limit",
    }
    if any(name in args.interfaces for name in trade_date_interfaces) and args.max_trade_dates <= 0:
        notes.append("按交易日循环的接口未设置 max_trade_dates，真实执行会拒绝大循环。")
    stock_loop_interfaces = {
        "daily_by_stock",
        "adj_factor",
        "fina_indicator",
        "forecast",
        "express",
        "income",
        "cashflow",
        "balancesheet",
        "fina_audit",
    }
    if any(name in args.interfaces for name in stock_loop_interfaces) and args.max_stocks <= 0:
        notes.append("逐股票接口未设置 max_stocks，真实执行会拒绝大循环。")
    if any(name in args.interfaces for name in stock_loop_interfaces):
        notes.append("逐股票接口支持 stock_offset 分片和 skip_existing 断点续跑，避免重复请求已缓存分区。")
    if "anns_d" in args.interfaces and args.ann_shard_days > 0 and args.max_ann_shards <= 0:
        notes.append("公告分片接口设置了 ann_shard_days 但未设置 max_ann_shards，真实执行会拒绝大循环。")
    if any(name in args.interfaces for name in ["anns_d", "news", "major_news"]):
        notes.append("新闻/公告接口可能需要单独权限；失败时记录 coverage，不中断整个流程。")
        notes.append("新闻/公告派生表必须使用 available_at <= decision_time 的记录。")
    return notes


def _print_summary(cache_dir: Path, *, dry_run: bool, records: list[TushareCallRecord]) -> None:
    print("A股研究Agent")
    print(f"tushare cache dir: {cache_dir}")
    print(f"dry_run: {dry_run}")
    print(f"records: {len(records)}")
    print(f"report: {REPORT_DIR / 'tushare_data_coverage.md'}")


if __name__ == "__main__":
    main()
