from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tushare_pro_adapter import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_REQUEST_INTERVAL_SECONDS,
    TushareCacheConfig,
    TushareProAdapter,
    table_path,
    write_cache_manifest,
    write_coverage_outputs,
)
from src.world_model.news_event_table import build_news_event_outputs  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_TOKEN_PATH = ROOT / "tushare_token.txt"
DEFAULT_DATES = [
    "20230830",
    "20230831",
    "20231030",
    "20231031",
    "20240426",
    "20240429",
    "20240430",
    "20240830",
    "20241030",
    "20241031",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Tushare anns_d smoke for weak 2023/2024 news blocks.")
    parser.add_argument("--execute", action="store_true", help="Actually call Tushare. Without it, only writes the plan.")
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES), help="Comma-separated YYYYMMDD dates.")
    parser.add_argument("--max-dates", type=int, default=len(DEFAULT_DATES))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--token-path", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--request-interval-seconds", type=float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
    parser.add_argument("--force", action="store_true", help="Refetch dates even when partition CSV already exists.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    report_dir = REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates = _parse_dates(args.dates)[: max(0, args.max_dates)]
    plan_path = report_dir / "historical_announcement_smoke_plan.md"
    _write_plan(plan_path, dates, args)

    if not args.execute:
        print("A股研究Agent")
        print(f"dry_run=True dates={len(dates)}")
        print(f"plan={plan_path}")
        return

    config = TushareCacheConfig(
        cache_dir=cache_dir,
        token_path=Path(args.token_path),
        request_interval_seconds=args.request_interval_seconds,
    )
    adapter = TushareProAdapter(config)
    rows: list[dict[str, Any]] = []
    try:
        for date in dates:
            partition = f"{date}_{date}"
            out_path = table_path(cache_dir, "anns_d", partition)
            if out_path.exists() and not args.force:
                existing = _row_count(out_path)
                rows.append(_summary_row(date, "skipped_existing", existing, out_path))
                continue
            try:
                frame = adapter.call("anns_d", start_date=date, end_date=date)
                path = adapter.write_table("anns_d", frame, partition=partition)
                rows.append(_summary_row(date, "ok_empty" if frame.empty else "ok", len(frame), path))
            except Exception as exc:
                rows.append(
                    {
                        "date": date,
                        "status": "failed",
                        "rows": 0,
                        "possible_row_cap": False,
                        "output_path": "",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:200],
                    }
                )
    finally:
        adapter.write_records()
        notes = [
            "bounded historical announcement smoke for H2023_2/H2024 weak blocks",
            "token read only from local untracked file or environment; never written to outputs",
            "each anns_d request is one date; default request interval >= 0.7s",
        ]
        write_cache_manifest(cache_dir, records=adapter.records, dry_run=False, notes=notes)
        write_coverage_outputs(cache_dir, report_dir, adapter.records, dry_run=False, notes=notes)

    event_table, feature_table = build_news_event_outputs(cache_dir)
    summary = pd.DataFrame(rows)
    csv_path = report_dir / "historical_announcement_smoke_coverage.csv"
    md_path = report_dir / "historical_announcement_smoke_coverage.md"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(_render_report(summary, event_table, feature_table, plan_path), encoding="utf-8")
    print("A股研究Agent")
    print(f"dates={len(dates)}")
    print(f"requested_records={len(adapter.records)}")
    print(f"derived_events={len(event_table)}")
    print(f"derived_features={len(feature_table)}")
    print(f"report={md_path}")


def _parse_dates(raw: str) -> list[str]:
    dates = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        if len(value) != 8 or not value.isdigit():
            raise ValueError(f"invalid YYYYMMDD date: {value}")
        dates.append(value)
    return list(dict.fromkeys(dates))


def _summary_row(date: str, status: str, rows: int, path: Path) -> dict[str, Any]:
    return {
        "date": date,
        "status": status,
        "rows": int(rows),
        "possible_row_cap": bool(int(rows) >= 6000),
        "output_path": str(path),
        "error_type": "",
        "error_message": "",
    }


def _row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1, 0)
    except OSError:
        return 0


def _write_plan(path: Path, dates: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# Historical Announcement Smoke Plan",
        "",
        "本计划只用于研究辅助，不自动交易，不接券商接口。",
        "",
        f"- execute: `{bool(args.execute)}`",
        f"- dates: `{','.join(dates)}`",
        f"- max_dates: `{args.max_dates}`",
        f"- request_interval_seconds: `{max(DEFAULT_REQUEST_INTERVAL_SECONDS, float(args.request_interval_seconds))}`",
        f"- force: `{bool(args.force)}`",
        "",
        "## Safety",
        "",
        "- 不输出、不记录 token 明文。",
        "- 每个请求只拉一个自然日的 `anns_d`。",
        "- 已存在分片默认跳过。",
        "- 若单日返回接近或达到 6000 行，标记 possible_row_cap，不能视为完整公告覆盖。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_report(summary: pd.DataFrame, events: pd.DataFrame, features: pd.DataFrame, plan_path: Path) -> str:
    lines = [
        "# Historical Announcement Smoke Coverage",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Summary",
        "",
        f"- plan: `{plan_path}`",
        f"- date_rows: `{len(summary)}`",
        f"- ok_or_existing_dates: `{int(summary['status'].isin(['ok', 'ok_empty', 'skipped_existing']).sum()) if not summary.empty else 0}`",
        f"- failed_dates: `{int(summary['status'].eq('failed').sum()) if not summary.empty else 0}`",
        f"- possible_row_cap_dates: `{int(summary['possible_row_cap'].sum()) if 'possible_row_cap' in summary else 0}`",
        f"- derived_tushare_event_rows: `{len(events)}`",
        f"- derived_tushare_feature_rows: `{len(features)}`",
        "",
        "## Date Coverage",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_empty_",
        "",
        "## Boundary",
        "",
        "- 这是小规模历史公告补洞 smoke，不是完整 2023/2024 新闻覆盖。",
        "- possible_row_cap 日期需要更细接口或官方源复核，不能直接宣称全量。",
        "- 下游仍必须通过 `available_at <= decision_time` 过滤。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
