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


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_TOKEN_PATH = ROOT / "tushare_token.txt"
CAP_ROW_THRESHOLD = 6000


TITLE_BUCKETS = {
    "financial_report": ["年报", "年度报告", "半年报", "半年度报告", "季报", "季度报告", "财务报表"],
    "forecast_or_express": ["业绩预告", "业绩快报", "业绩修正", "预增", "预减", "扭亏", "首亏", "续亏"],
    "audit_or_inquiry": ["审计", "非标", "问询", "监管函", "关注函", "回复"],
    "risk_warning": ["风险", "警示", "退市", "立案", "处罚", "诉讼", "冻结", "违规"],
    "shareholder_action": ["减持", "增持", "回购", "质押", "解除质押"],
    "financing_or_mna": ["重组", "收购", "定增", "发行", "可转债", "融资"],
    "investor_relation": ["投资者关系", "调研", "路演", "业绩说明会"],
    "routine_governance": ["董事会", "监事会", "股东大会", "独立董事", "章程"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached Tushare anns_d row-cap risk and optional ts_code split viability.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--row-cap-threshold", type=int, default=CAP_ROW_THRESHOLD)
    parser.add_argument("--execute-probe", action="store_true", help="Call Tushare for a tiny ts_code split probe. Default is local audit only.")
    parser.add_argument("--probe-date", default="", help="YYYYMMDD row-cap date to probe. Defaults to the first capped date.")
    parser.add_argument("--probe-ts-codes", default="", help="Comma-separated ts_codes to probe. Defaults to top cached stocks on probe date.")
    parser.add_argument("--max-probe-codes", type=int, default=3)
    parser.add_argument("--token-path", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--request-interval-seconds", type=float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = build_row_cap_audit(cache_dir, row_cap_threshold=args.row_cap_threshold)
    detail = build_row_cap_detail(cache_dir, audit)

    audit_csv = report_dir / "announcement_row_cap_audit.csv"
    detail_csv = report_dir / "announcement_row_cap_detail.csv"
    report_md = report_dir / "announcement_row_cap_audit.md"
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    probe = pd.DataFrame()
    probe_path = report_dir / "announcement_row_cap_ts_code_probe.csv"
    if args.execute_probe:
        probe = run_ts_code_probe(cache_dir, audit, args)
        probe.to_csv(probe_path, index=False, encoding="utf-8-sig")
    elif probe_path.exists():
        probe = pd.read_csv(probe_path, dtype=str)

    report_md.write_text(render_report(audit, detail, probe, args, audit_csv, detail_csv, probe_path), encoding="utf-8")
    print("A股研究Agent")
    print(f"audit_rows={len(audit)}")
    print(f"cap_dates={int(audit['possible_row_cap'].sum()) if not audit.empty else 0}")
    print(f"detail_rows={len(detail)}")
    print(f"probe_rows={len(probe)}")
    print(f"report={report_md}")


def build_row_cap_audit(cache_dir: Path, *, row_cap_threshold: int = CAP_ROW_THRESHOLD) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((cache_dir / "tables" / "anns_d").glob("*.csv")) if (cache_dir / "tables" / "anns_d").exists() else []:
        frame = _read_anns(path)
        ann_date = _date_from_partition(path)
        title = frame.get("title", pd.Series(dtype=str)).fillna("").astype(str) if not frame.empty else pd.Series(dtype=str)
        ts_code = frame.get("ts_code", pd.Series(dtype=str)).fillna("").astype(str) if not frame.empty else pd.Series(dtype=str)
        rows.append(
            {
                "partition": path.stem,
                "ann_date": ann_date,
                "rows": int(len(frame)),
                "possible_row_cap": bool(len(frame) >= row_cap_threshold),
                "unique_stocks": int(ts_code.nunique()) if not ts_code.empty else 0,
                "top_stock_share": _top_share(ts_code),
                "sh_count": int(ts_code.str.endswith(".SH").sum()) if not ts_code.empty else 0,
                "sz_count": int(ts_code.str.endswith(".SZ").sum()) if not ts_code.empty else 0,
                "bj_count": int(ts_code.str.endswith(".BJ").sum()) if not ts_code.empty else 0,
                "financial_report_rows": _bucket_count(title, "financial_report"),
                "forecast_or_express_rows": _bucket_count(title, "forecast_or_express"),
                "audit_or_inquiry_rows": _bucket_count(title, "audit_or_inquiry"),
                "risk_warning_rows": _bucket_count(title, "risk_warning"),
                "investor_relation_rows": _bucket_count(title, "investor_relation"),
                "routine_governance_rows": _bucket_count(title, "routine_governance"),
                "output_path": str(path),
            }
        )
    columns = [
        "partition",
        "ann_date",
        "rows",
        "possible_row_cap",
        "unique_stocks",
        "top_stock_share",
        "sh_count",
        "sz_count",
        "bj_count",
        "financial_report_rows",
        "forecast_or_express_rows",
        "audit_or_inquiry_rows",
        "risk_warning_rows",
        "investor_relation_rows",
        "routine_governance_rows",
        "output_path",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["ann_date", "partition"]).reset_index(drop=True)


def build_row_cap_detail(cache_dir: Path, audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if audit.empty:
        return pd.DataFrame(columns=["ann_date", "detail_type", "bucket", "rows", "share", "example"])
    cap_partitions = audit[audit["possible_row_cap"].astype(bool)]["partition"].astype(str).tolist()
    for partition in cap_partitions:
        path = cache_dir / "tables" / "anns_d" / f"{partition}.csv"
        frame = _read_anns(path)
        if frame.empty:
            continue
        ann_date = _date_from_partition(path)
        ts_code = frame.get("ts_code", pd.Series(dtype=str)).fillna("").astype(str)
        title = frame.get("title", pd.Series(dtype=str)).fillna("").astype(str)
        for suffix, group in ts_code.str.extract(r"(\.[A-Z]+)$", expand=False).fillna(".UNKNOWN").value_counts().items():
            rows.append(_detail_row(ann_date, "exchange_suffix", suffix, int(group), len(frame), ""))
        for prefix, group in ts_code.str.slice(0, 3).value_counts().head(12).items():
            rows.append(_detail_row(ann_date, "code_prefix3", prefix, int(group), len(frame), ""))
        for bucket in TITLE_BUCKETS:
            mask = _bucket_mask(title, bucket)
            if mask.any():
                example = str(title[mask].iloc[0])[:80]
                rows.append(_detail_row(ann_date, "title_bucket", bucket, int(mask.sum()), len(frame), example))
        top_stocks = ts_code.value_counts().head(8)
        names = frame.get("name", pd.Series(dtype=str)).fillna("").astype(str)
        for code, count in top_stocks.items():
            first_name = names[ts_code.eq(code)].iloc[0] if ts_code.eq(code).any() else ""
            rows.append(_detail_row(ann_date, "top_stock", code, int(count), len(frame), str(first_name)[:40]))
    return pd.DataFrame(rows).sort_values(["ann_date", "detail_type", "rows"], ascending=[True, True, False]).reset_index(drop=True)


def run_ts_code_probe(cache_dir: Path, audit: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(columns=_probe_columns())
    cap_dates = audit[audit["possible_row_cap"].astype(bool)]["ann_date"].astype(str).tolist()
    probe_date = args.probe_date.strip() or (cap_dates[0] if cap_dates else str(audit.iloc[0]["ann_date"]))
    if not probe_date or len(probe_date) != 8 or not probe_date.isdigit():
        raise ValueError(f"invalid probe date: {probe_date}")
    ts_codes = [item.strip() for item in args.probe_ts_codes.split(",") if item.strip()]
    if not ts_codes:
        partition = next((str(row["partition"]) for _, row in audit.iterrows() if str(row["ann_date"]) == probe_date), "")
        frame = _read_anns(cache_dir / "tables" / "anns_d" / f"{partition}.csv") if partition else pd.DataFrame()
        if "ts_code" in frame:
            ts_codes = frame["ts_code"].dropna().astype(str).value_counts().head(max(1, args.max_probe_codes)).index.tolist()
    ts_codes = ts_codes[: max(1, args.max_probe_codes)]
    config = TushareCacheConfig(
        cache_dir=cache_dir,
        token_path=Path(args.token_path),
        request_interval_seconds=args.request_interval_seconds,
    )
    adapter = TushareProAdapter(config)
    rows: list[dict[str, Any]] = []
    try:
        for ts_code in ts_codes:
            try:
                frame = adapter.call("anns_d", ts_code=ts_code, start_date=probe_date, end_date=probe_date)
                out_path = adapter.write_table("anns_d_probe_ts_code", frame, partition=f"{probe_date}_{ts_code}")
                matched_ratio = _matched_ratio(frame, ts_code)
                rows.append(
                    {
                        "probe_date": probe_date,
                        "ts_code": ts_code,
                        "status": "ok_empty" if frame.empty else "ok",
                        "rows": int(len(frame)),
                        "all_rows_match_ts_code": bool(matched_ratio == 1.0 or frame.empty),
                        "matched_ts_code_ratio": matched_ratio,
                        "output_path": str(out_path),
                        "error_type": "",
                        "error_message": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "probe_date": probe_date,
                        "ts_code": ts_code,
                        "status": "failed",
                        "rows": 0,
                        "all_rows_match_ts_code": False,
                        "matched_ts_code_ratio": 0.0,
                        "output_path": "",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:200],
                    }
                )
    finally:
        adapter.write_records()
        notes = [
            "tiny anns_d ts_code split probe for row-cap audit",
            "token read only from local untracked file or environment; never written to outputs",
        ]
        write_cache_manifest(cache_dir, records=adapter.records, dry_run=False, notes=notes)
        write_coverage_outputs(cache_dir, REPORT_DIR, adapter.records, dry_run=False, notes=notes)
    return pd.DataFrame(rows, columns=_probe_columns())


def render_report(
    audit: pd.DataFrame,
    detail: pd.DataFrame,
    probe: pd.DataFrame,
    args: argparse.Namespace,
    audit_csv: Path,
    detail_csv: Path,
    probe_csv: Path,
) -> str:
    cap_dates = int(audit["possible_row_cap"].sum()) if not audit.empty else 0
    total_dates = len(audit)
    cap_ratio = cap_dates / total_dates if total_dates else 0.0
    lines = [
        "# Announcement Row-Cap Coverage Audit",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Summary",
        "",
        f"- cached_anns_d_partitions: `{total_dates}`",
        f"- possible_row_cap_partitions: `{cap_dates}`",
        f"- possible_row_cap_ratio: `{cap_ratio:.4f}`",
        f"- execute_probe: `{bool(args.execute_probe)}`",
        f"- audit_csv: `{audit_csv}`",
        f"- detail_csv: `{detail_csv}`",
        f"- probe_csv: `{probe_csv}`",
        "",
        "## Interpretation",
        "",
        "- 单日返回达到 row-cap threshold 时，只能说明该日期至少有这些公告，不能视为全量公告覆盖。",
        "- 当前缓存适合做 time-safe evidence smoke，但不适合在 row-cap 日期上宣称完整历史新闻/公告覆盖。",
        "- 若 `ts_code` probe 成功，下一步应只对目标 universe 或抽样复核做按股补洞；全市场逐股补洞请求量太大，不适合直接执行。",
        "- 若 `ts_code` probe 失败，应优先用官方披露源或其他标准化接口做 row-cap 日期抽样核对。",
        "",
        "## Capped Partitions",
        "",
        _table(audit[audit["possible_row_cap"].astype(bool)].head(30)),
        "",
        "## Detail Highlights",
        "",
        _table(detail.head(80)),
        "",
    ]
    if not probe.empty:
        lines.extend(["## Ts Code Split Probe", "", _table(probe), ""])
    else:
        lines.extend(
            [
                "## Ts Code Split Probe",
                "",
                "- 未执行真实 probe；如需验证 `ts_code` 分片可行性，使用 `--execute-probe --max-probe-codes 3`。",
                "",
            ]
        )
    lines.extend(
        [
            "## Next Action",
            "",
            "1. 不直接扩大 DeepSeek round；先解决 row-cap 和样本均衡。",
            "2. 对 row-cap 日期只做目标 universe 或小样本按股复核，避免全市场逐股请求。",
            "3. 新闻问卷继续把公告缺失、row-cap、官方来源但低信号写成不确定性/反证，而不是正向 alpha。",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_anns(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "__empty__" in frame.columns and len(frame.columns) == 1:
        return pd.DataFrame()
    return frame


def _date_from_partition(path: Path) -> str:
    name = path.stem
    first = name.split("_")[0]
    return first if len(first) == 8 and first.isdigit() else ""


def _top_share(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    counts = values.value_counts()
    if counts.empty:
        return 0.0
    return round(float(counts.iloc[0] / len(values)), 4)


def _bucket_mask(titles: pd.Series, bucket: str) -> pd.Series:
    keywords = TITLE_BUCKETS[bucket]
    if titles.empty:
        return pd.Series(dtype=bool)
    pattern = "|".join(keywords)
    return titles.str.contains(pattern, regex=True, na=False)


def _bucket_count(titles: pd.Series, bucket: str) -> int:
    return int(_bucket_mask(titles, bucket).sum()) if not titles.empty else 0


def _detail_row(ann_date: str, detail_type: str, bucket: str, rows: int, total_rows: int, example: str) -> dict[str, Any]:
    share = rows / total_rows if total_rows else 0.0
    return {
        "ann_date": ann_date,
        "detail_type": detail_type,
        "bucket": bucket,
        "rows": rows,
        "share": round(float(share), 4),
        "example": example,
    }


def _matched_ratio(frame: pd.DataFrame, ts_code: str) -> float:
    if frame.empty or "ts_code" not in frame:
        return 1.0 if frame.empty else 0.0
    values = frame["ts_code"].dropna().astype(str)
    if values.empty:
        return 0.0
    return round(float(values.eq(ts_code).mean()), 4)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def _probe_columns() -> list[str]:
    return [
        "probe_date",
        "ts_code",
        "status",
        "rows",
        "all_rows_match_ts_code",
        "matched_ts_code_ratio",
        "output_path",
        "error_type",
        "error_message",
    ]


if __name__ == "__main__":
    main()
