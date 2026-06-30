"""Inventory local financial/announcement sources before backfilling.

This script is local-only: it never reads tokens and never calls Tushare. It
summarizes cached financial/announcement tables and highlights coverage gaps
that must be fixed before another DS promotion round.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402
from src.data.tushare_pro_adapter import DEFAULT_CACHE_DIR  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_OUTPUT_PREFIX = "financial_source_inventory_v1"

TARGET_TABLES: dict[str, dict[str, Any]] = {
    "fina_indicator": {
        "priority": "P0",
        "role": "structured financial metrics with disclosure dates",
        "date_candidates": ["f_ann_date", "ann_date"],
        "period_candidates": ["end_date"],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 1000,
    },
    "forecast": {
        "priority": "P0",
        "role": "performance forecast / revision signal",
        "date_candidates": ["ann_date", "f_ann_date"],
        "period_candidates": ["end_date"],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 500,
    },
    "express": {
        "priority": "P0",
        "role": "performance express pre-report signal",
        "date_candidates": ["ann_date", "f_ann_date"],
        "period_candidates": ["end_date"],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 500,
    },
    "anns_d": {
        "priority": "P0",
        "role": "official/standardized announcement titles and URLs",
        "date_candidates": ["f_ann_date", "ann_date"],
        "period_candidates": [],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 1000,
        "row_cap_threshold": 6000,
    },
    "income": {
        "priority": "P1",
        "role": "income statement metrics for richer financial variance tables",
        "date_candidates": ["f_ann_date", "ann_date"],
        "period_candidates": ["end_date"],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 500,
    },
    "cashflow": {
        "priority": "P1",
        "role": "cash-flow quality and accrual risk metrics",
        "date_candidates": ["f_ann_date", "ann_date"],
        "period_candidates": ["end_date"],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 500,
    },
    "balancesheet": {
        "priority": "P1",
        "role": "balance-sheet leverage and asset-quality metrics",
        "date_candidates": ["f_ann_date", "ann_date"],
        "period_candidates": ["end_date"],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 500,
    },
    "fina_audit": {
        "priority": "P1",
        "role": "audit opinion / non-standard audit risk",
        "date_candidates": ["ann_date", "f_ann_date"],
        "period_candidates": ["end_date"],
        "stock_candidates": ["ts_code"],
        "min_unique_stocks": 200,
    },
}

FINANCIAL_KEYWORDS = [
    "年度报告",
    "半年度报告",
    "季度报告",
    "业绩预告",
    "业绩快报",
    "业绩修正",
    "审计",
    "问询",
    "更正",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local financial source inventory without API calls.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tables, coverage, gap_plan = build_inventory(args.cache_dir)
    paths = write_outputs(args.output_prefix, tables, coverage, gap_plan)

    print("A股研究Agent")
    print(f"tables={len(tables)}")
    print(f"coverage_rows={len(coverage)}")
    print(f"gap_items={len(gap_plan)}")
    print(f"report={paths['report']}")


def build_inventory(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = summarize_tables(cache_dir)
    coverage = summarize_block_coverage(cache_dir)
    gap_plan = build_gap_plan(tables, coverage)
    return tables, coverage, gap_plan


def summarize_tables(cache_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table, spec in TARGET_TABLES.items():
        files = table_files(cache_dir, table)
        columns = sample_columns(files)
        rows.append(
            {
                "table": table,
                "priority": spec["priority"],
                "role": spec["role"],
                "files": len(files),
                "rows": sum(csv_row_count(path) for path in files),
                "empty_files": sum(1 for path in files if csv_row_count(path) == 0),
                "columns_sample": ";".join(columns[:24]),
                "has_disclosure_date_col": any(col in columns for col in spec["date_candidates"]),
                "has_period_col": any(col in columns for col in spec["period_candidates"]) if spec["period_candidates"] else False,
                "has_stock_col": any(col in columns for col in spec["stock_candidates"]),
                "possible_row_cap_files": possible_row_cap_files(table, files, int(spec.get("row_cap_threshold", 0))),
                "source_type": "paid_standardized_local_cache",
                "token_read": False,
                "api_called": False,
            }
        )
    return pd.DataFrame(rows)


def summarize_block_coverage(cache_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table, spec in TARGET_TABLES.items():
        frame = read_table_for_coverage(cache_dir, table, spec)
        for block, (start, end) in TIME_BLOCKS.items():
            if frame.empty:
                rows.append(empty_coverage_row(table, spec, block))
                continue
            dates = pd.to_datetime(frame["available_date"], errors="coerce")
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            selected = frame.loc[mask].copy()
            stock_col = "ts_code" if "ts_code" in selected else "code" if "code" in selected else ""
            rows.append(
                {
                    "table": table,
                    "priority": spec["priority"],
                    "time_block": block,
                    "rows": int(len(selected)),
                    "unique_stocks": int(selected[stock_col].astype(str).nunique()) if stock_col else 0,
                    "coverage_dates": int(dates.loc[mask].dt.date.nunique()) if mask.any() else 0,
                    "rows_missing_disclosure_date": int(frame["available_date"].eq("").sum()) if block == list(TIME_BLOCKS.keys())[0] else 0,
                    "financial_title_rows": financial_title_count(selected) if table == "anns_d" else 0,
                    "source_type": "paid_standardized_local_cache",
                    "token_read": False,
                    "api_called": False,
                }
            )
    return pd.DataFrame(rows)


def build_gap_plan(tables: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, table_row in tables.iterrows():
        table = str(table_row["table"])
        spec = TARGET_TABLES[table]
        table_cov = coverage[coverage["table"].astype(str).eq(table)]
        total_rows = int(table_row["rows"])
        total_unique = int(table_cov["unique_stocks"].max()) if not table_cov.empty else 0
        h2026_rows = int(table_cov.loc[table_cov["time_block"].eq("H2026_1"), "rows"].sum()) if not table_cov.empty else 0
        prior_rows = int(table_cov.loc[~table_cov["time_block"].eq("H2026_1"), "rows"].sum()) if not table_cov.empty else 0
        if total_rows == 0:
            status = "missing_table"
            next_action = "add controlled cache support and fetch small smoke shard before large backfill"
        elif not bool(table_row["has_disclosure_date_col"]):
            status = "missing_disclosure_date_column"
            next_action = "do not use in walk-forward judgement until ann_date/f_ann_date/available_at is available"
        elif total_unique < int(spec["min_unique_stocks"]):
            status = "undercovered_unique_stocks"
            next_action = "backfill broader stock universe with rate limit and coverage audit"
        elif prior_rows == 0 or h2026_rows == 0:
            status = "block_imbalanced"
            next_action = "backfill missing half-year blocks before DS or promotion"
        elif int(table_row["possible_row_cap_files"]) > 0:
            status = "row_cap_risk"
            next_action = "split dense announcement shards by date/code or cross-check official source before scaling"
        else:
            status = "ready_for_feature_rebuild_audit"
            next_action = "rebuild financial_report_events/features and rerun as-of/nonquant audits"
        rows.append(
            {
                "table": table,
                "priority": spec["priority"],
                "status": status,
                "rows": total_rows,
                "max_unique_stocks_per_block": total_unique,
                "prior_rows": prior_rows,
                "h2026_rows": h2026_rows,
                "possible_row_cap_files": int(table_row["possible_row_cap_files"]),
                "next_action": next_action,
                "usable_in_agent_default": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["priority", "status", "table"]).reset_index(drop=True)


def table_files(cache_dir: Path, table: str) -> list[Path]:
    directory = Path(cache_dir) / "tables" / table
    if directory.exists():
        return sorted(directory.glob("*.csv"))
    flat = Path(cache_dir) / "tables" / f"{table}.csv"
    return [flat] if flat.exists() else []


def sample_columns(files: list[Path]) -> list[str]:
    seen: list[str] = []
    for path in files[:20]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
        except (OSError, UnicodeDecodeError):
            header = []
        for col in header:
            col = str(col).lstrip("\ufeff")
            if col and col not in seen:
                seen.append(col)
    return seen


def csv_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            count = sum(1 for _ in handle)
    except (OSError, UnicodeDecodeError):
        return 0
    return max(count - 1, 0)


def possible_row_cap_files(table: str, files: list[Path], threshold: int) -> int:
    if table != "anns_d" or threshold <= 0:
        return 0
    return sum(1 for path in files if csv_row_count(path) >= threshold)


def read_table_for_coverage(cache_dir: Path, table: str, spec: dict[str, Any]) -> pd.DataFrame:
    files = table_files(cache_dir, table)
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception:
            continue
        if frame.empty:
            continue
        frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
        columns = set(frame.columns)
        keep = [col for col in set(spec["stock_candidates"] + spec["date_candidates"] + spec["period_candidates"] + ["title", "code"]) if col in columns]
        if not keep:
            continue
        part_date = partition_date(path.name)
        small = frame[keep].copy()
        small["available_date"] = choose_first_date(small, spec["date_candidates"])
        if part_date:
            small.loc[small["available_date"].eq(""), "available_date"] = part_date
        frames.append(small)
    if not frames:
        return pd.DataFrame(columns=["available_date"])
    out = pd.concat(frames, ignore_index=True)
    if "ts_code" in out:
        out["ts_code"] = out["ts_code"].fillna("").astype(str)
    if "code" in out:
        out["code"] = out["code"].fillna("").astype(str).str.zfill(6)
    out["available_date"] = out["available_date"].fillna("").astype(str)
    return out


def choose_first_date(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series("", index=frame.index, dtype="object")
    for col in candidates:
        if col not in frame:
            continue
        dates = frame[col].map(clean_date_text)
        out = out.mask(out.eq("") & dates.ne(""), dates)
    return out


def clean_date_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def partition_date(name: str) -> str:
    digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
    for item in digits:
        if len(item) == 8:
            return clean_date_text(item)
    return ""


def empty_coverage_row(table: str, spec: dict[str, Any], block: str) -> dict[str, Any]:
    return {
        "table": table,
        "priority": spec["priority"],
        "time_block": block,
        "rows": 0,
        "unique_stocks": 0,
        "coverage_dates": 0,
        "rows_missing_disclosure_date": 0,
        "financial_title_rows": 0,
        "source_type": "paid_standardized_local_cache",
        "token_read": False,
        "api_called": False,
    }


def financial_title_count(frame: pd.DataFrame) -> int:
    if frame.empty or "title" not in frame:
        return 0
    titles = frame["title"].fillna("").astype(str)
    return int(titles.map(lambda value: any(keyword in value for keyword in FINANCIAL_KEYWORDS)).sum())


def write_outputs(prefix: str, tables: pd.DataFrame, coverage: pd.DataFrame, gap_plan: pd.DataFrame) -> dict[str, Path]:
    table_path = REPORT_DIR / f"{prefix}_tables.csv"
    coverage_path = REPORT_DIR / f"{prefix}_block_coverage.csv"
    gap_path = REPORT_DIR / f"{prefix}_gap_plan.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    tables.to_csv(table_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    gap_plan.to_csv(gap_path, index=False, encoding="utf-8-sig")
    write_report(report_path, tables, coverage, gap_plan)
    return {
        "tables": table_path,
        "coverage": coverage_path,
        "gap_plan": gap_path,
        "report": report_path,
    }


def write_report(path: Path, tables: pd.DataFrame, coverage: pd.DataFrame, gap_plan: pd.DataFrame) -> None:
    lines = [
        "# Financial Source Inventory v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "本轮只扫描本地离线缓存；不读取 token，不请求 Tushare，不调用 DeepSeek。",
        "",
        "## Key Readout",
        "",
        f"- target_tables: `{len(TARGET_TABLES)}`",
        f"- cached_tables_with_rows: `{int(tables['rows'].gt(0).sum())}`",
        f"- missing_tables: `{int(tables['rows'].eq(0).sum())}`",
        f"- tables_with_row_cap_risk: `{int(tables['possible_row_cap_files'].gt(0).sum())}`",
        "",
        "## Gap Plan",
        "",
        table(gap_plan),
        "",
        "## Table Inventory",
        "",
        table(tables),
        "",
        "## Block Coverage",
        "",
        table(coverage),
        "",
        "## Decision",
        "",
        "- 不把任何 inventory 结果直接作为正向 alpha。",
        "- 缺 `ann_date/f_ann_date/available_at` 的表不得进入 walk-forward 正负判断。",
        "- `forecast`、`express`、`fina_indicator`、`anns_d` 是下一轮 P0 补洞；`income/cashflow/balancesheet/fina_audit` 是 P1，用于更稳定的财务质量与风险 variance table。",
        "- 补完后必须重建 `financial_report_events.csv` / `financial_report_features.csv`，再跑 as-of window 与非量化正向确认审计。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
